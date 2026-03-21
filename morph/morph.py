"""
MORPH v2 — Multi-agent Online Rewiring through Plasticity-guided Hierarchy
===========================================================================
Extends v1 with four additional plasticity mechanisms:

  1. BCM Metaplasticity     — per-link sliding threshold θ_ij (BCM theory).
                              Strongly potentiated links decay faster;
                              quiet links are easier to strengthen.

  2. Reward-modulated       — delivery burst strengthens recently-active links
     Plasticity               (dopamine-gated LTP analogue). Fixes the credit-
                              assignment gap: MORPH now learns which pairs
                              *complete* deliveries, not just which co-assign.

  3. Predictive Formation   — optional hint matrix boosts link formation before
                              co-assignment occurs (anticipatory synaptogenesis).
                              Closes the cold-start gap vs Proximity baselines.

  4. Neuromodulation        — global performance signal η(t) scales the
                              effective learning rate and exploration threshold
                              (dopamine/ACh arousal analogue). System explores
                              more when delivery rate falls, consolidates when
                              performing well.

v1 behaviour is recovered exactly by setting:
  bcm_gain=0, reward_alpha=0, neuromod_gain=0, neuromod_explore=0, pred_boost=0

API change vs v1:
  step(obs, jaccard, cov)                          # v1
  step(obs, jaccard, cov, delivery=0, pred_jac=None)  # v2 (backwards-compatible)

  delivery  : int   — number of deliveries completed this step
  pred_jac  : (N,N) — optional anticipatory co-assignment hint matrix
"""
from morph.morph_env import MovingAgentEnv, FixedTopology
from scipy.linalg import eigvalsh
import numpy as np


class MORPH:
    def __init__(self, N,
                 # ── v1 parameters ──────────────────────────────────────────
                 alpha=0.15, beta=0.06, decay=0.96,
                 theta_form_start=0.85, theta_form_end=0.58,
                 theta_form_anneal=100,
                 theta_prune=0.02, target_deg_frac=0.15,
                 max_new=5, grace_steps=12, k_slow=5,
                 # ── v2: BCM metaplasticity ──────────────────────────────────
                 bcm_tau=0.95,      # smoothing for sliding threshold (0=no memory)
                 bcm_gain=0.5,      # how much extra decay for over-potentiated links
                 # ── v2: reward-modulated plasticity ────────────────────────
                 reward_alpha=0.08, # learning rate for delivery-burst updates
                 # ── v2: neuromodulation ─────────────────────────────────────
                 neuromod_gain=0.4,      # scales effective_alpha when performing well
                 neuromod_explore=0.10,  # reduces theta_form when underperforming
                 neuromod_ema=0.03,      # EMA smoothing for delivery rate (~33-step window)
                 expected_delivery_rate=0.07,  # baseline deliveries/step for eta signal
                 # ── v2: predictive formation ────────────────────────────────
                 pred_boost=0.5,    # how much hint_jac boosts MI for link formation
                 ):
        # Core dimensions
        self.N = N

        # v1 params
        self.alpha = alpha
        self.beta  = beta
        self.decay = decay
        self.tf_start   = theta_form_start
        self.tf_end     = theta_form_end
        self.tf_anneal  = theta_form_anneal
        self.tp         = theta_prune
        self.target_deg = target_deg_frac * (N - 1)
        self.max_new    = max_new
        self.grace      = grace_steps
        self.k_slow     = k_slow
        self.t          = 0

        # v2 params
        self.bcm_tau    = bcm_tau
        self.bcm_gain   = bcm_gain
        self.reward_alpha = reward_alpha
        self.neuromod_gain    = neuromod_gain
        self.neuromod_explore = neuromod_explore
        self.neuromod_ema     = neuromod_ema
        self.expected_rate    = expected_delivery_rate
        self.pred_boost       = pred_boost

        # ── State matrices ──────────────────────────────────────────────────
        self.W        = np.zeros((N, N))   # synaptic weights
        self.A        = np.zeros((N, N))   # adjacency (structural links)
        self.link_age = np.zeros((N, N))   # age of each link (steps)
        self.H        = np.ones(N) * target_deg_frac   # homeostatic targets

        # v2 state
        self.theta_bcm    = np.zeros((N, N))   # BCM sliding threshold per link
        self.jac_last     = np.zeros((N, N))   # jaccard from previous step
        self.delivery_ema = expected_delivery_rate  # running delivery rate

        # ── History ─────────────────────────────────────────────────────────
        self.cov_h    = []
        self.ne_h     = []
        self.sg_h     = []
        self.prune_h  = []
        self.form_h   = []
        self.W_mean_h = []
        # v2 history
        self.eta_h    = []   # neuromodulation signal over time
        self.bcm_h    = []   # mean BCM threshold over active links

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def theta_form(self):
        """Base annealed formation threshold (further adjusted by neuromodulation)."""
        p = min(self.t / self.tf_anneal, 1.0)
        return self.tf_start - (self.tf_start - self.tf_end) * p

    # ── Internal utilities ──────────────────────────────────────────────────

    def _mi(self, obs):
        """Normalised cross-correlation as MI proxy."""
        oc = obs - obs.mean(axis=1, keepdims=True)
        on = oc / (oc.std(axis=1, keepdims=True) + 1e-9)
        MI = np.abs(on @ on.T) / obs.shape[1]
        np.fill_diagonal(MI, 0)
        return MI

    def _sg(self):
        """Fiedler value (spectral gap) of normalised Laplacian."""
        d = self.A.sum(axis=1)
        if d.sum() == 0:
            return 0.0
        Di = np.diag(1 / np.sqrt(np.maximum(d, 1e-9)))
        L  = np.eye(self.N) - Di @ self.A @ Di
        try:
            return float(eigvalsh(L, subset_by_index=[0, 1])[1])
        except Exception:
            return 0.0

    def get_jaccard(self, env):
        """Compute pairwise Jaccard task-coverage overlap (abstract env)."""
        d  = env.get_obs()
        ir = (d < env.jr).astype(float)
        inter = ir @ ir.T
        ni    = ir.sum(axis=1)
        union = ni[:, None] + ni[None, :] - inter
        return np.where(union > 0, inter / union, 0.0)

    # ── Main update ─────────────────────────────────────────────────────────

    def step(self, obs, jaccard, cov, delivery=0, pred_jac=None):
        """
        One plasticity step.

        Parameters
        ----------
        obs      : (N, D) observation matrix
        jaccard  : (N, N) task co-assignment overlap (or Jaccard)
        cov      : float  coverage / reward signal (scalar)
        delivery : int    deliveries completed this step (v2 reward signal)
        pred_jac : (N, N) optional anticipatory hint matrix (v2 predictive formation)
        """
        self.t += 1

        # ── 1. Neuromodulation signal ──────────────────────────────────────
        self.delivery_ema = (self.neuromod_ema * delivery
                             + (1 - self.neuromod_ema) * self.delivery_ema)
        # η: normalised deviation from expected rate
        eta = ((self.delivery_ema - self.expected_rate)
               / (self.expected_rate + 1e-9))
        eta = float(np.clip(eta, -1.0, 1.0))

        # Scale learning rate up when performing well
        effective_alpha = self.alpha * (1.0 + self.neuromod_gain * max(eta, 0.0))
        # Lower formation threshold when underperforming (explore new links)
        theta_form_adj = self.theta_form - self.neuromod_explore * max(-eta, 0.0)
        theta_form_adj = max(theta_form_adj, self.tf_end * 0.8)  # floor

        # ── 2. BCM-modulated synaptic decay ───────────────────────────────
        # ratio_ij = W_ij / theta_ij: >1 → over-potentiated → accelerate decay
        ratio = self.W / (self.theta_bcm + 1e-9)
        # Extra decay for links above their own history; zero below
        extra_decay = self.bcm_gain * np.clip(ratio - 1.0, 0.0, 3.0) * (1.0 - self.decay)
        effective_retention = np.clip(self.decay - extra_decay, 0.5, 1.0)
        self.W = self.W * effective_retention

        # ── 3. Hebbian synaptic potentiation ─────────────────────────────
        self.W += effective_alpha * jaccard * cov * self.A

        # ── 4. Reward-modulated plasticity (delivery burst) ──────────────
        # jac_last captures which links were active when the delivery was earned
        if delivery > 0:
            self.W += self.reward_alpha * delivery * self.jac_last * self.A

        # Store jaccard for next step's reward attribution
        self.jac_last = jaccard.copy()

        # Clip and zero diagonal
        np.clip(self.W, 0.0, 1.0, out=self.W)
        np.fill_diagonal(self.W, 0.0)

        # ── 5. Update BCM sliding threshold ───────────────────────────────
        self.theta_bcm = (self.bcm_tau * self.theta_bcm
                          + (1.0 - self.bcm_tau) * self.W)
        np.fill_diagonal(self.theta_bcm, 0.0)

        # ── 6. Homeostatic plasticity ─────────────────────────────────────
        deg = self.A.sum(axis=1)
        self.H += self.beta * (self.target_deg - deg) / (self.N - 1)
        np.clip(self.H, 0.0, 1.0, out=self.H)

        # ── 7. Structural plasticity (every k_slow steps) ─────────────────
        n_formed = n_pruned = 0
        if self.t % self.k_slow == 0:
            n_formed, n_pruned = self._structural_update(
                obs, jaccard, pred_jac, theta_form_adj)

        # ── 8. Link age tracking ──────────────────────────────────────────
        self.link_age = np.where(self.A > 0, self.link_age + 1, 0.0)

        # ── 9. Record history ─────────────────────────────────────────────
        ne = int(self.A.sum() // 2)
        wm = float(self.W[self.A > 0].mean()) if ne > 0 else 0.0
        self.cov_h.append(float(cov))
        self.ne_h.append(ne)
        self.sg_h.append(self._sg())
        self.prune_h.append(n_pruned)
        self.form_h.append(n_formed)
        self.W_mean_h.append(wm)
        self.eta_h.append(eta)
        bcm_active = float(self.theta_bcm[self.A > 0].mean()) if ne > 0 else 0.0
        self.bcm_h.append(bcm_active)

    def _structural_update(self, obs, jaccard, pred_jac, theta_form_adj):
        """
        Structural plasticity: form high-MI / high-Jaccard links, prune weak ones.
        Predictive formation: pred_jac boosts MI for anticipated co-assignments.
        """
        MI = self._mi(obs)

        # Blend in anticipatory signal if provided
        if pred_jac is not None and self.pred_boost > 0:
            MI = MI + self.pred_boost * pred_jac
            MI = np.clip(MI, 0.0, 1.0)
            np.fill_diagonal(MI, 0.0)

        # ── Pruning ───────────────────────────────────────────────────────
        prune_mask = (
            (self.A > 0)
            & (self.W < self.tp)
            & (self.link_age > self.grace)
        )
        n_pruned = int(prune_mask.sum() // 2)
        self.A[prune_mask] = 0.0
        self.W[prune_mask] = 0.0
        self.theta_bcm[prune_mask] = 0.0
        self.link_age[prune_mask] = 0.0

        # ── Formation ─────────────────────────────────────────────────────
        # Combined signal: MI + homeostatic gain per agent pair
        H_gain = self.H[:, None] + self.H[None, :]
        score  = MI * (1.0 + H_gain)

        # Candidate pairs: not yet linked, not self
        candidates = (self.A == 0) & ~np.eye(self.N, dtype=bool)
        score[~candidates] = -1.0

        n_formed = 0
        for _ in range(self.max_new):
            idx = int(np.argmax(score))
            i, j = divmod(idx, self.N)
            if score[i, j] < theta_form_adj:
                break
            # Form symmetric link
            self.A[i, j] = self.A[j, i] = 1.0
            self.link_age[i, j] = self.link_age[j, i] = 0.0
            score[i, :] = score[:, i] = -1.0
            score[j, :] = score[:, j] = -1.0
            n_formed += 1

        np.fill_diagonal(self.A, 0.0)
        return n_formed, n_pruned
