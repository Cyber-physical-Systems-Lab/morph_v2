"""
MORPH environment and baseline classes.

MovingAgentEnv  — N agents that move toward uncovered tasks
FixedTopology   — static baselines (full, ring, random, none)
MORPHFull       — MORPH variant with TD-error delta_ij synaptic rule
                  (used for ablation / comparison in appendix)
"""
import numpy as np, networkx as nx, warnings
from scipy.linalg import eigvalsh
warnings.filterwarnings('ignore')


class MovingAgentEnv:
    """
    N agents that move toward uncovered tasks (speed=speed).
    After task shift, agents near old locations lose coordination value.
    """
    def __init__(self, N=60, M=80, grid_size=10, n_clusters=4,
                 solo_r=1.5, joint_r=2.8, speed=0.12):
        self.N = N; self.M = M; self.gs = grid_size; self.nc = n_clusters
        self.sr = solo_r; self.jr = joint_r; self.speed = speed
        self.agent_pos = np.random.rand(N, 2) * grid_size
        self.reset_tasks()
        self.prev_cov = 0.3

    def reset_tasks(self):
        c = np.random.rand(self.nc, 2) * self.gs
        self.task_pos = np.clip(np.vstack(
            [c[i] + np.random.randn(self.M // self.nc, 2) * 1.1
             for i in range(self.nc)]), 0, self.gs)[:self.M]

    def move_agents(self, A):
        """Move each agent toward nearest uncovered task."""
        dists = self.get_obs()
        ir = (dists < self.jr).astype(float)
        joint = np.einsum('ik,ij,jk->k', ir, A, ir) > 0
        solo = (dists < self.sr).any(axis=0)
        uncovered = ~(joint | solo)
        if uncovered.any():
            uncov_pos = self.task_pos[uncovered]
            for i in range(self.N):
                d2u = np.linalg.norm(uncov_pos - self.agent_pos[i], axis=1)
                nearest = uncov_pos[np.argmin(d2u)]
                direction = nearest - self.agent_pos[i]
                dist = np.linalg.norm(direction) + 1e-9
                if dist > 0.1:
                    self.agent_pos[i] += self.speed * (direction / dist)
        self.agent_pos = np.clip(self.agent_pos, 0, self.gs)

    def get_obs(self):
        return np.linalg.norm(
            self.agent_pos[:, None, :] - self.task_pos[None, :, :], axis=2)

    def compute_joint_coverage(self, A):
        d = self.get_obs(); solo = (d < self.sr).any(axis=0)
        ir = (d < self.jr).astype(float)
        joint = np.einsum('ik,ij,jk->k', ir, A, ir) > 0
        cov = (joint.astype(float) + solo * 0.3 * (~joint)).mean()
        return cov

    def get_delta_ij(self, A, cov):
        """TD-error proxy per pair: delta_ij = jaccard_ij * delta_cov."""
        d = self.get_obs(); ir = (d < self.jr).astype(float)
        inter = ir @ ir.T; ni = ir.sum(axis=1)
        union = ni[:, None] + ni[None, :] - inter
        jaccard = np.where(union > 0, inter / union, 0.)
        delta_cov = cov - self.prev_cov
        self.prev_cov = cov
        return jaccard * delta_cov

    def shift_tasks(self): self.reset_tasks()


class FixedTopology:
    """Static coordination topology baselines."""
    def __init__(self, N, topology='full'):
        self.N = N
        if topology == 'full':
            self.A = np.ones((N, N)) - np.eye(N)
        elif topology == 'ring':
            self.A = np.zeros((N, N))
            for i in range(N): self.A[i, (i+1) % N] = self.A[(i+1) % N, i] = 1
        elif topology == 'random':
            self.A = nx.to_numpy_array(nx.erdos_renyi_graph(N, 0.12, seed=42))
        else:
            self.A = np.zeros((N, N))
        self.cov_h = []

    def step(self, cov): self.cov_h.append(cov)


class MORPHFull:
    """
    MORPH variant using TD-error (delta_ij) synaptic rule instead of Jaccard*cov.
    Synaptic rule: w_ij += alpha * delta_ij * r(t)
    Used for ablation comparisons in appendix.
    """
    def __init__(self, N, alpha=0.15, beta=0.06,
                 decay=0.93,
                 theta_form_start=0.88, theta_form_end=0.60, theta_form_anneal=100,
                 theta_prune=0.01,
                 target_deg_frac=0.12,
                 max_new=5, grace_steps=12, k_slow=5):
        self.N = N; self.alpha = alpha; self.beta = beta; self.decay = decay
        self.tf_start = theta_form_start; self.tf_end = theta_form_end
        self.tf_anneal = theta_form_anneal; self.tp = theta_prune
        self.target_deg = target_deg_frac * (N - 1)
        self.max_new = max_new; self.grace = grace_steps; self.k_slow = k_slow; self.t = 0
        self.W = np.zeros((N, N)); self.A = np.zeros((N, N))
        self.link_age = np.zeros((N, N)); self.H = np.ones(N) * target_deg_frac
        self.cov_h = []; self.deg_h = []; self.ne_h = []; self.sg_h = []
        self.prune_h = []; self.form_h = []

    @property
    def theta_form(self):
        p = min(self.t / self.tf_anneal, 1.0)
        return self.tf_start - (self.tf_start - self.tf_end) * p

    def _mi(self, obs):
        oc = obs - obs.mean(axis=1, keepdims=True)
        on = oc / (oc.std(axis=1, keepdims=True) + 1e-9)
        MI = np.abs(on @ on.T) / obs.shape[1]
        np.fill_diagonal(MI, 0); return MI

    def _sg(self):
        d = self.A.sum(axis=1)
        if d.sum() == 0: return 0.0
        Di = np.diag(1 / np.sqrt(np.maximum(d, 1e-9)))
        L = np.eye(self.N) - Di @ self.A @ Di
        try: return float(eigvalsh(L, subset_by_index=[0, 1])[1])
        except: return 0.0

    def step(self, obs, delta_ij, cov):
        # Synaptic: decay + TD-error update
        self.W *= self.decay
        self.W += self.alpha * delta_ij * self.A
        self.W = np.clip(self.W, 0, 1); self.W = (self.W + self.W.T) / 2
        self.link_age = np.where(self.A == 1, self.link_age + 1, 0)
        n_formed = n_pruned = 0
        if self.t % self.k_slow == 0:
            MI = self._mi(obs)
            cdeg = self.A.sum(axis=1)
            hg = (cdeg < self.target_deg * 1.8); cf = hg[:, None] & hg[None, :]
            form = (MI > self.theta_form) & (self.A == 0) & cf
            np.fill_diagonal(form, False); form = form & form.T
            if form.any():
                ci = np.argwhere(np.triu(form, 1))
                mv = MI[ci[:, 0], ci[:, 1]]; k = min(self.max_new, len(ci))
                for i, j in ci[np.argsort(mv)[-k:]]:
                    self.A[i, j] = self.A[j, i] = 1.
                    self.link_age[i, j] = self.link_age[j, i] = 0
                    n_formed += 1
            pg = (self.link_age >= self.grace)
            pr = (self.A == 1) & (self.W < self.tp) & pg
            n_pruned = int(pr.sum() // 2)
            self.A -= pr.astype(float)
            self.A = np.clip(self.A, 0, 1); np.fill_diagonal(self.A, 0)
        self.H += self.beta * (self.target_deg - self.A.sum(axis=1))
        self.H = np.clip(self.H, 0, self.N - 1)
        self.cov_h.append(cov); self.deg_h.append(self.A.sum(axis=1).copy())
        self.ne_h.append(int(self.A.sum() // 2))
        self.prune_h.append(n_pruned); self.form_h.append(n_formed)
        if self.t % 10 == 0: self.sg_h.append(self._sg())
        self.t += 1


# Backward-compatibility aliases
NeuroSYSFinal = MORPHFull
