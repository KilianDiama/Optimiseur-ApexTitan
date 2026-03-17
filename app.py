import torch
from torch.optim import Optimizer

class ApexTitanV15_Final_Edition(Optimizer):
    """
    ApexTitan V15 - L'optimum du second ordre Low-Rank.
    Optimisations : Correction de biais, Memory-Efficient, Stable Cholesky.
    """
    def __init__(self, params, lr=1e-3, beta=0.9, rank=128, 
                 update_freq=10, wd=0.01, eps=1e-6, alpha=0.1):
        
        defaults = dict(lr=lr, beta=beta, rank=rank, 
                        update_freq=update_freq, wd=wd, 
                        eps=eps, alpha=alpha)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # --- Initialisation Smart & Zero-Clone ---
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    
                    if p.ndim >= 2:
                        d_out = p.shape[0]
                        k = min(group['rank'], d_out)
                        # On initialise Q en float32 pour la précision de la base
                        state['Q'] = torch.nn.init.orthogonal_(
                            torch.empty(d_out, k, device=p.device, dtype=torch.float32)
                        )
                        state['eye'] = torch.eye(k, device=p.device, dtype=torch.float32)

                state['step'] += 1
                grad = p.grad
                
                # --- AdamW Style Weight Decay (In-place) ---
                if group['wd'] > 0:
                    p.mul_(1 - group['lr'] * group['wd'])

                # --- Momentum avec Correction de Biais ---
                exp_avg = state['exp_avg']
                exp_avg.mul_(group['beta']).add_(grad, alpha=1 - group['beta'])
                
                bias_correction = 1 - group['beta'] ** state['step']
                # Gradient redressé (Nesterov-like + Bias Correction)
                d_p = (grad + group['beta'] * exp_avg) / bias_correction

                # --- Préconditionneur Low-Rank (uniquement pour matrices > 1D) ---
                if p.ndim >= 2:
                    d_p = self._apply_preconditioner(d_p, state, group)

                # --- Update final ---
                p.add_(d_p, alpha=-group['lr'])

        return loss

    def _apply_preconditioner(self, d_p, state, group):
        orig_shape = d_p.shape
        # On travaille en float32 pour la décomposition
        g = d_p.view(orig_shape[0], -1).to(torch.float32)
        Q = state['Q']

        # --- Subspace Tracking (QR spectral) ---
        if state['step'] % group['update_freq'] == 0:
            # On projette le gradient pour rafraîchir la base Q
            # Approche type Power Method pour capturer les directions de variance max
            new_Q = torch.mm(g, torch.mm(g.t(), Q))
            Q.copy_(torch.linalg.qr(new_Q).Q)

        # --- Projection & Inversion Cholesky ---
        # On réduit la dimension : (Rank, d_out) x (d_out, d_in) -> (Rank, d_in)
        g_proj = torch.mm(Q.t(), g)
        
        # Matrice de courbure réduite (Rank x Rank)
        curvature = torch.mm(g_proj, g_proj.t())
        jitter = state['eye'] * (group['alpha'] + group['eps'])
        curvature.add_(jitter)

        try:
            # Résolution du système linéaire (L L^T x = g_proj)
            L = torch.linalg.cholesky(curvature)
            precond = torch.linalg.cholesky_solve(g_proj, L)
            
            # Reconstruction dans l'espace original
            d_p_res = torch.mm(Q, precond)
            return d_p_res.view(orig_shape).to(d_p.dtype)

        except RuntimeError:
            # Fallback de sécurité (si courbure singulière)
            return d_p
