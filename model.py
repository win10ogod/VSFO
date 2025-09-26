"""V-SiFu: Variational signal-in-full-flow model.

This module implements the architecture described in the accompanying
research note:

* a brain-inspired node graph with static semantic prototypes,
* a variational workspace (VAE) that compresses the visible signal history,
* variational entropic transport attention (VETA) solved via Sinkhorn,
* next-vector generation realised as energy-projected transport of signals.

The code is intentionally modular so that individual components (VAE,
Sinkhorn solver, node/edge parameterisations) can be swapped with research
variants.  The forward pass returns both the predicted next semantic vector
and a dictionary of loss terms corresponding to Eq. (9) of the note.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VSiFuForwardOutput:
    """Container for the forward pass outputs."""

    prediction: torch.Tensor
    transport_plan: torch.Tensor
    source_distribution: torch.Tensor
    target_distribution: torch.Tensor
    latent_sample: torch.Tensor
    losses: Dict[str, torch.Tensor]


class SinkhornSolver(nn.Module):
    """Differentiable Sinkhorn solver for entropic optimal transport.

    Args:
        epsilon: Entropic regularisation strength (``\varepsilon`` in the paper).
        max_iters: Maximum number of Sinkhorn iterations.
        tolerance: Stopping criterion on the log-scaling updates.
    """

    def __init__(self, epsilon: float = 0.1, max_iters: int = 100, tolerance: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.tolerance = tolerance

    def forward(self, cost: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Solve the entropic OT problem.

        Args:
            cost: Quadratic cost tensor of shape ``(batch, n_src, n_tgt)``.
            mu: Source distribution over active nodes ``(batch, n_src)``.
            nu: Target prior distribution ``(batch, n_tgt)``.

        Returns:
            transport_plan: Optimal transport plan ``\Pi``.
            log_transport: Logarithm of the optimal plan (useful for entropy).
        """

        if cost.ndim != 3:
            raise ValueError("cost must be a 3-D tensor (batch, src, tgt)")
        if mu.ndim != 2 or nu.ndim != 2:
            raise ValueError("mu and nu must be 2-D tensors")

        # Avoid log(0) by enforcing a numerical floor.
        tiny = torch.finfo(cost.dtype).tiny
        mu = mu.clamp_min(tiny)
        nu = nu.clamp_min(tiny)
        cost = cost / self.epsilon

        log_K = -cost  # (batch, src, tgt)
        log_mu = mu.log()
        log_nu = nu.log()

        log_u = torch.zeros_like(mu)
        log_v = torch.zeros_like(nu)

        for _ in range(self.max_iters):
            log_u_prev = log_u
            log_u = log_mu - torch.logsumexp(log_K + log_v.unsqueeze(-2), dim=-1)
            log_v = log_nu - torch.logsumexp(log_K.transpose(-2, -1) + log_u.unsqueeze(-2), dim=-1)

            if self.tolerance > 0:
                max_update = (log_u - log_u_prev).abs().max()
                if torch.isnan(max_update):
                    break
                if max_update < self.tolerance:
                    break

        log_pi = log_u.unsqueeze(-1) + log_K + log_v.unsqueeze(-2)
        transport = torch.exp(log_pi)
        return transport, log_pi


class VariationalSignalEncoder(nn.Module):
    """Encode a sequence of signal tensors into a latent workspace."""

    def __init__(self, signal_dim: int, latent_dim: int):
        super().__init__()
        self.gru = nn.GRU(signal_dim, signal_dim, batch_first=True)
        self.to_mu = nn.Linear(signal_dim, latent_dim)
        self.to_logvar = nn.Linear(signal_dim, latent_dim)

    def forward(self, history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if history.ndim != 3:
            raise ValueError("history must be a tensor of shape (batch, steps, dim)")
        _, h_n = self.gru(history)
        summary = h_n[-1]
        mu = self.to_mu(summary)
        logvar = self.to_logvar(summary)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class VSiFuModel(nn.Module):
    """Variational Signal-in-Full-flow model implementing VETA attention."""

    def __init__(
        self,
        num_nodes: int,
        signal_dim: int,
        latent_dim: int,
        energy_temperature: float = 1.0,
        sinkhorn_epsilon: float = 0.1,
        sinkhorn_iters: int = 100,
        sinkhorn_tolerance: float = 1e-6,
        beta: float = 1.0,
        gamma: float = 1e-3,
        tau: float = 1e-2,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.signal_dim = signal_dim
        self.latent_dim = latent_dim
        self.energy_temperature = energy_temperature
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

        # Node prototypes act as semantic anchors.
        self.node_prototypes = nn.Embedding(num_nodes, signal_dim)

        # Components of the cost matrix C_ij(z).
        self.cost_A = nn.Linear(signal_dim, signal_dim, bias=False)
        self.cost_B = nn.Linear(signal_dim, signal_dim, bias=False)
        self.cost_U = nn.Linear(latent_dim, signal_dim, bias=False)

        # Bilinear psi(p, z) = (U_psi p) · z for the target prior.
        self.psi_projection = nn.Linear(signal_dim, latent_dim, bias=False)

        # Edge transforms W_{ij} and biases b_{ij} via node factorisation.
        self.source_linear = nn.Embedding(num_nodes, signal_dim * signal_dim)
        self.target_linear = nn.Embedding(num_nodes, signal_dim * signal_dim)
        self.source_bias = nn.Embedding(num_nodes, signal_dim)
        self.target_bias = nn.Embedding(num_nodes, signal_dim)

        self.encoder = VariationalSignalEncoder(signal_dim, latent_dim)
        self.sinkhorn = SinkhornSolver(
            epsilon=sinkhorn_epsilon, max_iters=sinkhorn_iters, tolerance=sinkhorn_tolerance
        )

        # Observation noise of the Gaussian decoder.
        self.log_sigma = nn.Parameter(torch.zeros(1))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.node_prototypes.weight)
        for emb in (self.source_linear, self.target_linear):
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        for emb in (self.source_bias, self.target_bias):
            nn.init.zeros_(emb.weight)
        for linear in (self.cost_A, self.cost_B):
            nn.init.orthogonal_(linear.weight)
        nn.init.normal_(self.cost_U.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.psi_projection.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # Helper functions for the main modules
    # ------------------------------------------------------------------
    def initialise_prototypes(self, prototypes: torch.Tensor) -> None:
        """Manually set the node prototypes."""

        if prototypes.shape != self.node_prototypes.weight.shape:
            raise ValueError(
                "Prototype tensor must have shape "
                f"{self.node_prototypes.weight.shape}, got {prototypes.shape}."
            )
        with torch.no_grad():
            self.node_prototypes.weight.copy_(prototypes)

    def compute_source_distribution(self, signals: torch.Tensor) -> torch.Tensor:
        energy = torch.linalg.norm(signals, dim=-1)
        logits = self.energy_temperature * energy
        return torch.softmax(logits, dim=-1)

    def compute_target_prior(self, prototypes: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        projected = self.psi_projection(prototypes)
        logits = (projected * latent.unsqueeze(1)).sum(-1)
        return torch.softmax(logits, dim=-1)

    def compute_cost_matrix(
        self,
        source_signals: torch.Tensor,
        target_prototypes: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        A_s = self.cost_A(source_signals)
        B_p = self.cost_B(target_prototypes)
        U_z = self.cost_U(latent).unsqueeze(1).unsqueeze(1)
        diff = A_s.unsqueeze(2) - B_p.unsqueeze(1) - U_z
        cost = (diff**2).sum(-1)
        return cost

    def _edge_linear_transforms(
        self,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
        source_signals: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, n_src, _ = source_signals.shape
        n_tgt = target_indices.size(1)

        src_linear = self.source_linear(source_indices).view(batch, n_src, self.signal_dim, self.signal_dim)
        tgt_linear = self.target_linear(target_indices).view(batch, n_tgt, self.signal_dim, self.signal_dim)

        # Effective matrix is src_linear @ tgt_linear^T
        tgt_transposed = tgt_linear.transpose(-2, -1)
        combined = torch.einsum("bsij,btjk->bstik", src_linear, tgt_transposed)

        # Apply to source signals.
        intermediate = torch.einsum("bsd,bsde->bse", source_signals, src_linear)
        edge_linear = torch.einsum("bsi,btij->bstj", intermediate, tgt_transposed)

        bias_src = self.source_bias(source_indices)
        bias_tgt = self.target_bias(target_indices)
        edge_bias = bias_src.unsqueeze(2) + bias_tgt.unsqueeze(1)
        return edge_linear, edge_bias, combined

    def _project_to_span(self, basis: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        # basis: (batch, K, dim); vector: (batch, dim)
        batch, K, dim = basis.shape
        if K == 0:
            return vector
        BT = basis.transpose(1, 2)
        gram = torch.matmul(basis, BT)
        gram_inv = torch.linalg.pinv(gram)
        proj_coeff = torch.matmul(gram_inv, torch.matmul(basis, vector.unsqueeze(-1)))
        projected = torch.matmul(BT, proj_coeff).squeeze(-1)
        return projected

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(
        self,
        history: torch.Tensor,
        source_indices: torch.Tensor,
        source_signals: torch.Tensor,
        candidate_indices: torch.Tensor,
        target_vector: Optional[torch.Tensor] = None,
        target_index: Optional[torch.Tensor] = None,
    ) -> VSiFuForwardOutput:
        if history.ndim != 3:
            raise ValueError("history must be of shape (batch, steps, dim)")
        if source_indices.ndim != 2 or candidate_indices.ndim != 2:
            raise ValueError("Indices must be of shape (batch, items)")
        if source_signals.ndim != 3:
            raise ValueError("source_signals must be of shape (batch, items, dim)")

        z, mu_latent, logvar_latent = self.encoder(history)
        source_distribution = self.compute_source_distribution(source_signals)

        target_prototypes = self.node_prototypes(candidate_indices)
        target_distribution = self.compute_target_prior(target_prototypes, z)

        cost = self.compute_cost_matrix(source_signals, target_prototypes, z)
        transport_plan, log_transport = self.sinkhorn(cost, source_distribution, target_distribution)

        edge_linear, edge_bias, combined_matrix = self._edge_linear_transforms(
            source_indices, candidate_indices, source_signals
        )
        weighted_paths = transport_plan.unsqueeze(-1) * (edge_linear + edge_bias)
        tilde_prediction = weighted_paths.sum(dim=(1, 2))

        basis = edge_linear.reshape(edge_linear.size(0), -1, self.signal_dim)
        prediction = self._project_to_span(basis, tilde_prediction)

        losses: Dict[str, torch.Tensor] = {}
        loss = torch.zeros(prediction.size(0), device=prediction.device)

        # VAE KL term.
        kl = -0.5 * torch.sum(1 + logvar_latent - mu_latent.pow(2) - logvar_latent.exp(), dim=-1)
        losses["kl"] = kl.mean()
        loss = loss + self.beta * kl

        if target_vector is not None:
            if target_vector.ndim != 2:
                raise ValueError("target_vector must be of shape (batch, dim)")
            sigma_sq = torch.exp(2 * self.log_sigma)
            diff = prediction - target_vector
            recon = 0.5 * diff.pow(2).sum(-1) / sigma_sq + 0.5 * self.signal_dim * torch.log(sigma_sq)
            losses["reconstruction"] = recon.mean()
            loss = loss + recon

        # Path regulariser.
        frob_norm = combined_matrix.pow(2).sum(dim=(-2, -1))
        path_reg = (transport_plan * frob_norm).sum(dim=(1, 2))
        losses["path_reg"] = path_reg.mean()
        loss = loss + self.gamma * path_reg

        if target_index is not None:
            if target_index.ndim != 1:
                raise ValueError("target_index must be a 1-D tensor")
            target_proto = self.node_prototypes(target_index)
            proto_penalty = (prediction - target_proto).pow(2).sum(-1)
            losses["proto_reg"] = proto_penalty.mean()
            loss = loss + self.tau * proto_penalty

        losses["total"] = loss.mean()

        return VSiFuForwardOutput(
            prediction=prediction,
            transport_plan=transport_plan,
            source_distribution=source_distribution,
            target_distribution=target_distribution,
            latent_sample=z,
            losses=losses,
        )

    @torch.no_grad()
    def predict_next_vector(
        self,
        history: torch.Tensor,
        source_indices: torch.Tensor,
        source_signals: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> VSiFuForwardOutput:
        return self.forward(
            history=history,
            source_indices=source_indices,
            source_signals=source_signals,
            candidate_indices=candidate_indices,
            target_vector=None,
            target_index=None,
        )

    @torch.no_grad()
    def vector_to_tokens(self, vectors: torch.Tensor, top_k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        prototypes = self.node_prototypes.weight
        sim = F.cosine_similarity(vectors.unsqueeze(1), prototypes.unsqueeze(0), dim=-1)
        values, indices = torch.topk(sim, k=top_k, dim=-1)
        return indices, values
