"""Training script for the V-SiFu model.

The script defaults to a synthetic curriculum that mimics the signal
statistics described in the paper.  Researchers can swap the synthetic
dataset with their own preprocessed tensors by implementing a dataset
that returns the same fields as :class:`SyntheticSignalDataset`.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import torch
from torch.utils.data import DataLoader, Dataset

from model import VSiFuModel


@dataclass
class Batch:
    history: torch.Tensor
    source_indices: torch.Tensor
    source_signals: torch.Tensor
    candidate_indices: torch.Tensor
    target_vector: torch.Tensor
    target_index: torch.Tensor


class SyntheticSignalDataset(Dataset):
    """Generate synthetic training data following the paper's abstractions."""

    def __init__(
        self,
        num_nodes: int,
        signal_dim: int,
        history_length: int,
        num_sources: int,
        num_candidates: int,
        dataset_size: int,
        noise_scale: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.signal_dim = signal_dim
        self.history_length = history_length
        self.num_sources = num_sources
        self.num_candidates = num_candidates
        self.dataset_size = dataset_size
        self.noise_scale = noise_scale

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.prototype_bank = torch.randn(num_nodes, signal_dim, generator=generator)
        self.generator = generator

    def __len__(self) -> int:
        return self.dataset_size

    def _sample_indices(self, count: int) -> torch.Tensor:
        return torch.randint(0, self.num_nodes, (count,), generator=self.generator)

    def _sample_noise(self, shape: Iterable[int]) -> torch.Tensor:
        return self.noise_scale * torch.randn(*shape, generator=self.generator)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:  # noqa: D401
        history_nodes = self._sample_indices(self.history_length)
        history = self.prototype_bank[history_nodes] + self._sample_noise((self.history_length, self.signal_dim))

        source_indices = self._sample_indices(self.num_sources)
        source_signals = self.prototype_bank[source_indices] + self._sample_noise((self.num_sources, self.signal_dim))

        candidate_indices = self._sample_indices(self.num_candidates)
        # Ensure the first candidate is the target for supervised training.
        target_index = candidate_indices[0]
        target_vector = self.prototype_bank[target_index] + self._sample_noise((self.signal_dim,))

        return {
            "history": history.float(),
            "source_indices": source_indices.long(),
            "source_signals": source_signals.float(),
            "candidate_indices": candidate_indices.long(),
            "target_vector": target_vector.float(),
            "target_index": target_index.long(),
        }


def collate_batch(batch: Iterable[Dict[str, torch.Tensor]]) -> Batch:
    history = torch.stack([item["history"] for item in batch])
    source_indices = torch.stack([item["source_indices"] for item in batch])
    source_signals = torch.stack([item["source_signals"] for item in batch])
    candidate_indices = torch.stack([item["candidate_indices"] for item in batch])
    target_vector = torch.stack([item["target_vector"] for item in batch])
    target_index = torch.stack([item["target_index"] for item in batch])
    return Batch(
        history=history,
        source_indices=source_indices,
        source_signals=source_signals,
        candidate_indices=candidate_indices,
        target_vector=target_vector,
        target_index=target_index,
    )


def load_vocab(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the V-SiFu model")
    parser.add_argument("--vocab", type=Path, required=True, help="Path to the vocabulary JSON file")
    parser.add_argument("--signal-dim", type=int, default=64, help="Dimensionality of node embeddings")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent workspace dimensionality")
    parser.add_argument("--history-length", type=int, default=6, help="Number of history steps for the VAE")
    parser.add_argument("--num-sources", type=int, default=5, help="Active source nodes per step")
    parser.add_argument("--num-candidates", type=int, default=6, help="Candidate target nodes per step")
    parser.add_argument("--num-samples", type=int, default=4096, help="Number of synthetic training samples")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=1.0, help="Weight of the KL term")
    parser.add_argument("--gamma", type=float, default=1e-3, help="Weight of the path regulariser")
    parser.add_argument("--tau", type=float, default=1e-2, help="Weight of the prototype penalty")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=Path, default=None, help="Optional path to store the trained weights")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_metrics(losses: Dict[str, float]) -> str:
    parts = [f"{key}={value:.4f}" for key, value in losses.items()]
    return ", ".join(parts)


def train() -> None:
    args = parse_args()
    set_seed(args.seed)

    vocab = load_vocab(args.vocab)
    num_nodes = len(vocab)

    dataset = SyntheticSignalDataset(
        num_nodes=num_nodes,
        signal_dim=args.signal_dim,
        history_length=args.history_length,
        num_sources=args.num_sources,
        num_candidates=args.num_candidates,
        dataset_size=args.num_samples,
        seed=args.seed,
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = VSiFuModel(
        num_nodes=num_nodes,
        signal_dim=args.signal_dim,
        latent_dim=args.latent_dim,
        beta=args.beta,
        gamma=args.gamma,
        tau=args.tau,
    )
    model.initialise_prototypes(dataset.prototype_bank)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        metrics = {"total": 0.0, "reconstruction": 0.0, "kl": 0.0, "path_reg": 0.0, "proto_reg": 0.0}
        for batch in dataloader:
            optimizer.zero_grad()
            history = batch.history.to(device)
            source_indices = batch.source_indices.to(device)
            source_signals = batch.source_signals.to(device)
            candidate_indices = batch.candidate_indices.to(device)
            target_vector = batch.target_vector.to(device)
            target_index = batch.target_index.to(device)

            output = model(
                history=history,
                source_indices=source_indices,
                source_signals=source_signals,
                candidate_indices=candidate_indices,
                target_vector=target_vector,
                target_index=target_index,
            )

            total_loss = output.losses["total"]
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            metrics["total"] += total_loss.item()
            metrics["kl"] += output.losses["kl"].item()
            if "reconstruction" in output.losses:
                metrics["reconstruction"] += output.losses["reconstruction"].item()
            if "path_reg" in output.losses:
                metrics["path_reg"] += output.losses["path_reg"].item()
            if "proto_reg" in output.losses:
                metrics["proto_reg"] += output.losses["proto_reg"].item()

        batch_count = len(dataloader)
        averaged = {key: value / batch_count for key, value in metrics.items() if value != 0.0}
        print(f"Epoch {epoch + 1}/{args.epochs} :: {format_metrics(averaged)}")

    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "config": vars(args)}, args.save_path)
        print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    train()
