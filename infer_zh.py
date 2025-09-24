"""Inference helper for the Chinese V-SiFu checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from model import VSiFuModel


def load_vocab(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_inverse_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    return {idx: token for token, idx in vocab.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V-SiFu inference on synthetic Chinese prompts")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--signal-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--history-length", type=int, default=6)
    parser.add_argument("--num-sources", type=int, default=5)
    parser.add_argument("--num-candidates", type=int, default=6)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def sample_context(
    model: VSiFuModel,
    history_length: int,
    num_sources: int,
    num_candidates: int,
    generator: torch.Generator,
) -> Dict[str, torch.Tensor]:
    device = model.node_prototypes.weight.device
    num_nodes = model.num_nodes
    prototypes = model.node_prototypes.weight.detach()

    history_indices = torch.randint(0, num_nodes, (history_length,), generator=generator, device=device)
    source_indices = torch.randint(0, num_nodes, (num_sources,), generator=generator, device=device)
    candidate_indices = torch.randint(0, num_nodes, (num_candidates,), generator=generator, device=device)

    noise = lambda *shape: 0.05 * torch.randn(*shape, generator=generator, device=device)

    history = prototypes[history_indices] + noise(history_length, model.signal_dim)
    source_signals = prototypes[source_indices] + noise(num_sources, model.signal_dim)

    return {
        "history": history.unsqueeze(0),
        "source_indices": source_indices.unsqueeze(0),
        "source_signals": source_signals.unsqueeze(0),
        "candidate_indices": candidate_indices.unsqueeze(0),
    }


def main() -> None:
    args = parse_args()
    vocab = load_vocab(args.vocab)
    inv_vocab = build_inverse_vocab(vocab)
    num_nodes = len(vocab)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = VSiFuModel(
        num_nodes=num_nodes,
        signal_dim=args.signal_dim,
        latent_dim=args.latent_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    with torch.no_grad():
        batch = sample_context(
            model,
            history_length=args.history_length,
            num_sources=args.num_sources,
            num_candidates=args.num_candidates,
            generator=generator,
        )
        output = model.predict_next_vector(**batch)
        prediction = output.prediction
        indices, scores = model.vector_to_tokens(prediction, top_k=args.top_k)

    print("候選節點 (節點 :: 餘弦相似度):")
    top_indices = indices[0].cpu().tolist()
    top_scores = scores[0].cpu().tolist()
    for idx, score in zip(top_indices, top_scores):
        token = inv_vocab.get(idx, f"<unk:{idx}>")
        print(f"  {token:<20} :: {score:.4f}")

    transport = output.transport_plan[0].cpu()
    source_ids = batch["source_indices"].squeeze(0).cpu().tolist()
    target_ids = batch["candidate_indices"].squeeze(0).cpu().tolist()
    print("\n傳輸計畫質量 (來源 -> 目標 :: 質量):")
    for i, src in enumerate(source_ids):
        for j, tgt in enumerate(target_ids):
            mass = transport[i, j].item()
            if mass > 1e-3:
                src_token = inv_vocab.get(src, str(src))
                tgt_token = inv_vocab.get(tgt, str(tgt))
                print(f"  {src_token} -> {tgt_token} :: {mass:.4f}")


if __name__ == "__main__":
    main()
