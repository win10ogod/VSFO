# V-SiFu: Variational Signal-in-Full-flow

This repository implements **V-SiFu** – a brain-inspired, variational
transport–attention model for next-vector generation.  The code mirrors the
mathematical construction described in the accompanying research note:

1. **Variational workspace (VAE)** compresses the observable signal history.
2. **Variational Entropic Transport Attention (VETA)** aligns the source
   signal distribution with latent-conditioned target priors through an
   entropic optimal transport plan solved by Sinkhorn iterations.
3. **Next-vector synthesis** aggregates transported edge transforms and
   projects the resulting energy onto the reachable subspace, yielding a
   continuous semantic vector that can be decoded back to discrete nodes.

The implementation retains the brain-inspired static semantic mapping of
SiFu/BriLLM while decoupling attention computation from sequence length.

## Repository layout

| File / folder | Description |
| --- | --- |
| `model.py` | Modular PyTorch implementation of V-SiFu (VAE, VETA, energy projection). |
| `train.py` | Synthetic training loop illustrating the VSFO objective. |
| `infer_en.py`, `infer_zh.py` | Convenience scripts for running inference with trained checkpoints. |
| `run_en.sh`, `run_zh.sh` | Example commands for training on synthetic English/Chinese vocabularies. |
| `figs/` | Figures from the original BriLLM/SiFu description (for reference). |
| `vocab_*.json` | Example vocabularies used to instantiate the brain-style node graph. |

## Quick start

Install PyTorch first (`pip install torch`).  Then train a synthetic model:

```bash
bash run_en.sh
```

The script calls `train.py`, which generates synthetic signal flows using the
selected vocabulary, trains V-SiFu with the VSFO objective and stores the
checkpoint under `checkpoints/`.

Run inference with the trained weights:

```bash
python infer_en.py \
  --checkpoint checkpoints/vsifu_en.pt \
  --vocab vocab_wiki_4k_en.json
```

The inference script samples a synthetic context, performs variational
transport attention, and prints the top candidate nodes together with the
transport-plan mass for each source→target route.

## Training with custom data

`train.py` expects a dataset that yields the following tensors per sample:

- `history`: `(T, d)` tensor of historical signal vectors for the VAE.
- `source_indices`: `(k,)` long tensor identifying active source nodes.
- `source_signals`: `(k, d)` tensor containing the associated signal states.
- `candidate_indices`: `(m,)` long tensor with candidate target nodes.
- `target_vector`: `(d,)` target semantic vector.
- `target_index`: scalar long tensor referencing the ground-truth node.

`SyntheticSignalDataset` in `train.py` demonstrates the required structure.
Custom datasets can subclass `torch.utils.data.Dataset` to produce the same
fields and be plugged directly into the training loop.

## Architectural notes

- **Variational workspace:** `VariationalSignalEncoder` encodes the history
  via a GRU and optimises the ELBO.  The KL term corresponds to the
  information bottleneck constraint described in the paper.
- **VETA:** `SinkhornSolver` computes the entropic OT plan `Π` between the
  energy-derived source distribution and the latent-conditioned target prior.
- **Next-vector generation:** edge transforms `W_{ij}` are factorised through
  source/target node embeddings.  The transport-weighted sum is projected onto
  the span of reachable edge outputs to respect the SiFu interpretability
  constraint.
- **VSFO objective:** the loss unifies reconstruction, KL, path regularisation
  and prototype consistency, mirroring Eq. (9) of the note.

## Checkpoints

Training scripts save checkpoints as PyTorch dictionaries containing the model
state and the configuration used to create it.  Load them with
`torch.load(checkpoint)['model_state_dict']` and feed the resulting state into
`VSiFuModel.load_state_dict`.

## License

This repository follows the original BriLLM release terms.
