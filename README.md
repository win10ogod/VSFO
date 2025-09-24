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

## Quick start (Linux/macOS)

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

## Windows setup and usage

The project runs on Windows as long as Python and PyTorch are available.  The
steps below assume PowerShell (Windows Terminal works the same) and Python
3.9+ installed from [python.org](https://www.python.org) with the *Add Python to
PATH* option enabled.

### 1. Prepare a virtual environment

```powershell
cd C:\path\to\VSFO
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Install the PyTorch wheel that matches your hardware; CPU-only works on any
# machine.  Replace the command below with the one recommended on
# https://pytorch.org/get-started/locally/ if you need CUDA.
pip install torch
```

### 2. Train the synthetic English model

Run `train.py` directly; PowerShell uses the backtick (`` ` ``) for line
continuations, as shown below:

```powershell
python train.py --vocab .\vocab_wiki_4k_en.json `
  --signal-dim 64 `
  --latent-dim 32 `
  --num-samples 4096 `
  --epochs 5 `
  --batch-size 32 `
  --save-path .\checkpoints\vsifu_en.pt
```

The script creates the `checkpoints` directory if it does not exist and saves
the trained weights.  Adjust flags such as `--device cuda` (if a CUDA-enabled
PyTorch build is installed) or `--history-length`, `--num-sources`, and
`--num-candidates` to explore different curriculum settings.

### 3. Run inference on Windows

Use the trained checkpoint to generate next-vector predictions:

```powershell
python infer_en.py --checkpoint .\checkpoints\vsifu_en.pt --vocab .\vocab_wiki_4k_en.json --device cpu
```

Swap `infer_en.py` for `infer_zh.py` and point `--vocab` to
`vocab_wiki_4k.json` to run the Chinese example.  Increase `--top-k` to display
more decoded candidates or set `--history-length`, `--num-sources`, and
`--num-candidates` to control the synthetic prompt used for inference.

### 4. Optional maintenance

- `python train.py --help` and `python infer_en.py --help` print the full list of
  configurable parameters, including regularisation weights (`--beta`, `--gamma`,
  `--tau`) and random seeds.
- Run `deactivate` in PowerShell to exit the virtual environment when you are
  done.
- To upgrade PyTorch or install GPU-enabled builds, follow the wheel selection
  guidance on the [official PyTorch download page](https://pytorch.org/get-started/locally/).

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
