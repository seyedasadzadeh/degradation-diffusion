# Copilot / AI agent instructions — degradation-diffusion

This repository is a small demo that trains a time-series diffusion model (in a notebook) to synthesize degradation traces.
Keep changes local, minimal, and notebook-friendly unless you extract reusable code into .py modules.

Quick references
- Primary entry: `train_and_validate.ipynb` (all core logic: data generation, models, training loop, sampling).
- Short project description: `README.md`.

Big picture (what matters)
- Data generator: `ParisLawDegradation` (class) produces synthetic crack-growth episodes → returns arrays shaped (N, length+1).
- Data layout used everywhere: X is prepared as a tensor with shape (N, C, L) where C is channels (usually 1) and L is sequence length.
- Models:
  - `DegDiffusion` — compact Conv/MLP denoiser defined in the notebook.
  - `TimeSeriesDiffusionModel` — Transformer-like model with `TimeEmbedding` and `TransformerBlock`.
- Diffusion design: each model implements a `q_sample(x0, t, noise)` method which (importantly) sets `self.betas` and returns the noisy xt. Sampling code reads `model.betas` to build schedules — don't remove that side-effect.

Important repo-specific patterns & gotchas
- Device: notebook uses Apple Metal device calls like `.to('mps')`. If running on CPU/GPU, update device strings accordingly (e.g., `'cpu'` or `'cuda'`) across all `.to(...)` calls in the notebook.
- Beta schedule inconsistency: `TimeSeriesDiffusionModel.q_sample` uses betas up to `1e-1`, while `DegDiffusion.q_sample` uses `1e-2`. If you change the schedule, update all q_sample implementations and sampling code that reads `model.betas`.
- Time embedding: `TimeSeriesDiffusionModel` has a commented WARNING: "Double-check adding here!" — pay attention when altering time embedding semantics (embedding shape is expanded and repeated to length L).
- Indexing and dtypes: `q_sample` indexes `alphas_bar[t]` where `t` is a batched tensor — ensure `t` is a LongTensor on the same device as model tensors.
- Training split pattern: the training loop splits a sequence into `s0` (available signal) and `ns1` (noisy remainder) using `s0_len = 50`. When changing sequence lengths, also update `s0_len` and downstream slicing in sampling and loss computation.
- Sampling relies on `model.betas` being present (set during training/inference q_sample). Ensure any refactor preserves that behavior or write an explicit schedule provider.

Developer workflows (how to run & iterate)
- Install dependencies (recommended minimal set inferred from the notebook):
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas scikit-learn matplotlib jupyter
```
- Open and iterate in Jupyter (recommended):
```bash
jupyter lab train_and_validate.ipynb
```
- Quick dev loop tips:
  - Reduce `epoch` count (notebook uses up to 50_000) and batch size for quick iteration.
  - If you get device errors on non-Apple machines, replace `.to('mps')` with `.to('cpu')` or `.to('cuda')`.
  - To run cells as a script for CI or reproducible runs, export the notebook to a script:
```bash
jupyter nbconvert --to script train_and_validate.ipynb
python train_and_validate.py   # after editing the exported script
```

- Files & symbols to reference when editing
- `train_and_validate.ipynb` — contains these important classes/functions:
  - `ParisLawDegradation.generate_episode` — produces `episodes`
  - `TimeSeriesDiffusionModel`, `TimeEmbedding`, `TransformerBlock`
  - `DegDiffusion` — alternate denoiser
  - `model.q_sample(...)` — implements forward diffusion and stores `self.betas`
  - training loop: slices `x0` into `s0` and `ns1` and computes `pred_noise` vs `output_noise`

Do's and don'ts for automated edits
- Do: preserve the q_sample → model.betas side-effect or replace sampling code to accept an explicit schedule.
- Do: keep device handling explicit and consistent across tensors (inputs, t, noise, model parameters).
- Don't: change the X tensor layout (N,C,L) without updating all slices and model projections — many operations assume this shape.
- Don't: assume an external test harness exists — this repo is a demo notebook; add tests only after extracting logic into .py modules.

If something is unclear or you need conventions added (packaging, versions, tests), tell me which area to expand and I'll update this file.
