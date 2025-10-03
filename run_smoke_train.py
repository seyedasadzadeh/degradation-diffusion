#!/usr/bin/env python3
"""Small CLI to run a smoke training run for degradation-diffusion.

Runs a few epochs on a tiny synthetic dataset to validate end-to-end training
and sampling paths. Exit code 0 on success, non-zero on failure.
"""
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run a tiny smoke training run")
    parser.add_argument("--model", choices=["timeseries", "deg"], default="timeseries", help="Which model to run")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--length", type=int, default=32, help="sequence length")
    parser.add_argument("--n", type=int, default=64, help="number of generated episodes")
    args = parser.parse_args()

    try:
        import numpy as np
    except Exception:
        print("Missing dependency: numpy. Install with `pip install numpy`.")
        return 2

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except Exception:
        print("Missing dependency: torch. Ensure the 'smartchp' conda env has torch installed.")
        return 3

    # import local package
    try:
        from degdiff.generators import ParisLawDegradation
        from degdiff.model_def import TimeSeriesDiffusionModel, DegDiffusion
    except Exception as e:
        print("Failed importing local package 'degdiff':", e)
        print("Make sure to run this script with PYTHONPATH=src from the repo root, or install the package.")
        return 4

    # device selection
    if args.device == "auto":
        device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device

    print(f"Running smoke train with model={args.model} epochs={args.epochs} batch={args.batch} device={device}")

    # determinism
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # tiny synthetic dataset
    N = int(args.n)
    length = int(args.length)
    gen = ParisLawDegradation(length=length - 1, dim=1, C=1e-8)
    x0 = np.abs(np.random.randn(N)) * 1e-3 + 1e-4
    episodes = gen.generate_episode(x0)  # shape (N, length)

    # quick cleanup
    episodes = episodes[~np.isnan(episodes).any(axis=1)]
    episodes = episodes[(episodes < 1).any(axis=1)]

    # convert to torch and normalize per-sample min/max as in notebook
    data = torch.tensor(episodes, dtype=torch.float32).to(device)
    X = data[:, None, ...]  # (N, C, L)
    # avoid divide-by-zero
    mn = torch.min(X, 2)[0][..., None]
    mx = torch.max(X, 2)[0][..., None]
    denom = (mx - mn)
    denom[denom == 0] = 1.0
    X = (X - mn) / denom

    C = X.shape[1]
    L = X.shape[2]

    # instantiate model
    if args.model == 'timeseries':
        model = TimeSeriesDiffusionModel(channels=C, T=50).to(device)
    else:
        model = DegDiffusion(channels=C, sl=L, T=50).to(device)

    opt = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    s0_len = max(1, L // 4)

    try:
        for epoch in range(args.epochs):
            idx = torch.randint(0, X.shape[0], (args.batch,))
            x0_batch = X[idx].to(device)
            t = torch.randint(0, model.T, (args.batch,)).to(device)
            noise = torch.randn_like(x0_batch).to(device)
            xt = model.q_sample(x0_batch, t, noise)

            s0 = x0_batch[:, :, :s0_len]
            ns1 = xt[:, :, s0_len:]
            noise1 = noise[:, :, s0_len:]
            noise0 = torch.zeros_like(s0).to(device)
            output_noise = torch.cat([noise0, noise1], dim=2)

            pred_noise = model(s0, ns1, t)
            loss = loss_fn(pred_noise, output_noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_val = loss.item()
            print(f"epoch={epoch} loss={loss_val:.6f}")
            if not (np.isfinite(loss_val)):
                print("Failure: loss is not finite", loss_val)
                return 10

    except Exception as e:
        print("Training failed with exception:", e)
        return 11

    # quick sampling check (single sample)
    try:
        with torch.no_grad():
            n_samples = min(4, X.shape[0])
            x = torch.randn(n_samples, C, L).to(device)
            betas = getattr(model, 'betas', None)
            if betas is None:
                # ensure betas exist by running q_sample once on a sample
                t_sample = torch.randint(0, model.T, (n_samples,)).to(device)
                noise_sample = torch.randn_like(x).to(device)
                _ = model.q_sample(x, t_sample, noise_sample)
                betas = getattr(model, 'betas', None)

            alphas = 1.0 - betas
            alphas_bar = torch.cumprod(alphas, dim=0)

            for tstep in reversed(range(model.T)):
                t_batch = torch.full((n_samples,), tstep, dtype=torch.long).to(device)
                # use the notebook's single-step sampling update
                noise_pred = model(x0_batch[:n_samples, :, :s0_len], x[:, :, s0_len:], t_batch)
                alpha = alphas[tstep]
                alpha_bar = alphas_bar[tstep]
                x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * noise_pred)
                if tstep > 0:
                    x += torch.sqrt(betas[tstep]) * torch.randn_like(x)

    except Exception as e:
        print("Sampling failed with exception:", e)
        return 12

    print("Smoke training completed successfully.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
