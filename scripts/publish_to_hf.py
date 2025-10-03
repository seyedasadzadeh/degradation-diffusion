"""Publish model artifacts to a Hugging Face model repository.

This script avoids importing heavy packages (like torch) unless strictly
necessary. If a checkpoint is provided in `model_publish.yml` (or via the
MODEL_CHECKPOINT env var), the script will upload that checkpoint file as-is
and will not import torch. If no checkpoint is provided, the script will
attempt to import torch to construct a fresh model and serialize its
state_dict; if torch is not available the script will still upload
`model_def.py` and warn the user.

Environment / secrets expected in CI:
- HF_TOKEN: your Hugging Face write token (set as GitHub secret)

"""
import os
import sys
import yaml
import argparse


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def copy_checkpoint(checkpoint_path: str, out_name: str) -> str | None:
    """Copy user-provided checkpoint to the expected output name.

    Returns the destination path on success, or None on failure.
    """
    try:
        import shutil
        shutil.copy(checkpoint_path, out_name)
        print(f'Using provided checkpoint: {checkpoint_path} -> {out_name}')
        return out_name
    except Exception as e:
        print('Failed to copy checkpoint:', e)
        return None


def build_and_save_model(out_name: str) -> str | None:
    """Attempt to instantiate a demo model and save its state.

    Returns the saved file path on success, or None if torch/model unavailable.
    """
    try:
        import torch
        from degdiff.model_def import TimeSeriesDiffusionModel
    except Exception as e:
        print('Torch or local model definition not available:', e)
        print('No checkpoint provided and torch not present — will upload only model_def.py')
        return None

    device = 'cpu'
    model = TimeSeriesDiffusionModel(channels=1, T=50).to(device)
    state = model.state_dict()

    # Prefer safetensors when available
    try:
        from safetensors.torch import save_file as safe_save
        np_state = {k: v.cpu().numpy() for k, v in state.items()}
        safe_save(np_state, out_name)
        print('Saved safetensors to', out_name)
        return out_name
    except Exception:
        try:
            import torch as _torch
            _torch.save(state, out_name)
            print('Saved torch checkpoint to', out_name)
            return out_name
        except Exception as e:
            print('Failed saving model state:', e)
            return None


def upload_to_hf(repo_id: str, saved_path: str | None, model_def_path: str) -> bool:
    """Clone the HF repo locally, copy artifacts, and push to the hub.

    Returns True on success, False on failure.
    """
    try:
        from huggingface_hub import HfApi
        hf_token = os.environ.get('HF_TOKEN')
        if not hf_token:
            print('HF_TOKEN environment variable not set. Aborting upload.')
            return False

        api = HfApi()
        # create repo if not exists
        try:
            api.create_repo(token=hf_token, name=repo_id.split('/')[-1], private=False, exist_ok=True, repo_type='model')
        except Exception:
            pass

        uploaded_any = False

        # Upload saved weights if available using the HTTP API (no git-lfs)
        if saved_path and os.path.exists(saved_path):
            try:
                api.upload_file(
                    path_or_fileobj=saved_path,
                    path_in_repo=os.path.basename(saved_path),
                    repo_id=repo_id,
                    token=hf_token,
                )
                print('Uploaded weights to repo:', os.path.basename(saved_path))
                uploaded_any = True
            except Exception as e:
                print('Failed uploading weights via HTTP API:', e)
                return False
        else:
            print('No saved weights to upload (no checkpoint and torch unavailable).')

        # Upload model_def.py
        if os.path.exists(model_def_path):
            try:
                api.upload_file(
                    path_or_fileobj=model_def_path,
                    path_in_repo=os.path.basename(model_def_path),
                    repo_id=repo_id,
                    token=hf_token,
                )
                print('Uploaded', os.path.basename(model_def_path))
                uploaded_any = True
            except Exception as e:
                print('Failed uploading model_def.py via HTTP API:', e)
                return False
        else:
            print('Warning: model_def.py not found at', model_def_path)

        if uploaded_any:
            print('Uploaded artifacts to', repo_id)
            return True
        else:
            print('No artifacts uploaded.')
            return False
    except Exception as e:
        print('Failed uploading to HF (unexpected error):', e)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='model_publish.yml')
    parser.add_argument('--repo-id', help='hf repo id override')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    repo_id = args.repo_id or os.environ.get('HF_REPO_ID') or cfg.get('repo_id')
    if not repo_id:
        print('Error: repo_id not configured (set HF_REPO_ID or repo_id in model_publish.yml)')
        return 2

    model_def_path = cfg.get('model_def_path', 'src/degdiff/model_def.py')
    out_name = cfg.get('out_model_filename', 'model.safetensors')

    checkpoint_path = cfg.get('checkpoint', '') or os.environ.get('MODEL_CHECKPOINT', '')
    saved_path = None

    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            print(f'Configured checkpoint {checkpoint_path} does not exist. Aborting.')
            return 6
        saved_path = copy_checkpoint(checkpoint_path, out_name)
        if not saved_path:
            return 7
    else:
        saved_path = build_and_save_model(out_name)

    ok = upload_to_hf(repo_id, saved_path, model_def_path)
    return 0 if ok else 5


if __name__ == '__main__':
    sys.exit(main())
