import yaml
from huggingface_hub import upload_file
import os

def load_config(config_path):
    """Load YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config {config_path}: {e}")
        raise

def main(config_path):
    """Upload model_def.py and best.safetensors to Hugging Face repo."""
    # Load config
    config = load_config(config_path)
    repo_id = config['repo_id']
    checkpoint = config['checkpoint']  # e.g., checkpoints/best.safetensors
    model_def_path = config['model_def_path']  # e.g., src/degdiff/model_def.py
    out_model_filename = config['out_model_filename']  # e.g., model.safetensors

    # Verify files exist
    if not os.path.exists(model_def_path):
        raise FileNotFoundError(f"Model definition file {model_def_path} not found")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint file {checkpoint} not found")

    # Upload files to Hugging Face
    try:
        upload_file(
            path=model_def_path,
            path_in_repo='model_def.py',
            repo_id=repo_id,
            repo_type='model'
        )
        print(f"Uploaded {model_def_path} to {repo_id}/model_def.py")
        upload_file(
            path=checkpoint,
            path_in_repo=out_model_filename,
            repo_id=repo_id,
            repo_type='model'
        )
        print(f"Uploaded {checkpoint} to {repo_id}/{out_model_filename}")
    except Exception as e:
        print(f"Error uploading files to {repo_id}: {e}")
        raise

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Upload model files to Hugging Face')
    parser.add_argument('--config', default='model_publish.yml', help='Path to config file')
    args = parser.parse_args()
    main(args.config)