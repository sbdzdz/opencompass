import os
import torch
import logging
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_models(models_dir="models"):
    """Download InternVL models from HuggingFace."""
    os.makedirs(models_dir, exist_ok=True)

    logger.info(f'Downloading InternVL models to {models_dir}...')
    snapshot_download(
        repo_id="OpenGVLab/InternVL2_5-Pretrain-Models",
        repo_type="model",
        allow_patterns=[
            "InternVL2_5-1B-Pretrain/*",
            "InternVL2_5-2B-Pretrain/*",
            "InternVL2_5-4B-Pretrain/*"
        ],
        local_dir=models_dir
    )
    logger.info('Download completed.')

def load_model(model_path, models_dir="models"):
    """Load an InternVL model from local checkpoint."""
    full_model_path = os.path.join(models_dir, model_path)
    logger.info(f'Loading model from {full_model_path}...')

    config = AutoConfig.from_pretrained(full_model_path, trust_remote_code=True)
    if hasattr(config, 'llm_config'):
        config.llm_config.attn_implementation = 'flash_attention_2'

    model = AutoModel.from_pretrained(
        full_model_path,
        torch_dtype=torch.bfloat16,
        config=config,
        trust_remote_code=True
    )

    logger.info('Model loaded successfully.')
    return model

if __name__ == "__main__":
    models_dir = "models"
    download_models(models_dir)