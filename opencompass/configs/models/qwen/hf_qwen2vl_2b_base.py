from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='qwen2vl-2b-base-hf',
        path='weizhiwang/Open-Qwen2VL-base',
        max_out_len=8192,
        max_seq_len=8192,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]