from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='qwen2vl-2b-sft-hf',
        path='weizhiwang/Open-Qwen2VL',
        max_out_len=8192,
        max_seq_len=8192,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]