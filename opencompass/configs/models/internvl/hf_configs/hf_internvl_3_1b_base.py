from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='internvl-3-1b-hf',
        path='OpenGVLab/InternVL3-1B-Pretrained',
        max_out_len=8192,
        max_seq_len=8192,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]