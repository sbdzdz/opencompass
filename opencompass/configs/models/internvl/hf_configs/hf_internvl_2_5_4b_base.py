from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='internvl2_5-4b-base-hf',
        path='OpenGVLab/InternVL2_5-Pretrain-Models/InternVL2_5-4B-Pretrain',
        max_out_len=8192,
        max_seq_len=8192,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]