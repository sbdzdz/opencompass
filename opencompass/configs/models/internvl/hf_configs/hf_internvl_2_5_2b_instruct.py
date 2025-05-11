from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='internvl2_5-2b-hf',
        path='OpenGVLab/InternVL2_5-2B',
        max_out_len=8192,
        max_seq_len=8192,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]