from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='internvl2_5-4b-base-turbomind',
        path='./models/InternVL2_5-4B-Pretrain',
        engine_config=dict(session_len=16384, max_batch_size=16, tp=1),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=16384),
        max_seq_len=16384,
        max_out_len=16384,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]