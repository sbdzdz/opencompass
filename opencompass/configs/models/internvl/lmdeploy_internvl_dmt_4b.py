from opencompass.models import TurboMindModelwithChatTemplate

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='internvl-3-1b-instruct-turbomind',
        path='./models/4B/checkpoint-1500',
        engine_config=dict(session_len=8192, max_batch_size=16, tp=1),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=8192),
        max_seq_len=16384,
        max_out_len=16384,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]