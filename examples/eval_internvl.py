from mmengine.config import read_base

from opencompass.models import TurboMindModelwithChatTemplate

MAX_TOKENS_VALUES = [1024, 2048, 4096, 8192]

with read_base():
    from opencompass.configs.datasets.ceval.ceval_gen_2daf24 import ceval_datasets
    from opencompass.configs.datasets.CLUE_C3.CLUE_C3_gen_8c358f import C3_datasets
    from opencompass.configs.datasets.cmmlu.cmmlu_gen_c13365 import cmmlu_datasets
    from opencompass.configs.datasets.GaokaoBench.GaokaoBench_no_subjective_gen_4c31db import GaokaoBench_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.hellaswag.hellaswag_10shot_gen_e42710 import hellaswag_datasets
    from opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    from opencompass.configs.datasets.math.math_0shot_gen_393424 import math_datasets
    from opencompass.configs.datasets.mbpp_cn.mbpp_cn_gen_9114d5 import mbpp_cn_datasets
    from opencompass.configs.datasets.mbpp.sanitized_mbpp_mdblock_gen_a447ff import sanitized_mbpp_datasets
    from opencompass.configs.datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets
    from opencompass.configs.datasets.nq.nq_gen_3dcea1 import nq_datasets
    from opencompass.configs.datasets.race.race_gen_69ee4f import race_datasets
    from opencompass.configs.datasets.TheoremQA.TheoremQA_5shot_gen_6f0af8 import TheoremQA_datasets
    from opencompass.configs.datasets.triviaqa.triviaqa_gen_2121ce import triviaqa_datasets
    from opencompass.configs.datasets.winogrande.winogrande_5shot_gen_b36770 import winogrande_datasets

datasets = [
    *ceval_datasets,
    *C3_datasets,
    *cmmlu_datasets,
    *GaokaoBench_datasets,
    *gsm8k_datasets,
    *hellaswag_datasets,
    *humaneval_datasets,
    *math_datasets,
    *mbpp_cn_datasets,
    *sanitized_mbpp_datasets,
    *mmlu_datasets,
    *nq_datasets,
    *race_datasets,
    *TheoremQA_datasets,
    *triviaqa_datasets,
    *winogrande_datasets,
]

model_paths = [
    # InternVL 2.5 base models
    './models/InternVL2_5-1B-Pretrain',
    './models/InternVL2_5-2B-Pretrain',
    './models/InternVL2_5-4B-Pretrain',

    # InternVL 2.5 instruct models
    'OpenGVLab/InternVL2_5-1B',
    'OpenGVLab/InternVL2_5-2B',
    'OpenGVLab/InternVL2_5-4B',

    # InternVL 3.0 base models
    'OpenGVLab/InternVL3-1B-Pretrained',
    'OpenGVLab/InternVL3-2B-Pretrained',

    # InternVL 3.0 instruct models
    'OpenGVLab/InternVL3-1B-Instruct',
    'OpenGVLab/InternVL3-2B-Instruct',
]

def get_model_abbr(path, max_tokens):
    base_abbr = path.split('/')[-1].lower()
    return f"{base_abbr}-{max_tokens}"

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr=get_model_abbr(path, max_tokens),
        path=path,
        engine_config=dict(session_len=16384, max_batch_size=16, tp=1),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=max_tokens),
        max_seq_len=16384,
        max_out_len=16384,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
    for path in model_paths
    for max_tokens in MAX_TOKENS_VALUES
]

print(f"Created configuration for {len(models)} InternVL models with various max_new_tokens values")
