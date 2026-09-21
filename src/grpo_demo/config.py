"""Run configuration. Two named configs: `baseline` (no KL) and `fixed_kl` (KL penalty)."""

from dataclasses import dataclass, asdict, field


@dataclass
class Config:
    # --- identity -------------------------------------------------------
    run: str = "baseline"
    hf_repo: str = ""  # model repo that holds checkpoints, metrics and samples

    # --- models ---------------------------------------------------------
    policy_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    reward_model: str = "lvwerra/distilbert-imdb"

    # --- data -----------------------------------------------------------
    dataset: str = "stanfordnlp/imdb"
    n_train_prompts: int = 2048
    n_eval_prompts: int = 64
    min_prompt_chars: int = 40
    max_prompt_chars: int = 180
    label_filter: int | None = 0   # 0 = negative IMDB reviews only

    # --- GRPO -----------------------------------------------------------
    total_steps: int = 200
    prompts_per_step: int = 8          # B
    group_size: int = 8                # G -> B*G completions per step
    micro_batch: int = 16              # sequences per fwd/bwd chunk
    max_new_tokens: int = 48
    temperature: float = 1.0
    top_p: float = 1.0
    learning_rate: float = 3e-6
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    adv_eps: float = 1e-4
    beta: float = 0.0                  # KL coefficient; 0.0 == no mitigation

    # --- evaluation -----------------------------------------------------
    eval_every: int = 20
    eval_temperature: float = 0.7
    eval_top_p: float = 0.95
    eval_max_new_tokens: int = 48

    # --- checkpointing --------------------------------------------------
    full_ckpt_every: int = 50          # weights + optimizer pushed to the Hub
    seed: int = 0
    output_dir: str = "results"

    def to_dict(self) -> dict:
        return asdict(self)


BASELINE = Config(
    run="baseline",
    beta=0.0,
    hf_repo="rohanjain2312/grpo-reward-hacked-sentiment-qwen05b",
)

FIXED_KL = Config(
    run="fixed_kl",
    beta=0.04,
    hf_repo="rohanjain2312/grpo-reward-hacking-fixed-kl-qwen05b",
)

CONFIGS = {"baseline": BASELINE, "fixed_kl": FIXED_KL}


def get_config(name: str, **overrides) -> Config:
    if name not in CONFIGS:
        raise KeyError(f"unknown config {name!r}; choose from {sorted(CONFIGS)}")
    cfg = Config(**CONFIGS[name].to_dict())
    for k, v in overrides.items():
        if v is None:
            continue
        if not hasattr(cfg, k):
            raise KeyError(f"unknown config field {k!r}")
        setattr(cfg, k, v)
    return cfg
