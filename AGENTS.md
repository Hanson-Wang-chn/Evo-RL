# Agent Instructions

## Python Environment

Use the following Conda environment for this repository:

```bash
conda activate lerobot
```

If the named environment is unavailable, use the absolute path:

```bash
conda activate /llm_jzm/cache/conda_env/lerobot
```

If using `conda activate lerobot`:
- Treat it as personal machine mode.
- Do not set shared-machine Hugging Face cache or token overrides.

If using `conda activate /llm_jzm/cache/conda_env/lerobot`:
- Treat it as shared machine mode.
- Hugging Face cache is not in the default location.
- Set Hugging Face token to the shared token for runs on this machine.
- Set Wandb token to the shared token for runs on this machine

Use:

```bash
export HF_HOME=/llm_jzm/cache/huggingface/
export HF_TOKEN=<SHARED_HF_TOKEN>
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_API_KEY=<SHARED_WANDB_API_KEY>
wandb login --relogin "$WANDB_API_KEY"
```

## User Preferences

- During coding, avoid try/except unless explicitly requested; prefer direct and rough failure/debug output.
- Reuse existing code/tools first; do not rebuild from scratch unless needed.
- Do not preserve forward/backward compatibility unless explicitly requested; prefer direct cleanup and replacement over compatibility shims.
- For every experiment run request (e.g., training/evaluation runs), always create a new Notion page with purpose, exact parameters/command, and key results/observations.
- Debug/troubleshooting runs do not count as experiments by default, and do not require creating a Notion experiment page unless explicitly requested.
- Write complete notes proactively and boldly; user will decide what to keep.
