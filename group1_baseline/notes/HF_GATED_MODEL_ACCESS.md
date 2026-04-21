# Hugging Face Gated Model Access (LLaMA) - Simple Steps

Use this when you get:
- `403 Forbidden`
- `GatedRepoError`
- "you are not in the authorized list"

## 1) Log in to the same HF account used by your token

Go to:
- `https://huggingface.co/`

Confirm you are logged into the correct account.

## 2) Open the model page and request access

Go to:
- `https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct`

Then:
1. Click `Request access` (or `Agree and access`).
2. Accept terms/license if prompted.
3. Submit request.

## 3) Wait for approval

- Some models are instant after agreement.
- Some require manual approval.
- Until approved, token-based downloads will keep failing with 403.

## 4) Verify your token on server

```bash
cd /root/final_project/group1_baseline
source .venv/bin/activate
hf auth whoami
```

If this fails, re-login:

```bash
hf auth login --token "$HF_TOKEN"
```

If `HF_TOKEN` is empty in the shell, load it from `.env` first:

```bash
cd /root/final_project/group1_baseline
source .venv/bin/activate
set -a
source .env
set +a
echo -n "$HF_TOKEN" | wc -c
```

If count is `0`, `.env` is missing/invalid.  
Expected format:

```bash
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
```

## 5) Test gated access quickly

```bash
python - << 'PY'
import os
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    token=os.environ.get("HF_TOKEN"),
)
print("Tokenizer access OK")
PY
```

## 6) If still failing

Check:
1. You requested access from the same HF account tied to `HF_TOKEN`.
2. `HF_TOKEN` in `.env` is correct and loaded in current shell/kernel.
3. You restarted notebook kernel after token/setup changes.

Note:
- You may see: `Environment variable HF_TOKEN is set and is the current active token`.
- This is normal and means your shell env token is being used (good for notebook runs).
