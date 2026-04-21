# Hugging Face Setup (Simple Steps)

## What you need to know first

- You need **one Hugging Face token** for your account.
- You do **not** create one token per model.
- For gated models (like LLaMA), you must also request/accept access on the model page.

## Step 1: Create HF account

1. Go to `https://huggingface.co/`
2. Sign up or log in.

## Step 2: Request model access (for gated models)

1. Open the model page you need (example: LLaMA model page).
2. Click request/accept access.
3. Wait until access is approved on your account.

Without approval, downloads fail even if your token is valid.

## Step 3: Create one token

1. Open Hugging Face settings -> Access Tokens:
   - `https://huggingface.co/settings/tokens`
2. Create token with `Read` permission.
3. Copy the token once and store it securely.

## Step 4: Save token on your machine/server (recommended: env var)

### Linux/macOS

```bash
export HF_TOKEN="your_token_here"
```

### Windows PowerShell (current session)

```powershell
$env:HF_TOKEN="your_token_here"
```

### Windows PowerShell (persist for your user)

```powershell
setx HF_TOKEN "your_token_here"
```

Open a new shell after `setx`.

## Step 5: Log in with CLI on the machine that will run jobs

```bash
hf auth login --token "$HF_TOKEN"
```

If you are on Windows PowerShell:

```powershell
hf auth login --token $env:HF_TOKEN
```

## Step 6: Verify

```bash
hf auth whoami
```

You should see your HF username.

## Step 7: Use the same token for all required models

- LLaMA downloads
- CLIP downloads
- Any other HF-hosted assets

Same token, same account, as long as that account has access.
