# Example: Base Model vs LoRA Adapter Inference

This example shows the same IT support prompt run with **(1) base model only** and **(2) base model + LoRA adapter**, so you can compare output style and content. Commands are run from the `code/` directory with the virtual environment activated.

## Commands

**Base model (no adapter):**
```bash
python -m chapter05.generate \
  --base Qwen/Qwen3-4B-Instruct-2507 \
  --prompt "My VPN disconnects every few minutes. How do I troubleshoot it?"
```

**Base model + LoRA adapter:**
```bash
python -m chapter05.generate \
  --base Qwen/Qwen3-4B-Instruct-2507 \
  --adapter chapter05/runs/it_lora \
  --prompt "My VPN disconnects every few minutes. How do I troubleshoot it?"
```

## Base model output (no adapter)

The base model produces a long, heavily formatted answer with markdown headings, emoji, and external links. With the default `--max_new_tokens` (128) it is cut off mid-list:

```
A VPN disconnecting every few minutes can be frustrating, but it's usually caused by one of several common issues. Here's a step-by-step troubleshooting guide to help you identify and fix the problem:

---

### 🔍 1. Check Your Internet Connection
- A weak or unstable internet connection can cause the VPN to drop.
- Test your connection speed and stability using tools like Speedtest.net.
- Try browsing or streaming without the VPN — if your connection is unstable, that's likely the root cause.

✅ Fix: Improve your Wi-Fi signal, move closer to the router, or
```

## Base model + LoRA adapter output

The fine-tuned model produces a concise, direct IT-support answer in plain prose: no markdown headings, no emoji, no external links, and it completes within the token limit:

```
If your VPN disconnects every few minutes, it could be due to a number of reasons. First, check if you have any antivirus software installed, as it may be interfering with the VPN connection. Try disabling your antivirus temporarily to see if that resolves the issue.

Additionally, ensure that your VPN client is up to date and that you have a stable internet connection. If the problem persists, try restarting your router or modem, as this can sometimes resolve temporary connectivity issues.

If none of these solutions work, consider contacting your VPN provider for further assistance, as the issue may be on their end.
```

## What to notice

- **Base:** Long, decorated answer with markdown headings, emoji (🔍, ✅), and external links. Hits the token limit and is truncated mid-list.
- **LoRA adapter:** Concise, single-voice IT-support prose. No headings, no emoji, no links. Completes within the limit and reads like an internal helpdesk reply.
- Both answers are reasonable; the adapter reflects the format and tone of the IT support training data, which is the point of fine-tuning here. (The raw transcript also includes the chat-template tokens `<|im_start|>` / `<|im_end|>` around each turn; they are stripped above for readability.)

## Screenshots (terminal output)

**Base vs LoRA adapter:**
![Base vs LoRA adapter inference](../images/chap5-inference_base_vs_adapter.png)
