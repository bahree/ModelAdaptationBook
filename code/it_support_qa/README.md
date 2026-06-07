# Contoso IT-support Q&A (starter domain dataset)

A small, **hand-authored, CC0-licensed** IT-support Q&A set built to fill two gaps the tech editor (Andrew) flagged across chapters 5 to 8. It is deliberately **ours and self-contained**, so it does not depend on the Chapter 3 IT rewrite and carries no third-party licensing.

> Used in the book: the Contoso base-vs-adapter contrast in **chapter 5** (section 5.1.8, "Where an adapter beats prompting," figure 5.5) and the Q&A data-prep examples draw on this set, and the same assistant is carried forward in chapters 6, 8, and 9. It is still a compact starter (see the note on size at the end): enough to illustrate the contrast, not to ship.

## Why this exists

1. **Data quality for Q&A** (`data/quality_pairs.jsonl`). The book defines "high-quality data" only for classification (consistent labels). For Q&A it only asserts quality without showing it. These good-vs-bad pairs make Q&A quality concrete: each `bad` variant carries a `flaw` label (incomplete, wrong_format, hallucinated_detail, off_tone, inconsistent_terms, dangerous_wrong, generic_no_domain).
2. **A domain-adaptation example where LoRA beats prompting** (`data/it_qa_train.jsonl`). Every worked example so far uses generic Dolly Q&A, where a base model plus a prompt is already fine. This set is domain-specific in a way a prompt cannot fake.

## The domain hook (why prompting cannot match it)

Answers use **Contoso-internal tool names** and a **fixed micro-format** the base model has never seen:

- Internal terms: GlobalConnect VPN, AccessHub, Contoso MFA (Authenticator app), the StandardBuild image, the #it-help Slack channel, ServiceNow request types.
- Format: one-sentence direct answer, then numbered `Steps:`, then a fixed escalation line (ServiceNow + #it-help, P1 if a shared service is down).

A prompt cannot inject names the model does not know; LoRA learns them. That is what makes the base-vs-adapter contrast real rather than cosmetic (the gap Andrew kept asking to see).

Contoso is a fictional company; every tool name is invented. Nothing is scraped or proprietary.

## Files

| File | What | Use |
| --- | --- | --- |
| `make_it_qa.py` | Hand-authored content + emitter (stdlib only) | edit/extend the data here |
| `data/it_qa_train.jsonl` | 24 Q&A in ChatML (`messages` + `category`), Contoso house style | domain LoRA training (#3); the "good" reference for quality |
| `data/quality_pairs.jsonl` | 5 questions, each with a `good` answer and labeled `bad` variants | the Q&A data-quality illustration (#2) |
| `LICENSE` | CC0 1.0 (public domain dedication) | redistribute freely in the public repo |

Format matches chapters 5/6/8 (`{"messages": [...], "category": ...}`), so it drops into the existing LoRA/SFT pipelines without conversion.

## Suggested use (hybrid, per the plan)

- **In the chapter:** show the contrast (base/prompted vs domain-LoRA on a Contoso-terminology question) and explain why a prompt cannot match it; show one good-vs-bad quality pair.
- **In the repo:** the full set + the training/eval run, reproducible.

It is a *starter*: 24 rows is enough to illustrate, not to ship. Expand to a few hundred (same house style) before quoting accuracy numbers, and keep the `source` tag so synthetic expansions stay separable from the hand-authored seed.

## Results so far (real A30 run, domain LoRA)

The seed was expanded with `anthropic/claude-sonnet-4.5` via OpenRouter (all
gated to the house style), with a per-category target so thin categories are not
starved. Final: **166 training rows** (18 hand + 148 synthetic, balanced across
categories: hardware 34, access 30, software 27, cloud/network/security 25 each),
6 held out as golden. A LoRA (r=16, ~5 epochs, ~90 s on an A30) was scored
against base and a format-prompted base. Numbers in `results/results.json`, chart
in `results/contrast.png`:

| config | internal tool in body | house format | token-F1 |
| --- | --- | --- | --- |
| base | 0.00 | 0.00 | 0.198 |
| base + prompt | 0.00 | 0.00 | 0.233 |
| adapter | 0.50 | 0.83 | 0.576 |

Read it honestly:

- **House format and escalation: 0.83 vs 0.00.** The adapter reliably produces
  the Contoso house format (one-line answer, numbered steps, ServiceNow/#it-help
  escalation); base and a format-prompted base do not match it.
- **Token-F1 nearly triples** (0.198 -> 0.576), the gain a prompt cannot supply.
- **Internal-tool recall in the body is 0.50** (vs 0.00 for base and base+prompt,
  which never use Contoso vocabulary because no prompt supplied it). This rose
  from 0.17 once the thin categories were first filled, a clean demonstration that
  per-category data volume drives recall.

Two honest caveats, both useful chapter lessons in their own right:

- **Diminishing returns on volume.** A second hardware top-up (20 -> 34) did not
  move the metrics: internal-tool recall stayed at 0.50 and token-F1 was flat.
  Once a category is adequately represented, adding more of the same buys little;
  the early 0.17 -> 0.50 jump was the real signal.
- **Tiny eval is noisy.** With only 6 golden items, house format reads 0.83 here
  versus 1.00 on the previous run, which is a single eval item flipping, not a
  regression. Trust the direction (adapter >> base/prompt), not the third digit.

This is a proof of concept on the branch, not a chapter figure: the contrast is
real (format, F1, and half the answers naming the right internal tool), and the
lever for the remaining gap is more *diverse* per-category data, which is itself
the book's point about domain-data volume and quality.

## Regenerate

```bash
python3 make_it_qa.py   # writes data/it_qa_train.jsonl and data/quality_pairs.jsonl
```
