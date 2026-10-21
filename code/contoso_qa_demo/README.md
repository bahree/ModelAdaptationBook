# Contoso IT-support Q&A (starter domain dataset)

A small, **hand-authored, CC0-licensed** IT-support Q&A set for a fictional company, Contoso. It is self-contained and carries no third-party licensing, so it can be redistributed freely. It serves two purposes in the book:

1. **A concrete data-quality example for question answering** (`data/quality_pairs.jsonl`). Data quality is easy to define for classification (consistent labels) but harder to show for open-ended Q&A. These good-versus-bad answer pairs make it concrete: each `bad` variant carries a `flaw` label (incomplete, wrong_format, hallucinated_detail, off_tone, inconsistent_terms, dangerous_wrong, generic_no_domain).
2. **A case where a fine-tuned adapter clearly beats prompting** (`data/it_qa_train.jsonl`). The answers use Contoso-internal tool names and a fixed answer format the base model has never seen, so a prompt cannot fake them. That makes the adapter-versus-prompting contrast unambiguous, where a more general Q&A set (on which a base model plus a good prompt is often already enough) would not.

It is used in **chapter 5** (the base-vs-adapter contrast in section 5.1.8, figure 5.5, and the Q&A data-prep examples), and the same Contoso assistant recurs in chapters 6, 8, and 9.

> **Not the book's main training data.** The fine-tuning pipeline in chapters 5 to 8 trains on a separate, larger corpus of real Stack Exchange IT Q&A (Super User, Ask Ubuntu, Server Fault) at `code/data/it_support/`. This folder holds only the small Contoso demo set used for the section 5.1.8 contrast and the Q&A data-quality example.

## Why prompting cannot match it

Answers use **Contoso-internal tool names** and a **fixed micro-format** the base model has never seen:

- **Internal terms:** GlobalConnect VPN, AccessHub, Contoso MFA (Authenticator app), the StandardBuild image, the #it-help Slack channel, ServiceNow request types.
- **Format:** a one-sentence direct answer, then numbered `Steps:`, then a fixed escalation line (ServiceNow plus #it-help, P1 if a shared service is down).

A prompt cannot inject names the model does not know; a LoRA adapter learns them. That is what makes the base-versus-adapter contrast real rather than cosmetic. Contoso is fictional; every tool name is invented, and nothing is scraped or proprietary.

## Files

| File | What | Use |
| --- | --- | --- |
| `make_it_qa.py` | Hand-authored content + emitter (standard library only) | edit or extend the data here |
| `data/it_qa_train.jsonl` | 24 Q&A in ChatML (`messages` + `category`), Contoso house style | domain LoRA training; the "good" reference for quality |
| `data/quality_pairs.jsonl` | 5 questions, each with a `good` answer and labeled `bad` variants | the Q&A data-quality illustration |
| `LICENSE` | CC0 1.0 (public domain dedication) | redistribute freely |

The format matches chapters 5, 6, and 8 (`{"messages": [...], "category": ...}`), so it drops into the existing LoRA/SFT pipelines without conversion.

## How it is used

- **In the book:** the contrast (a base or format-prompted base versus a domain LoRA on a Contoso-terminology question), plus one good-versus-bad quality pair.
- **In the repo:** the full set plus a reproducible training and evaluation run.

It is a **starter**: 24 hand-authored rows are enough to illustrate the idea, not to ship. Expand to a few hundred in the same house style before quoting accuracy numbers, and keep the `source` tag so synthetic expansions stay separable from the hand-authored seed.

## Results (domain LoRA on a single GPU)

The seed was expanded to **166 training rows** (18 hand-authored, 148 synthesized with `anthropic/claude-sonnet-4.5` via OpenRouter, all gated to the house style and balanced across categories: hardware 34, access 30, software 27, cloud/network/security 25 each), with 6 held out for evaluation. A LoRA (r=16, about 5 epochs, roughly a minute on a single 24 GB GPU) was scored against the base model and a format-prompted base. Timing varies by accelerator; the functional results below do not. Full numbers are in `results/results.json`, with a chart in `results/contrast.png`:

| config | internal tool in body | house format | token-F1 |
| --- | --- | --- | --- |
| base | 0.00 | 0.00 | 0.198 |
| base + prompt | 0.00 | 0.00 | 0.233 |
| adapter | 0.50 | 0.83 | 0.576 |

What the numbers show:

- **House format: 0.83 vs 0.00.** The adapter reliably produces the Contoso house format (one-line answer, numbered steps, ServiceNow/#it-help escalation); neither the base nor a format-prompted base matches it.
- **Token-F1 nearly triples** (0.198 to 0.576), a gain a prompt cannot supply.
- **Internal-tool recall in the body is 0.50** (vs 0.00 for the base and prompted base, which never use Contoso vocabulary because no prompt supplied it). It rose from 0.17 once the thinner categories were filled out, showing that per-category data volume drives recall.

Two things to keep in mind:

- **Volume has diminishing returns.** A second hardware top-up (20 to 34 rows) did not move the metrics: internal-tool recall stayed at 0.50 and token-F1 was flat. Once a category is adequately represented, adding more of the same buys little; the early 0.17 to 0.50 jump was the real signal.
- **A tiny evaluation set is noisy.** With only 6 held-out items, house format reads 0.83 here versus 1.00 on another run, which is a single item flipping rather than a regression. Trust the direction (the adapter well above the base and the prompt), not the third decimal.

The contrast is real (format, token-F1, and half the answers naming the right internal tool), and the lever for the remaining gap is more diverse per-category data, which is itself the book's point about domain-data volume and quality.

## Regenerate

```bash
python3 make_it_qa.py   # writes data/it_qa_train.jsonl and data/quality_pairs.jsonl
```
