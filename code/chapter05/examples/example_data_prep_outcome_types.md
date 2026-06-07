# Data-prep outcome types (worked examples)

These illustrate the response types discussed in the chapter's "Data quality
iterations" section, using the Contoso IT-support assistant. Each is a single
training row in the same `messages` format produced by
`scripts/listing_5_2_prepare_dataset.py` (see `dolly_to_messages`).

> **These rows are illustrative.** The Dolly 15K subset used in this chapter
> contains no refusals and no tone tags, so these are examples of what you would
> *add* for an internal assistant. They are not part of the committed
> `data/dolly_subset/`, and adding them would change the training mix and the
> eval numbers, so the pipeline is left as-is.

## Refusal (decline, then redirect)

A response type Dolly never contains. Teaching it explicitly is one fix for the
safety-refusal regression discussed in the Results section.

```json
{"messages": [
  {"role": "system",    "content": "You are Contoso's internal IT support assistant."},
  {"role": "user",      "content": "What is the admin password for the finance file share?"},
  {"role": "assistant", "content": "I can't share admin passwords. To get access, open the IT portal, go to Access > File Shares, and submit a request for your manager to approve."}
]}
```

## Clarification (ask instead of guessing)

When the request is underspecified, the target behaviour is to ask for the
missing detail rather than hallucinate one.

```json
{"messages": [
  {"role": "system",    "content": "You are Contoso's internal IT support assistant."},
  {"role": "user",      "content": "My laptop will not connect."},
  {"role": "assistant", "content": "Let us narrow it down: are you trying to reach Wi-Fi, the VPN, or a wired connection, and is there an error on screen? Meanwhile, confirm Wi-Fi is on and you are in range of a Contoso access point."}
]}
```

## Tone tag (marker in the system prompt, repeated at inference)

```json
{"messages": [
  {"role": "system",    "content": "[Tone: Friendly] You are Contoso's internal IT support assistant."},
  {"role": "user",      "content": "How do I reset my password?"},
  {"role": "assistant", "content": "Quick one! Head to portal.contoso.com/reset, sign in with your badge ID, and follow the prompts. You will be back in within a couple of minutes."}
]}
```

The `[Tone: Friendly]` marker is part of the system prompt during training;
prepend the same marker at inference to select that voice. Keep one marker
vocabulary across the dataset, or train a separate adapter per tone.

## Inter-annotator agreement on Q&A

Agreement is harder to pin down on Q&A than on classification. Ask two support
engineers to write the gold answer for "How do I install the latest Node LTS on
my dev machine?" and you may get two reasonable but different answers: one points
to Contoso's internal `dev-setup` tool and its approved steps, the other gives a
generic `nvm` command. A three-label sentiment task rarely has this ambiguity; an
open-ended Q&A answer routinely does. Because a model's ceiling is roughly the
agreement level of its labels, write an answer-key guideline (preferred tools,
required steps, house format) before annotating, so reviewers converge on one
shape rather than several defensible ones.
