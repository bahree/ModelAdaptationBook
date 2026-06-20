# Chapter 5 notices (datasets and attribution)

## Primary Dataset

- **Stack Exchange network IT Q&A**: Super User, Ask Ubuntu, and Server Fault (license: CC-BY-SA-4.0)
  - Source sites: `https://superuser.com`, `https://askubuntu.com`, `https://serverfault.com`
  - Real-world IT and help-desk questions and answers contributed by the Stack Exchange community
  - Cleaned and reformatted into the book's house style for fine-tuning demonstrations in this chapter
  - Per-example source URLs and contributor attribution are written to `data/it_support/attribution.jsonl`
  - Commercially viable under share-alike terms (CC-BY-SA-4.0 license)

## General-Capability Mix-In

- **Databricks Dolly 15K**: `databricks/databricks-dolly-15k` (license: CC-BY-SA-3.0)
  - Dataset: `https://huggingface.co/datasets/databricks/databricks-dolly-15k`
  - Created by Databricks employees in March-April 2023
  - 15,000 instruction-response pairs across 7 task categories
  - Used as a small general-instruction mix-in for capability retention alongside the IT core
  - Commercially viable (CC-BY-SA-3.0 license)

## Why this dataset?

This dataset is used because:
1. **Realistic enterprise task**: real IT and help-desk questions match the IT-support running example that threads chapters 4 through 9, rather than a toy or generic corpus.
2. **Real-world content**: drawn from the Stack Exchange network (Super User, Ask Ubuntu, Server Fault), not a synthetic example.
3. **Measurable tasks**: distinct IT task types enable clear before/after evaluation, with a small Dolly mix-in preserving general-instruction ability.
4. **Appropriate size**: a few hundred examples is ideal for LoRA demonstration.

## Share-alike note

The Stack Exchange content (CC-BY-SA-4.0) and the Databricks Dolly 15K mix-in (CC-BY-SA-3.0) both carry share-alike obligations. Any redistribution of the prepared dataset or derived material must preserve the attribution in `data/it_support/attribution.jsonl` and remain under compatible CC-BY-SA terms.
