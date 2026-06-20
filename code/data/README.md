# `code/data/`: the book's IT-support training corpus

This folder holds the **real, attributable** IT-support data used by the hands-on
pipeline in chapters 5 to 8 (LoRA, full SFT, distillation, DPO).

| Folder | What it is | Built by | License |
| --- | --- | --- | --- |
| `it_support/` | Real Stack Exchange IT Q&A (Super User, Ask Ubuntu, Server Fault) as the domain core, plus a small Databricks Dolly slice for general-capability retention. Holds the train / valid / preference splits, the manifest, and per-example source attribution. | `../scripts/build_it_support_dataset.py` | Stack Exchange CC-BY-SA-4.0; Dolly CC-BY-SA-3.0 |
| `it_support_fmt/` | The **same `it_support` data, reformatted** into the assistant's house answer style. `train.jsonl` here is the file the SFT actually trains on. Not a separate dataset, a processed view of `it_support`. | `../scripts/reformat_it_answers.py` | derived from `it_support` |

Build both from the `code/` directory with:

```bash
python scripts/build_it_support_dataset.py   # -> data/it_support/
python scripts/reformat_it_answers.py         # -> data/it_support_fmt/train.jsonl
```

## Not the same as `code/contoso_qa_demo/`

`code/contoso_qa_demo/` (one level up, not here in `data/`) is a separate, small, hand-authored Contoso Q&A set used only for the chapter 5 section 5.1.8 adapter-beats-prompting demo and the Q&A data-quality example. It is a self-contained mini-project with its own builder, demo, results, and CC0 license. This folder (`data/it_support*`) is the real Stack Exchange corpus the chapters fine-tune on. See [`../contoso_qa_demo/README.md`](../contoso_qa_demo/README.md).
