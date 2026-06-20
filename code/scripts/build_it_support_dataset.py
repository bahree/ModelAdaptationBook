"""Build the book's shared IT-support dataset: real Stack Exchange IT Q&A
(domain core) + a small Dolly slice (general-capability / anti-forgetting
mix-in). Replaces the old Dolly-keyword-filtered "IT support" set, which was
~90% general-knowledge Q&A (Andrew Ch6 C10).

Sources
  - Stack Exchange IT sites via HuggingFaceH4/stack-exchange-preferences
    (CC-BY-SA-4.0): superuser.com, askubuntu.com, serverfault.com.
    question + best answer -> SFT; high-vs-low score answers -> DPO pairs.
  - databricks/databricks-dolly-15k (CC-BY-SA-3.0): general retention slice.

Outputs (code/data/it_support/)
  - train.jsonl / valid.jsonl  : chat-format SFT (80% IT, 20% Dolly in train; valid is IT-only)
  - preferences.jsonl          : DPO pairs (prompt / chosen / rejected) from SE scores
  - manifest.json              : counts, filters, seed, licenses, topic distribution
  - attribution.jsonl          : per-example Stack Exchange source URL (CC-BY-SA requirement)

Run from code/:  python scripts/build_it_support_dataset.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import itertools
import os
import re
import sys
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.jsonl import write_jsonl  # noqa: E402
from common.manifest import write_json  # noqa: E402
from common.seed import seed_everything  # noqa: E402

IT_SYSTEM = "You are an IT support assistant. Provide clear, step-by-step answers."
GEN_SYSTEM = "You are a helpful assistant."

SE_SITES = ["superuser.com", "askubuntu.com", "serverfault.com"]
DOLLY_CATEGORIES = ["closed_qa", "open_qa", "information_extraction", "summarization"]

# Lightweight topic tags for the per-category table (the H4 dataset carries no
# tags). First match wins; order matters (specific topics before generic ones).
TOPICS = [
    ("networking", ["network", "wifi", "wi-fi", "dns", "vpn", "router", "ethernet", "firewall",
                    "proxy", "subnet", "ip address", "port forward", "ssh", "tcp", "dhcp",
                    "nginx", "apache", "domain"]),
    ("security", ["password", "encrypt", "ssl", "tls", "certificate", "malware", "virus",
                  "permission", "authentication", "gpg", "login", "credential", "hash", "secure"]),
    ("windows", ["windows", "outlook", "excel", "registry", "powershell", ".exe", "blue screen",
                 "bsod", "cmd", "explorer", "office", "onedrive", "wsl"]),
    ("linux", ["linux", "ubuntu", "bash", "apt", "sudo", "kernel", "grub", "systemd", "shell",
               "debian", "rpm", "yum", "cron", "chmod", "chown", "mount"]),
    ("hardware", ["disk", "ssd", "hdd", "cpu", "gpu", "ram", "memory", "usb", "monitor", "keyboard",
                  "battery", "bios", "driver", "boot", "partition", "raid", "fan"]),
    ("software", ["install", "update", "upgrade", "application", "program", "software", "crash",
                  "error", "package", "dependency", "version", "config", "compile"]),
]
REAL_TOPICS = ["networking", "security", "windows", "linux", "hardware", "software"]


def topic_of(text: str) -> str:
    tl = " " + text.lower() + " "
    for name, kws in TOPICS:
        if any(k in tl for k in kws):
            return name
    return "general"


def clean_html(raw: str) -> str:
    """HTML -> plain text, preserving code as fenced/inline code, dropping images."""
    soup = BeautifulSoup(raw or "", "html.parser")
    for pre in soup.find_all("pre"):
        pre.replace_with("\n```\n" + pre.get_text().strip() + "\n```\n")
    for code in soup.find_all("code"):
        code.replace_with("`" + code.get_text() + "`")
    for img in soup.find_all("img"):
        img.decompose()
    text = html.unescape(soup.get_text())
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def best_answer(answers):
    """Accepted answer, else the highest-scored; None if nothing decent."""
    cand = [a for a in answers if a.get("selected") or (a.get("pm_score") or 0) >= 3]
    if not cand:
        return None
    cand.sort(key=lambda a: (a.get("selected", False), a.get("pm_score") or 0), reverse=True)
    return cand[0]


def pref_pair(answers):
    """(chosen, rejected) by score gap >= 3, else None."""
    scored = sorted(answers, key=lambda a: a.get("pm_score") or 0, reverse=True)
    if len(scored) < 2:
        return None
    hi, lo = scored[0], scored[-1]
    if (hi.get("pm_score") or 0) - (lo.get("pm_score") or 0) < 3:
        return None
    return hi, lo


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/it_support")
    ap.add_argument("--train", type=int, default=450)
    ap.add_argument("--valid", type=int, default=50)
    ap.add_argument("--dolly_frac", type=float, default=0.20)
    ap.add_argument("--prefs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scan_per_site", type=int, default=5000,
                    help="Max rows to stream per SE site")
    ap.add_argument("--min_q", type=int, default=30)
    ap.add_argument("--max_q", type=int, default=1200)
    ap.add_argument("--min_a", type=int, default=50)
    ap.add_argument("--max_a", type=int, default=2000)
    return ap.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    import random
    rng = random.Random(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_dolly = round(args.train * args.dolly_frac)
    n_it_train = args.train - n_dolly
    n_it_valid = args.valid

    # ---- 1. Collect Stack Exchange IT examples + preference pairs ----
    sft_pool, pref_pool = [], []
    per_site_sft = Counter()
    for site in SE_SITES:
        try:
            ds = load_dataset("HuggingFaceH4/stack-exchange-preferences",
                              data_dir=f"data/{site}", split="train", streaming=True)
        except Exception as e:
            print(f"  skip {site}: {repr(e)[:120]}", flush=True)
            continue
        got = 0
        for r in itertools.islice(ds, args.scan_per_site):
            q = clean_html(r["question"])
            if not (args.min_q <= len(q) <= args.max_q):
                continue
            url = (r.get("metadata") or ["", "", ""])[0]
            ba = best_answer(r["answers"])
            if ba:
                a = clean_html(ba["text"])
                if args.min_a <= len(a) <= args.max_a:
                    sft_pool.append({"site": site, "url": url, "q": q, "a": a, "topic": topic_of(q)})
                    per_site_sft[site] += 1
                    got += 1
            pp = pref_pair(r["answers"])
            if pp:
                ch, rj = clean_html(pp[0]["text"]), clean_html(pp[1]["text"])
                if args.min_a <= len(ch) <= args.max_a and args.min_a <= len(rj) <= args.max_a:
                    pref_pool.append({"site": site, "url": url, "q": q, "chosen": ch, "rejected": rj})
        print(f"  {site}: {got} SFT candidates", flush=True)

    rng.shuffle(sft_pool)
    rng.shuffle(pref_pool)
    if len(sft_pool) < n_it_train + n_it_valid:
        raise RuntimeError(f"Only {len(sft_pool)} IT SFT examples; need {n_it_train + n_it_valid}")

    # ---- Stratified sampling: balance categories from the real pool ----
    # Floor each of the 6 real topics, fill the remainder from "general"
    # (real IT questions that did not hit a topic keyword). This matches the
    # book's own advice: match the natural distribution but ensure every
    # category has enough coverage for a meaningful per-category eval.
    from collections import defaultdict
    by_topic = defaultdict(list)
    for e in sft_pool:
        by_topic[e["topic"]].append(e)

    n_valid_per = max(1, n_it_valid // (len(REAL_TOPICS) + 1))   # ~7 each
    n_train_per = max(40, n_it_train // (len(REAL_TOPICS) + 1))  # ~51 each, general fills rest

    it_train, it_valid, used = [], [], set()

    def take(bucket, k):
        picked = []
        for e in bucket:
            if k <= 0:
                break
            key = (e["url"], e["a"][:40])
            if key in used:
                continue
            used.add(key)
            picked.append(e)
            k -= 1
        return picked

    for t in REAL_TOPICS:
        b = by_topic.get(t, [])
        it_valid += take(b, n_valid_per)
        it_train += take(b, n_train_per)
    # Fill remaining slots from "general" (then any leftover real topics).
    rest = by_topic.get("general", []) + [e for t in REAL_TOPICS for e in by_topic.get(t, [])]
    it_valid += take(rest, n_it_valid - len(it_valid))
    it_train += take(rest, n_it_train - len(it_train))
    rng.shuffle(it_train)
    rng.shuffle(it_valid)

    def to_msg(ex, system=IT_SYSTEM):
        return {"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": ex["q"]},
            {"role": "assistant", "content": ex["a"]},
        ], "category": ex.get("topic", "general"), "source": ex.get("site", "dolly")}

    # ---- 2. Dolly general-retention slice (NO IT filter; general capability) ----
    print("Loading Dolly for retention slice...", flush=True)
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    dpool = []
    for row in dolly:
        if row.get("category") not in DOLLY_CATEGORIES:
            continue
        i, c, resp = row.get("instruction", ""), row.get("context", ""), row.get("response", "")
        if not (40 <= len(i) + len(c) + len(resp) <= 2000):
            continue
        u = f"{c}\n\n{i}".strip() if c.strip() else i
        dpool.append({"q": u, "a": resp, "site": "dolly", "topic": "general"})
    rng.shuffle(dpool)
    dolly_train = dpool[:n_dolly]

    # ---- 3. Assemble + write ----
    train_rows = [to_msg(e) for e in it_train] + [to_msg(e, GEN_SYSTEM) for e in dolly_train]
    rng.shuffle(train_rows)
    valid_rows = [to_msg(e) for e in it_valid]
    pref_rows = [{"prompt": p["q"], "chosen": p["chosen"], "rejected": p["rejected"]}
                 for p in pref_pool[:args.prefs]]

    write_jsonl(out / "train.jsonl", train_rows)
    write_jsonl(out / "valid.jsonl", valid_rows)
    write_jsonl(out / "preferences.jsonl", pref_rows)
    write_jsonl(out / "attribution.jsonl",
                [{"url": e["url"], "site": e["site"]} for e in (it_train + it_valid)])

    topic_dist = dict(Counter(r["category"] for r in train_rows))
    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": args.seed,
        "sources": {
            "it_core": {"dataset": "HuggingFaceH4/stack-exchange-preferences",
                        "sites": SE_SITES, "license": "CC-BY-SA-4.0"},
            "general_mixin": {"dataset": "databricks/databricks-dolly-15k", "license": "CC-BY-SA-3.0"},
        },
        "mix": {"it_train": n_it_train, "dolly_train": n_dolly, "dolly_frac": args.dolly_frac},
        "counts": {"train": len(train_rows), "valid": len(valid_rows), "preferences": len(pref_rows),
                   "it_sft_pool": len(sft_pool), "pref_pool": len(pref_pool)},
        "per_site_sft_candidates": dict(per_site_sft),
        "train_topic_distribution": topic_dist,
        "filters": {"min_q": args.min_q, "max_q": args.max_q, "min_a": args.min_a, "max_a": args.max_a},
        "attribution_note": "Stack Exchange content is CC-BY-SA-4.0; see attribution.jsonl for source URLs.",
    }
    write_json(out / "manifest.json", manifest)

    print(f"\nWrote {out}/", flush=True)
    print(f"  train.jsonl       {len(train_rows)} ({n_it_train} IT + {n_dolly} Dolly)", flush=True)
    print(f"  valid.jsonl       {len(valid_rows)} (IT-only)", flush=True)
    print(f"  preferences.jsonl {len(pref_rows)} DPO pairs", flush=True)
    print(f"  IT SFT pool       {len(sft_pool)} | pref pool {len(pref_pool)}", flush=True)
    print(f"  per-site SFT      {dict(per_site_sft)}", flush=True)
    print(f"  train topics      {topic_dist}", flush=True)
    sys.stdout.flush()
    os._exit(0)  # avoid HF streaming generator teardown coredump (output already flushed)


if __name__ == "__main__":
    main()
