"""

Demonstrates:
  • DatasetManifest — SHA-256 content hash, source provenance, quality metadata
  • Lineage recording — connecting dataset versions to model checkpoints
  • Dataset diff — comparing two versions to see what changed
  • Retention schedule — GDPR-compliant deletion planning

No API key required. No HuggingFace datasets required.

Install:
    pip install datasets  (optional — only needed for the HuggingFace save demo)
"""

import os
import json
import hashlib
import datetime
import random
from dataclasses import dataclass, field
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# DatasetManifest
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetManifest:
    """
    Version manifest for a fine-tuning dataset.

    One manifest per saved dataset. Attach it in the same directory as the
    data files. After each training run, call record_training_run() to extend
    the lineage chain:

        dataset v1.0.0  →  model checkpoint  →  eval scores
                        →  dataset v2.0.0   →  model checkpoint  →  eval scores

    The manifest answers two governance questions at any point:
      1. What data trained this model?  (SHA-256, sources, composition)
      2. Is that data still valid?      (deletion log, retention deadline)
    """
    version:           str
    created_at:        str   = field(default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z")
    sha256:            str   = ""
    num_examples:      int   = 0
    num_train:         int   = 0
    num_eval:          int   = 0
    sources:           list  = field(default_factory=list)
    pii_scrubbed:      bool  = False
    annotation_kappa:  float | None = None
    synthetic_ratio:   float = 0.0
    quality_pass_rate: float | None = None
    centroid_sim:      float | None = None
    trained_models:    list  = field(default_factory=list)
    deletion_log:      list  = field(default_factory=list)
    retain_until:      str   = ""

    # ── Content hashing ───────────────────────────────────────────────────────

    def compute_hash(self, examples: list[dict]) -> "DatasetManifest":
        """Compute and store the SHA-256 hash of the full dataset content."""
        content       = json.dumps(examples, sort_keys=True).encode()
        self.sha256   = hashlib.sha256(content).hexdigest()
        self.num_examples = len(examples)
        return self

    # ── Lineage recording ─────────────────────────────────────────────────────

    def record_training_run(
        self,
        checkpoint_path: str,
        eval_scores:     dict,
        training_loss:   float | None = None,
        notes:           str = "",
    ) -> "DatasetManifest":
        """
        Append a training run record to the lineage chain.

        Call immediately after model.save_pretrained() so the manifest
        connects dataset → checkpoint → eval results in one atomic update.

        Args:
            checkpoint_path: Path where the model was saved
            eval_scores:     Per-category evaluation results
            training_loss:   Final training loss (from trainer.train())
            notes:           Free-text notes about this training run
        """
        # Append rather than overwrite — a dataset may be used for multiple
        # training runs (e.g. hyperparameter search) and each should be recorded
        self.trained_models.append({
            "checkpoint":    checkpoint_path,
            "trained_at":    datetime.datetime.utcnow().isoformat() + "Z",
            "training_loss": training_loss,
            "eval_scores":   eval_scores,
            "notes":         notes,
        })
        return self

    # ── Deletion compliance ───────────────────────────────────────────────────

    def log_deletion(self, user_id: str, affected_examples: int) -> "DatasetManifest":
        """
        Record a user deletion request against this dataset version.

        If affected_examples > 0, a clean retraining is required. Document
        the obligation here so it is not forgotten.

        Args:
            user_id:           Identifier of the user requesting deletion
            affected_examples: Number of examples from this user in the dataset
        """
        self.deletion_log.append({
            "user_id":           user_id,
            "requested_at":      datetime.datetime.utcnow().isoformat() + "Z",
            "affected_examples": affected_examples,
            # Retraining is only needed when the user's data is actually in this dataset.
            # If affected_examples == 0, the user was not in this version, so no retrain.
            "retraining_needed": affected_examples > 0,
            "status":            "pending",
        })
        return self

    def set_retention(self, years: int = 2) -> "DatasetManifest":
        """Set the GDPR-compliant retention deadline for this dataset."""
        # Parse the ISO timestamp stored at creation time
        created  = datetime.datetime.fromisoformat(self.created_at.rstrip("Z"))
        # Approximate years using 365-day periods; adjust for leap years if needed
        deadline = created + datetime.timedelta(days=years * 365)
        self.retain_until = deadline.isoformat() + "Z"
        return self

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Write the manifest to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(vars(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DatasetManifest":
        """Load a manifest from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def display(self) -> None:
        """Pretty-print the manifest to stdout."""
        print(f"  Version     : {self.version}")
        print(f"  Created     : {self.created_at}")
        print(f"  SHA-256     : {self.sha256[:24]}...")
        print(f"  Examples    : {self.num_examples} ({self.num_train} train / {self.num_eval} eval)")
        print(f"  Sources     : {self.sources}")
        print(f"  PII scrubbed: {self.pii_scrubbed}")
        if self.annotation_kappa is not None:
            print(f"  Annotation κ: {self.annotation_kappa:.3f}")
        if self.synthetic_ratio > 0:
            print(f"  Synthetic   : {self.synthetic_ratio:.0%}")
        if self.quality_pass_rate is not None:
            print(f"  Pass rate   : {self.quality_pass_rate:.0%}")
        if self.retain_until:
            print(f"  Retain until: {self.retain_until[:10]}")
        if self.trained_models:
            print("  Models trained on this data:")
            for m in self.trained_models:
                scores = m.get("eval_scores", {})
                avg = sum(scores.values()) / len(scores) if scores else 0
                print(f"    {m['checkpoint']}  loss={m.get('training_loss','?')}  "
                      f"avg_score={avg:.0%}")
        if self.deletion_log:
            pending = sum(1 for d in self.deletion_log if d["status"] == "pending")
            print(f"  Deletion log: {len(self.deletion_log)} requests, {pending} pending")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset diff
# ──────────────────────────────────────────────────────────────────────────────

def diff_manifests(
    manifest_a: DatasetManifest,
    manifest_b: DatasetManifest,
) -> dict:
    """
    Compare two dataset versions and return a structured diff.

    Useful for understanding what changed between a training run that worked
    and one that did not, or for auditing data evolution over time.

    Args:
        manifest_a: Older dataset version
        manifest_b: Newer dataset version

    Returns:
        dict with changed fields and a change summary
    """
    changes = {}
    # Only check fields that are directly comparable with ==.
    # trained_models is excluded because the lineage chain grows over time
    # and a larger list is expected, not a sign of data change.
    fields = [
        "num_examples", "num_train", "num_eval", "sources",
        "synthetic_ratio", "quality_pass_rate", "pii_scrubbed",
    ]
    for f in fields:
        val_a = getattr(manifest_a, f)
        val_b = getattr(manifest_b, f)
        if val_a != val_b:
            changes[f] = {"before": val_a, "after": val_b}

    # SHA-256 comparison is the ground truth for content equality.
    # If the hash changed, the data changed — regardless of what fields say.
    if manifest_a.sha256 != manifest_b.sha256:
        changes["content"] = "changed"
    else:
        changes["content"] = "unchanged"

    return {
        "version_a":   manifest_a.version,
        "version_b":   manifest_b.version,
        "changes":     changes,
        "num_changes": len(changes) - 1,   # "content" entry is informational, not a field change
    }


# ──────────────────────────────────────────────────────────────────────────────
# Retention policy
# ──────────────────────────────────────────────────────────────────────────────

def check_retention_status(manifest: DatasetManifest) -> dict:
    """
    Check whether a dataset is within its GDPR retention window.

    If the dataset has passed its retain_until date, it must be deleted or
    anonymised. Any model trained on it must be retrained on compliant data.

    Args:
        manifest: DatasetManifest with a retain_until date set

    Returns:
        dict with status ("active", "expiring_soon", "expired") and days remaining
    """
    if not manifest.retain_until:
        return {"status": "unknown", "message": "No retention deadline set. Call set_retention()."}

    deadline  = datetime.datetime.fromisoformat(manifest.retain_until.rstrip("Z"))
    now       = datetime.datetime.utcnow()
    # Negative days_left means the deadline has already passed
    days_left = (deadline - now).days

    if days_left < 0:
        # Dataset has passed its GDPR retention deadline — delete or anonymise now
        return {"status": "expired", "days": days_left, "action": "Delete immediately"}
    elif days_left < 90:
        # Within 90 days of expiry — schedule deletion before it becomes a compliance issue
        return {"status": "expiring_soon", "days": days_left,
                "action": f"Schedule deletion within {days_left} days"}
    else:
        return {"status": "active", "days": days_left, "action": "No action needed"}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 3.7 — Data Versioning and Lineage")
    print("=" * 60)

    # ── Build a realistic synthetic dataset for demonstration ─────────────────
    rng = random.Random(42)
    fake_train = [
        {"messages": [{"role": "user", "content": f"Finance Q{i}"},
                       {"role": "assistant", "content": f"Answer {i}"}],
         "source": "real" if i < 40 else "synthetic",
         "category": rng.choice(["sentiment", "financial_qa", "translation"])}
        for i in range(50)
    ]
    fake_eval = fake_train[:5]
    fake_train = fake_train[5:]

    # ── Create manifest v1.0.0 ────────────────────────────────────────────────
    print("\n[1] Creating manifest v1.0.0")
    m1 = DatasetManifest(
        version="1.0.0",
        num_train=len(fake_train),
        num_eval=len(fake_eval),
        sources=["financial_phrasebank", "sujet_finance"],
        pii_scrubbed=True,
        annotation_kappa=0.79,
        synthetic_ratio=0.20,
        quality_pass_rate=0.88,
    )
    m1.compute_hash(fake_train + fake_eval)
    m1.set_retention(years=2)
    m1.display()

    # ── Save manifest ─────────────────────────────────────────────────────────
    os.makedirs("/tmp/ch03_demo/v1_0_0", exist_ok=True)
    m1.save("/tmp/ch03_demo/v1_0_0/manifest.json")
    print("\n  Saved to /tmp/ch03_demo/v1_0_0/manifest.json")

    # ── Record a training run ─────────────────────────────────────────────────
    print("\n[2] Recording training run on v1.0.0")
    m1.record_training_run(
        checkpoint_path="./qwen-open-finance-v1-output",
        eval_scores={
            "sentiment":    0.82,
            "financial_qa": 0.79,
            "translation":  0.71,
            "rag":          0.55,
            "math_code":    0.45,
        },
        training_loss=0.412,
        notes="First training run, 2,000 examples, 1 epoch",
    )
    m1.save("/tmp/ch03_demo/v1_0_0/manifest.json")
    print("  Training run recorded:")
    print("  Checkpoint  : ./qwen-open-finance-v1-output")
    print("  Eval scores : sentiment=82%  financial_qa=79%  rag=55%  math=45%")
    print("  Training loss: 0.412")

    # ── Create manifest v2.0.0 ────────────────────────────────────────────────
    print("\n[3] Creating manifest v2.0.0 (with targeted synthetic data)")
    fake_train_v2 = fake_train + [
        {"messages": [{"role": "user", "content": f"Math Q{i}"},
                       {"role": "assistant", "content": f"Math A{i} = {i*42}"}],
         "source": "synthetic", "category": "math_code"}
        for i in range(20)
    ]
    m2 = DatasetManifest(
        version="2.0.0",
        num_train=len(fake_train_v2),
        num_eval=len(fake_eval),
        sources=["financial_phrasebank", "sujet_finance", "synthetic_claude"],
        pii_scrubbed=True,
        annotation_kappa=0.79,
        synthetic_ratio=0.30,
        quality_pass_rate=0.91,
    )
    m2.compute_hash(fake_train_v2 + fake_eval)
    m2.set_retention(years=2)
    os.makedirs("/tmp/ch03_demo/v2_0_0", exist_ok=True)
    m2.save("/tmp/ch03_demo/v2_0_0/manifest.json")
    m2.display()

    # ── Diff the two versions ─────────────────────────────────────────────────
    print("\n[4] Diff: v1.0.0 → v2.0.0")
    diff = diff_manifests(m1, m2)
    print(f"  {diff['num_changes']} fields changed:")
    for field, change in diff["changes"].items():
        if field == "content":
            print(f"    content : {change}")
        else:
            print(f"    {field:20s}: {change['before']} → {change['after']}")

    # ── Retention status ──────────────────────────────────────────────────────
    print("\n[5] Retention status")
    for version, manifest in [("v1.0.0", m1), ("v2.0.0", m2)]:
        status = check_retention_status(manifest)
        print(f"  {version}: {status['status'].upper():15s}  "
              f"{status['days']} days remaining  |  {status['action']}")

    # ── Deletion audit trail ──────────────────────────────────────────────────
    print("\n[6] GDPR deletion request")
    m1.log_deletion(user_id="user_98421", affected_examples=3)
    m1.save("/tmp/ch03_demo/v1_0_0/manifest.json")
    pending = [d for d in m1.deletion_log if d["status"] == "pending"]
    print("  Deletion logged: user_98421  |  3 examples affected")
    print(f"  Retraining required: {pending[0]['retraining_needed']}")
    print("  Schedule retraining on v1.0.0 with user_98421 examples removed")

    # ── Load back and verify ───────────────────────────────────────────────────
    print("\n[7] Load manifest from disk and verify SHA-256")
    loaded = DatasetManifest.load("/tmp/ch03_demo/v1_0_0/manifest.json")
    rehash, _ = loaded.sha256, loaded.compute_hash(fake_train + fake_eval)
    match = loaded.sha256 == m1.sha256
    print(f"  Loaded version : {loaded.version}")
    print(f"  SHA-256 match  : {match}  ({'✓ data integrity confirmed' if match else '✗ data modified'})")

    # ── Key takeaways ─────────────────────────────────────────────────────────
    print("\n[8] Key takeaways")
    print("  • A SHA-256 manifest costs 5 lines of code and prevents hours of debugging")
    print("  • Record the training run IN the manifest immediately after saving the model")
    print("  • Set a retention deadline at dataset creation — retrofitting is painful")
    print("  • The manifest is your audit trail: dataset version → model → eval scores")
    print("\nDone.")
