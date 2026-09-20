"""Demonstrate automated rollback using the model registry.

Companion script to chapter 9 (not a numbered listing in the chapter).

This script simulates a production scenario where:
1. Model v1 is deployed and performing well.
2. Model v2 is trained and promoted -- but shows evaluation degradation.
3. An automated check detects the drop and triggers rollback to v1.
4. A verification step confirms v1 is back in service.

No real models are loaded -- the script simulates eval metrics to
demonstrate the *workflow*.  In production, replace the simulated
metrics with calls to your evaluation pipeline.

Run from code/:
    python -m chapter09.rollback_demo --registry_dir chapter09/data/demo_registry
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from chapter09.model_registry import (
    get_active_version,
    list_versions,
    promote_version,
    register_model,
    rollback,
)


# ---------------------------------------------------------------------------
# Simulated evaluation metrics
# ---------------------------------------------------------------------------
SIMULATED_METRICS = {
    "it-support-v1": {"overall_f1": 0.72, "safety_pass_rate": 1.0},
    "it-support-v2": {"overall_f1": 0.58, "safety_pass_rate": 0.75},
}

# Thresholds
F1_THRESHOLD = 0.65
SAFETY_THRESHOLD = 0.90


def simulate_eval(version_tag: str) -> dict:
    """Return simulated eval metrics for a model version."""
    return SIMULATED_METRICS.get(version_tag, {"overall_f1": 0.50, "safety_pass_rate": 0.50})


def check_thresholds(metrics: dict) -> tuple[bool, list[str]]:
    """Check whether metrics meet deployment thresholds.

    Returns:
        Tuple of (passed, list_of_failures).
    """
    failures = []
    if metrics.get("overall_f1", 0) < F1_THRESHOLD:
        failures.append(
            f"overall_f1 ({metrics['overall_f1']:.2f}) < threshold ({F1_THRESHOLD})"
        )
    if metrics.get("safety_pass_rate", 0) < SAFETY_THRESHOLD:
        failures.append(
            f"safety_pass_rate ({metrics['safety_pass_rate']:.2f}) < threshold ({SAFETY_THRESHOLD})"
        )
    return len(failures) == 0, failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demonstrate automated rollback workflow"
    )
    parser.add_argument(
        "--registry_dir",
        default="chapter09/data/demo_registry",
        help="Directory for the demo registry (created fresh each run)",
    )
    parser.add_argument(
        "--output",
        default="chapter09/eval/rollback_report.json",
        help="Path to save the rollback report",
    )
    args = parser.parse_args()

    registry_dir = Path(args.registry_dir)
    # Start with a clean registry for the demo
    reg_file = registry_dir / "registry.json"
    if reg_file.exists():
        reg_file.unlink()

    timeline = []

    def log_event(event: str, details: dict | None = None) -> None:
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": event,
            "details": details or {},
        }
        timeline.append(entry)
        print(f"  [{entry['timestamp']}] {event}")

    print("=" * 65)
    print("ROLLBACK DEMO: Automated Model Lifecycle Management")
    print("=" * 65)

    # Step 1: Register and deploy v1
    print("\nStep 1: Register model v1")
    register_model(
        registry_dir=registry_dir,
        version_tag="it-support-v1",
        technique="sft",
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        data_hash="data-v1-abc123",
        checkpoint_path="chapter06/runs/sft_run1",
        eval_metrics=SIMULATED_METRICS["it-support-v1"],
        notes="Chapter 6 SFT model -- stable baseline",
    )
    promote_version(registry_dir, "it-support-v1")
    log_event("v1_deployed", {"version": "it-support-v1", "status": "active"})

    # Step 2: Register and deploy v2
    print("\nStep 2: Register model v2 (with degraded metrics)")
    register_model(
        registry_dir=registry_dir,
        version_tag="it-support-v2",
        technique="dpo",
        base_model="Qwen/Qwen3-4B-Instruct-2507",
        data_hash="data-v2-def456",
        checkpoint_path="chapter08/runs/dpo_run1",
        eval_metrics=SIMULATED_METRICS["it-support-v2"],
        notes="DPO model -- optimized for helpfulness",
    )
    promote_version(registry_dir, "it-support-v2")
    log_event("v2_deployed", {"version": "it-support-v2", "status": "active"})

    # Step 3: Automated evaluation check detects degradation
    print("\nStep 3: Post-deployment evaluation check")
    active = get_active_version(registry_dir)
    assert active is not None
    metrics = simulate_eval(active["version_tag"])
    passed, failures = check_thresholds(metrics)
    log_event("eval_check", {
        "version": active["version_tag"],
        "metrics": metrics,
        "passed": passed,
        "failures": failures,
    })

    if not passed:
        print(f"\n  ALERT: Deployment check FAILED for {active['version_tag']}")
        for f in failures:
            print(f"    - {f}")

        # Step 4: Automated rollback
        print("\nStep 4: Executing automated rollback")
        rolled_back = rollback(registry_dir)
        log_event("rollback_executed", {
            "rolled_back_to": rolled_back["version_tag"],
            "reason": failures,
        })
        print(f"  Rolled back to: {rolled_back['version_tag']}")

        # Step 5: Verify rollback
        print("\nStep 5: Verifying rollback")
        verify_active = get_active_version(registry_dir)
        assert verify_active is not None
        verify_metrics = simulate_eval(verify_active["version_tag"])
        verify_passed, verify_failures = check_thresholds(verify_metrics)
        log_event("rollback_verified", {
            "version": verify_active["version_tag"],
            "metrics": verify_metrics,
            "passed": verify_passed,
        })

        if verify_passed:
            print(f"  Rollback verified: {verify_active['version_tag']} passes all checks")
        else:
            print(f"  WARNING: Rollback target also fails checks: {verify_failures}")

    else:
        print(f"\n  Deployment check PASSED for {active['version_tag']}")
        log_event("deployment_healthy", {"version": active["version_tag"]})

    # Print final registry state
    print("\n" + "-" * 65)
    print("Final registry state:")
    for entry in list_versions(registry_dir):
        print(f"  {entry['version_tag']:<25} status={entry['status']:<10} "
              f"technique={entry['technique']}")

    # Save rollback report
    report = {
        "demo_completed": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "f1_threshold": F1_THRESHOLD,
        "safety_threshold": SAFETY_THRESHOLD,
        "timeline": timeline,
        "final_active_version": (get_active_version(registry_dir) or {}).get("version_tag"),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nRollback report saved to {out_path}")


if __name__ == "__main__":
    main()
