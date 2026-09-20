"""Build the two production sets used by the chapter 9 drift demo.

The drift detector needs a *reference* (training) distribution and a
*production* distribution to compare. The chapter shows two contrasting runs
against the same reference (``data/it_support/train.jsonl``):

1. **Baseline (healthy, YELLOW).** Production = the held-out validation split
   (``data/it_support/valid.jsonl``). Same domain, same source, so drift is
   low: this is what a normal same-domain split looks like and the number you
   calibrate your thresholds against. (No file is written for this one; the run
   points straight at valid.jsonl.)

2. **Shifted (genuine drift, RED).** Production = a simulated batch of recent
   queries collected six months after deployment, after the organisation
   migrated to Kubernetes-based CI and a cloud SSO provider. These topics are
   absent from the training data (kubernetes, docker, okta, azure ad, container,
   cloud = 0/450 training questions; verified), so this is real covariate shift,
   not a small-sample artifact. This script writes that file.

Run from code/:
    python -m chapter09.scripts.build_drift_demo_sets
writes -> chapter09/data/shifted_production.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

SYSTEM = "You are an IT support assistant. Provide clear, step-by-step answers."

# Simulated "six months later" production queries. The organisation has moved
# to Kubernetes-based CI and a cloud SSO provider; users now ask about topics
# the model never saw in training. Phrasing is deliberately mixed (formal and
# Slack-shorthand) to also exercise prompt drift.
SHIFTED_QUESTIONS = [
    "How do I configure single sign-on with Okta for our internal dashboard?",
    "Our Azure AD SSO login keeps redirecting in a loop, how do I fix it?",
    "kubectl get pods shows CrashLoopBackOff, how do I debug this?",
    "How do I roll back a failed deployment in Kubernetes?",
    "My docker container exits immediately with code 137, what does that mean?",
    "How do I set resource limits on a Kubernetes pod?",
    "SSO works on desktop but not on the mobile app, any ideas?",
    "How do I migrate our on-prem Active Directory users to cloud SSO?",
    "kubectl logs is empty for a pod that clearly failed, how do I see why?",
    "How do I mount a persistent volume in a Kubernetes deployment?",
    "docker build is super slow, how do I use layer caching properly?",
    "How do I rotate the OAuth client secret for our SSO integration?",
    "Our ingress controller returns 502 for one service only, how to debug?",
    "How do I give a service account read access to a Kubernetes namespace?",
    "okta push notifications not arriving, mfa broken pls help",
    "How do I expose a containerized app on the cluster to the internet?",
    "What is the difference between a Kubernetes Deployment and a StatefulSet?",
    "How do I configure SAML assertions for our SSO provider?",
    "docker compose up fails with port already allocated, how to fix?",
    "How do I set up horizontal pod autoscaling based on CPU?",
    "Our SSO session times out after 5 minutes, where do I change that?",
    "k8s node is NotReady, how do I bring it back?",
    "How do I pass secrets to a container without baking them into the image?",
    "How do I federate our cloud SSO with a partner organisation?",
    "kubectl apply says field is immutable, how do I update the deployment?",
    "How do I debug a DNS resolution failure between two pods?",
    "container keeps getting OOMKilled, how do I raise the memory limit?",
    "How do I configure conditional access policies in our cloud SSO?",
    "How do I drain a Kubernetes node before maintenance?",
    "docker pull rate limited, how do we set up a pull-through cache?",
    "How do I set up a readiness probe so traffic waits for the app to start?",
    "SSO group claims are missing from the token, how do I add them?",
    "How do I run a one-off job in Kubernetes and clean it up after?",
    "Our SCIM provisioning from the IdP is not creating users, why?",
    "How do I tail logs from all pods in a deployment at once?",
    "How do I configure a NetworkPolicy to isolate a namespace?",
    "azure ad conditional access blocking service account, how to allow it?",
    "How do I roll a secret across all pods without downtime?",
    "kubectl exec into a distroless container fails, how do I troubleshoot?",
    "How do I set up blue-green deployment on Kubernetes with two services?",
    "Our SSO metadata URL changed, where do I update it in the app?",
    "How do I limit a container to a single CPU core?",
    "How do I configure pod affinity so two services land on the same node?",
    "okta sign-in works but app says unauthorized, token audience mismatch?",
    "How do I back up and restore etcd for our cluster?",
    "How do I enable MFA enforcement for all users in cloud SSO?",
    "container clock is wrong, how do I sync time inside the pod?",
    "How do I configure a sidecar container for logging?",
    "How do I troubleshoot an SSO certificate that expired overnight?",
    "How do I scale a deployment to zero and back without losing config?",
]


def main() -> None:
    out_dir = Path("chapter09/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "shifted_production.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for q in SHIFTED_QUESTIONS:
            rec = {"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": q},
            ]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(SHIFTED_QUESTIONS)} shifted production queries to {out_path}")


if __name__ == "__main__":
    main()
