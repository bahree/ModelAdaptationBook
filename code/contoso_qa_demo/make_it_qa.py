"""Starter IT-support Q&A dataset for the domain-adaptation example (Ch5+).

Hand-authored, license-free (CC0), and independent of the Ch3 IT rewrite. It
serves two purposes the book currently lacks a concrete asset for:

  1. Data quality for Q&A (the "what does good Q&A data look like" gap): see
     data/quality_pairs.jsonl, good-vs-bad answers to the same question.
  2. A domain-adaptation example where LoRA beats prompting (the "show me one"
     gap): data/it_qa_train.jsonl, clean Q&A in Contoso's internal house style
     and terminology that a base model cannot produce from a prompt alone.

The domain hook is deliberate: answers use Contoso-internal tool names
(GlobalConnect VPN, AccessHub, the StandardBuild image, the #it-help channel)
and a fixed micro-format (one-line answer, numbered steps, escalation line). A
prompt cannot inject names the base model has never seen; LoRA can. That is what
makes the contrast with prompting real.

Contoso is a fictional company (the standard Microsoft sample name); all tool
names are invented. Nothing here is scraped or proprietary.

Run:  python3 make_it_qa.py     (stdlib only; writes data/*.jsonl)
"""
from __future__ import annotations

import json
from pathlib import Path

SYSTEM = "You are Contoso's internal IT support assistant. Give clear, accurate help."

# Fixed house style (this is what LoRA learns and a prompt cannot fully specify):
#   line 1: one-sentence direct answer
#   "Steps:" + numbered, imperative steps
#   escalation line referencing ServiceNow + #it-help
# plus internal terminology: GlobalConnect VPN, AccessHub, Contoso MFA
# (Authenticator app), StandardBuild image, #it-help Slack channel.

ESC = ("Still stuck? Open a ticket in ServiceNow under \"IT Support\" "
       "(set P1 if a shared service is down) and drop a note in #it-help.")

# (category, question, answer). Answers follow the house style consistently.
QA = [
    ("access", "How do I connect to the VPN from my laptop?",
     "Use the GlobalConnect VPN client, which is preinstalled on the StandardBuild image.\n"
     "Steps:\n1. Open GlobalConnect VPN from the system tray.\n"
     "2. Sign in with your Contoso account and approve the Contoso MFA prompt in the Authenticator app.\n"
     "3. Select the Corporate profile and click Connect.\n" + ESC),
    ("access", "I forgot my password. How do I reset it?",
     "Reset it yourself through AccessHub; IT cannot see or set your password for you.\n"
     "Steps:\n1. Go to AccessHub and choose Forgot password.\n"
     "2. Verify with the Contoso MFA prompt in the Authenticator app.\n"
     "3. Set a new password that meets the 14-character policy.\n" + ESC),
    ("access", "My MFA prompts stopped arriving on my phone. What do I do?",
     "Re-register your device in AccessHub; lost prompts almost always mean the Authenticator enrollment expired.\n"
     "Steps:\n1. In AccessHub, open Security info and remove the old Authenticator entry.\n"
     "2. Add it again and scan the QR code with the Authenticator app.\n"
     "3. Send a test prompt to confirm it arrives.\n" + ESC),
    ("access", "A new hire starts Monday and has no email or login yet. How do I fix this?",
     "Provisioning runs through AccessHub and can be expedited by IT.\n"
     "Steps:\n1. Confirm the hire's record exists in AccessHub (HR creates it on offer acceptance).\n"
     "2. In ServiceNow, raise an Onboarding request with the start date.\n"
     "3. IT provisions the Contoso account, mailbox, and StandardBuild laptop.\n" + ESC),
    ("network", "The Wi-Fi keeps dropping in the warehouse when I move around.",
     "This is usually roaming between access points; switch to the Contoso-Corp network rather than Contoso-Guest.\n"
     "Steps:\n1. Forget Contoso-Guest and connect to Contoso-Corp with your Contoso account.\n"
     "2. Approve the Contoso MFA prompt to authenticate to the network.\n"
     "3. If a specific aisle is dead, note the nearest pillar number for the network team.\n" + ESC),
    ("network", "Internal sites are slow to load but the public internet is fine.",
     "Slow internal-only traffic usually means the VPN split-tunnel is misrouting.\n"
     "Steps:\n1. Disconnect and reconnect GlobalConnect VPN on the Corporate profile.\n"
     "2. Confirm you are on Corporate, not the Backup profile.\n"
     "3. Flush DNS (ipconfig /flushdns on Windows) and retry.\n" + ESC),
    ("hardware", "My laptop will not boot after last night's update, just a black screen.",
     "Try the StandardBuild recovery shortcut before assuming a hardware fault.\n"
     "Steps:\n1. Hold the power button 10 seconds to force off, then power on.\n"
     "2. At the Contoso logo, press F11 to open StandardBuild recovery.\n"
     "3. Choose Roll back last update.\n"
     "If recovery does not appear, the disk may have failed. " + ESC),
    ("hardware", "My docking station stopped charging the laptop and the second monitor is dead.",
     "Docks usually recover with a power-cycle in the right order.\n"
     "Steps:\n1. Unplug the dock from power for 30 seconds.\n"
     "2. Reconnect dock power first, then the laptop USB-C cable.\n"
     "3. If the monitor stays dark, test it on a colleague's dock to isolate dock vs monitor.\n" + ESC),
    ("hardware", "I need a replacement charger; I left mine at the other office.",
     "Loaner chargers are stocked at the IT walk-up bar, no ticket needed for same-day pickup.\n"
     "Steps:\n1. Bring your laptop to the IT walk-up bar to match the wattage.\n"
     "2. Collect a loaner StandardBuild charger.\n"
     "3. If you need one shipped to a home office, raise a Hardware request in ServiceNow.\n" + ESC),
    ("software", "Outlook crashes on launch since this morning.",
     "A corrupt local profile is the most common cause; rebuild it before reinstalling.\n"
     "Steps:\n1. Close Outlook and open Control Panel > Mail > Show Profiles.\n"
     "2. Add a new profile, set it as default, and reopen Outlook.\n"
     "3. Let it re-sync from the server (your mail is safe in Exchange).\n" + ESC),
    ("software", "How do I install the latest Node LTS on my dev machine?",
     "Use the Contoso Software Center; manual installers are blocked by policy on StandardBuild.\n"
     "Steps:\n1. Open Software Center from the Start menu.\n"
     "2. Search for Node LTS (Approved) and click Install.\n"
     "3. Reopen your terminal so the PATH refreshes.\n"
     "Need a version that is not listed? " + ESC),
    ("software", "Can you add the Grammarly extension to the approved list?",
     "Extension approvals go through the Software Review queue, not a direct install.\n"
     "Steps:\n1. In ServiceNow, raise a Software Review request naming the extension and the business need.\n"
     "2. Security reviews it (usually 3 to 5 business days).\n"
     "3. If approved, it appears in Software Center for one-click install.\n" + ESC),
    ("cloud", "What is Contoso's equivalent of Amazon EKS?",
     "Contoso standardizes on AKS (Azure Kubernetes Service); EKS is not used internally.\n"
     "Steps:\n1. Request an AKS namespace through the Platform portal in AccessHub.\n"
     "2. Authenticate kubectl with az login and the Contoso MFA prompt.\n"
     "3. Deploy through the standard pipeline templates in the Platform repo.\n" + ESC),
    ("cloud", "My kubectl commands fail against the prod cluster with an auth error.",
     "Your AKS token has likely expired; re-authenticate rather than editing kubeconfig by hand.\n"
     "Steps:\n1. Run az login and approve the Contoso MFA prompt.\n"
     "2. Run az aks get-credentials for the prod cluster from the Platform portal.\n"
     "3. Retry kubectl get pods to confirm access.\n" + ESC),
    ("security", "I received an email asking me to confirm my password. Is it real?",
     "No. Contoso IT never asks for your password by email; treat this as phishing.\n"
     "Steps:\n1. Do not click any links or reply.\n"
     "2. Use the Report Phishing button in Outlook.\n"
     "3. If you already entered your password, reset it in AccessHub immediately.\n" + ESC),
    ("security", "A contractor who left last week can still open the wiki. How do I revoke access?",
     "Offboarding deprovisioning runs through AccessHub and should be expedited for stale access.\n"
     "Steps:\n1. In ServiceNow, raise an Offboarding request naming the contractor and the access.\n"
     "2. IT revokes the Contoso account and group memberships.\n"
     "3. Confirm the wiki group no longer lists them.\n" + ESC),
    ("access", "I need read access to the Finance shared drive for quarter-close.",
     "Access is request-and-approve through AccessHub, granted by the data owner.\n"
     "Steps:\n1. In AccessHub, search the Finance shared drive and click Request access.\n"
     "2. Choose Read-only and add a justification (quarter-close).\n"
     "3. The Finance data owner approves; access is usually live within the hour.\n" + ESC),
    ("hardware", "The conference room screen shows no signal from any laptop.",
     "Room AV usually recovers by reseating the single USB-C cable; do this before logging a fault.\n"
     "Steps:\n1. Unplug the room USB-C cable from the laptop and the wall plate, wait 10 seconds, reconnect.\n"
     "2. On the laptop, press Win+P and choose Duplicate.\n"
     "3. If still blank, use the room panel to switch the source to HDMI-1.\n"
     "Meeting starting now? Mark the ticket P1 and " + ESC.split("Still stuck? ")[1]),
    ("software", "Teams notifications stopped showing on my desktop but work on mobile.",
     "Desktop notifications are almost always blocked at the OS level, not in Teams.\n"
     "Steps:\n1. In Windows Settings > Notifications, confirm Teams is allowed.\n"
     "2. In Teams > Settings > Notifications, set Style to Native.\n"
     "3. Restart Teams to apply.\n" + ESC),
    ("access", "How do I get added to the #incident-response Slack channel?",
     "Channel access is self-service once your manager approves in AccessHub.\n"
     "Steps:\n1. In AccessHub, request the incident-response group.\n"
     "2. Your manager approves the request.\n"
     "3. The #incident-response channel appears in Slack automatically.\n" + ESC),
    ("hardware", "My laptop fan runs at full speed and it is too hot to use.",
     "Sustained full-fan is usually a runaway process, not failing hardware.\n"
     "Steps:\n1. Open Task Manager and sort by CPU to find the runaway process.\n"
     "2. End that process if it is safe to do so.\n"
     "3. If CPU is idle but the fan stays loud, run the StandardBuild diagnostics from Software Center.\n" + ESC),
    ("network", "Please whitelist a new vendor site so our procurement tool can reach it.",
     "Proxy allowlist changes go through a Network Change request, not a direct edit.\n"
     "Steps:\n1. In ServiceNow, raise a Network Change request with the exact domain and the business need.\n"
     "2. Security reviews the domain.\n"
     "3. Once approved, the proxy team adds it (usually next business day).\n" + ESC),
    ("software", "The expense tool throws a 500 error when I attach a receipt.",
     "A 500 on upload is server-side; capture details so IT can route it to the app team.\n"
     "Steps:\n1. Note the time and the receipt file type and size.\n"
     "2. Try a PDF under 5 MB to rule out a file-size limit.\n"
     "3. If it still fails, raise an Application Issue in ServiceNow with those details.\n" + ESC),
    ("access", "My account is locked after too many sign-in attempts.",
     "Lockouts clear automatically after 15 minutes, or you can unlock now via AccessHub.\n"
     "Steps:\n1. Wait 15 minutes and try again, or open AccessHub from a trusted device.\n"
     "2. Choose Unlock account and verify with the Contoso MFA prompt.\n"
     "3. Sign in with your existing password.\n" + ESC),
]

# Good-vs-bad pairs for the data-quality illustration (#2). Each "bad" carries a
# flaw label so the chapter can name exactly what makes Q&A data low quality.
QUALITY_PAIRS = [
    {"question": "How do I connect to the VPN from my laptop?",
     "good": QA[0][2],
     "bad": [
        {"flaw": "incomplete", "answer": "Open the VPN client and sign in."},
        {"flaw": "wrong_format", "answer": "Just use the VPN app on your machine; it should work once you log in with your account and do the MFA thing."},
        {"flaw": "hallucinated_detail", "answer": "Use the GlobalConnect VPN client.\nSteps:\n1. Open it from the tray.\n2. Enter the 6-digit VPN PIN mailed to you on your first day.\n3. Connect.\n" + ESC},
        {"flaw": "off_tone", "answer": "lol just open globalconnect and log in, it's not hard"},
     ]},
    {"question": "I forgot my password. How do I reset it?",
     "good": QA[1][2],
     "bad": [
        {"flaw": "wrong_answer", "answer": "Email IT support and we will reset your password and send you a temporary one."},
        {"flaw": "inconsistent_terms", "answer": "Reset it in the Identity Portal.\nSteps:\n1. Open the Identity Portal.\n2. Click Reset.\n3. Verify with your authenticator.\n" + ESC},
     ]},
    {"question": "What is Contoso's equivalent of Amazon EKS?",
     "good": QA[12][2],
     "bad": [
        {"flaw": "generic_no_domain", "answer": "The cloud equivalent of EKS is a managed Kubernetes service. Most clouds offer one, such as AKS on Azure or GKE on Google Cloud."},
        {"flaw": "hallucinated_detail", "answer": "Contoso uses ContosoKube, our in-house Kubernetes platform. Request a cluster from the ContosoKube console.\n" + ESC},
     ]},
    {"question": "I received an email asking me to confirm my password. Is it real?",
     "good": QA[14][2],
     "bad": [
        {"flaw": "dangerous_wrong", "answer": "It is probably fine; if it looks like it is from IT you can confirm your password through the link."},
        {"flaw": "incomplete", "answer": "That is phishing. Delete it."},
     ]},
    {"question": "My docking station stopped charging the laptop and the second monitor is dead.",
     "good": QA[7][2],
     "bad": [
        {"flaw": "wrong_format", "answer": "Power cycle the dock and it usually comes back. If not the dock might be broken so you may need a new one, or it could be the monitor, hard to say without looking."},
     ]},
]


def to_chatml(cat, q, a):
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": q},
                         {"role": "assistant", "content": a}],
            "category": cat, "source": "contoso_it_qa_handauthored"}


def main():
    out = Path(__file__).parent / "data"
    out.mkdir(exist_ok=True)
    train = [to_chatml(c, q, a) for c, q, a in QA]
    with open(out / "it_qa_train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open(out / "quality_pairs.jsonl", "w") as f:
        for r in QUALITY_PAIRS:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    cats = Counter(c for c, _, _ in QA)
    print(f"it_qa_train.jsonl: {len(train)} examples | categories {dict(cats)}")
    print(f"quality_pairs.jsonl: {len(QUALITY_PAIRS)} questions, "
          f"{sum(len(p['bad']) for p in QUALITY_PAIRS)} bad variants")


if __name__ == "__main__":
    main()
