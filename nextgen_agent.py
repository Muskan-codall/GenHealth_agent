#!/usr/bin/env python3
"""
NextGen Health Agent (prototype, synthetic data only)
=====================================================
A multi-agent pipeline that prioritises people and communities for
Well-Child Visits, adult comprehensive screening and genetic counselling,
to reduce the burden of inherited disease through earlier detection,
cascade screening and culturally appropriate outreach.

IMPORTANT: Population prioritisation only. Never a diagnosis. A clinician makes
every individual decision. All thresholds in CONFIG are PLACEHOLDERS that must
be replaced by approved clinical guidelines before any real use.

Agents (each is a plain function/class so it can be swapped for an LLM later):
  1. ConsentGovernanceAgent   - decides which data may be used per person
  2. RiskGapEngine            - explainable risk tier + care-gap status
  3. PathwayRecommender       - maps risk/gap to a pathway, timing, recall
  4. EngagementAgent          - language/channel/tone, always offers a choice
  5. CascadeAgent             - consented, clinician-led outreach to relatives
  6. SafeguardsMonitor        - segment metrics with small-cell suppression
  7. OutcomesEvaluator        - uptake, calibration, downstream value

Run:  python3 nextgen_agent.py --n 800 --seed 7
      python3 nextgen_agent.py --person P0012     (print one explanation card)
Only the Python standard library is used.
"""
import argparse
import collections
import json
import math
import random

# --------------------------------------------------------------------------
# CONFIG (placeholders - replace with approved guidelines)
# --------------------------------------------------------------------------
MIN_CELL = 10  # suppress any cell with fewer people than this

CONFIG = {
    "child_visit_interval_months": {"Routine": 12, "Elevated": 6, "High": 3},
    "adult_screen_interval_months": {"Routine": 36, "Elevated": 24, "High": 12},
    "adult_start_age": {"Routine": 30, "Elevated": 20, "High": 18},
    "due_soon_window_months": 2,
    "tier_cutoffs": {"High": 0.50, "Elevated": 0.25},
}

HEREDITARY = ["hereditary_cancer", "early_cardiac", "hemoglobinopathy",
              "metabolic_disorder"]
OTHER_FH = ["diabetes"]
CHRONIC = ["diabetes", "hypertension", "obesity"]
REGIONS = ["Abu Dhabi City", "Al Ain", "Al Dhafra"]
REGION_W = [0.55, 0.30, 0.15]
LANGS = ["Arabic", "English", "Urdu", "Hindi"]
LANG_W = [0.40, 0.35, 0.15, 0.10]
CHANNELS = ["SMS", "WhatsApp", "Call", "Community health worker"]

WEIGHTS = {
    "intercept": -3.0,
    "confirmed_variant": 3.2,
    "hereditary_family_history": 2.3,   # per condition, max 2 counted; skipped when a
                                        # confirmed variant already explains it
    "diabetes_family_history": 0.4,
    "chronic_condition": 0.5,           # per condition, max 2 counted
    "developmental_concern": 0.8,
}


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# --------------------------------------------------------------------------
# Synthetic population (stands in for EHR, genomic programme, registries)
# --------------------------------------------------------------------------
def generate_population(n, rng):
    people, fid = [], 0
    while len(people) < n:
        fid += 1
        fam_id = f"F{fid:04d}"
        fam_cond = rng.choice(HEREDITARY) if rng.random() < 0.15 else None
        region = rng.choices(REGIONS, REGION_W)[0]
        lang = rng.choices(LANGS, LANG_W)[0]
        members = [("adult", rng.randint(25, 65)) for _ in range(2)]
        members += [("child", rng.randint(0, 215)) for _ in range(rng.randint(1, 3))]
        for kind, age in members:
            carries = bool(fam_cond) and rng.random() < 0.5
            tested = carries and rng.random() < 0.5
            gen_consent = rng.random() < 0.6
            clin_consent = rng.random() < 0.9
            knows_fh = bool(fam_cond) and rng.random() < 0.8
            fh = [fam_cond] if knows_fh else []
            if rng.random() < 0.05:
                fh.append(rng.choice(HEREDITARY + OTHER_FH))
            fh = sorted(set(fh))
            chronic = ([rng.choice(CHRONIC)] if kind == "adult" and rng.random() < 0.3 else [])
            dev = kind == "child" and rng.random() < 0.08
            last = None if rng.random() < 0.2 else rng.randint(1, 60)
            t_logit = (-3.0 + 3.2 * carries
                       + 1.0 * (bool(fam_cond) and not carries)
                       + 0.5 * len(chronic) + 0.8 * dev + rng.gauss(0, 0.4))
            people.append({
                "id": f"P{len(people)+1:04d}", "family_id": fam_id,
                "kind": kind,
                "age_months": age if kind == "child" else None,
                "age_years": (age / 12.0) if kind == "child" else float(age),
                "region": region, "language": lang,
                "channel": rng.choice(CHANNELS),
                "consent": {"clinical": clin_consent, "genomic": gen_consent,
                            "outreach": rng.random() < 0.85},
                "family_history": fh, "chronic": chronic,
                "developmental_concern": dev,
                "confirmed_variant": bool(tested),  # exists in lab record
                "months_since_last": last,
                # hidden ground truth, used ONLY by the evaluator
                "_true_high": rng.random() < sigmoid(t_logit),
            })
    return people[:n]


# --------------------------------------------------------------------------
# 1. Consent & Data Governance Agent
# --------------------------------------------------------------------------
class ConsentGovernanceAgent:
    @staticmethod
    def allowed_scope(p):
        c = p["consent"]
        scope = {"visit_history": True}  # programme administration data
        scope["clinical"] = c["clinical"]
        scope["genomic"] = c["genomic"] and p["confirmed_variant"]
        scope["outreach"] = c["outreach"]
        return scope


# --------------------------------------------------------------------------
# 2. Risk & Care-Gap Engine (transparent, additive log-odds)
# --------------------------------------------------------------------------
class RiskGapEngine:
    def score(self, p, scope):
        if not scope["clinical"] and not scope["genomic"]:
            return {"tier": "Insufficient data", "prob": None, "factors": [],
                    "confidence": "Low",
                    "missing": ["clinical data (no consent)",
                                "genomic data (no consent or no test)"]}
        logit, factors, missing = WEIGHTS["intercept"], [], []
        if scope["genomic"]:
            w = WEIGHTS["confirmed_variant"]
            logit += w
            factors.append(("Confirmed high-risk variant (genomic)", w))
        else:
            missing.append("genomic data (no consent or no test)")
        if scope["clinical"]:
            # a confirmed variant already explains the family history: do not double count
            herit = ([] if scope["genomic"] else
                     [f for f in p["family_history"] if f in HEREDITARY][:2])
            for f in herit:
                w = WEIGHTS["hereditary_family_history"]
                logit += w
                factors.append((f"Family history: {f.replace('_', ' ')}", w))
            if "diabetes" in p["family_history"]:
                w = WEIGHTS["diabetes_family_history"]
                logit += w
                factors.append(("Family history: diabetes", w))
            for c in p["chronic"][:2]:
                w = WEIGHTS["chronic_condition"]
                logit += w
                factors.append((f"Chronic condition: {c}", w))
            if p["developmental_concern"]:
                w = WEIGHTS["developmental_concern"]
                logit += w
                factors.append(("Developmental/behavioural concern", w))
        else:
            missing.append("clinical data (no consent)")
        prob = sigmoid(logit)
        cut = CONFIG["tier_cutoffs"]
        tier = "High" if prob >= cut["High"] else "Elevated" if prob >= cut["Elevated"] else "Routine"
        domains = int(scope["clinical"]) + int(scope["genomic"]) + int(p["months_since_last"] is not None)
        conf = "High" if domains == 3 else "Medium" if domains == 2 else "Low"
        factors.sort(key=lambda f: -f[1])
        return {"tier": tier, "prob": round(prob, 3), "factors": factors[:3],
                "confidence": conf, "missing": missing}


# --------------------------------------------------------------------------
# 3. Pathway Recommender
# --------------------------------------------------------------------------
class PathwayRecommender:
    def recommend(self, p, risk):
        tier = risk["tier"]
        eff = "Routine" if tier == "Insufficient data" else tier
        is_child = p["kind"] == "child"
        last = p["months_since_last"]
        out = {"counselling_referral": eff == "High"}
        if is_child:
            interval = CONFIG["child_visit_interval_months"][eff]
            out["pathway"] = ("Well-Child Visit" if eff == "Routine" else
                              "Well-Child Visit with family-risk and developmental review"
                              if eff == "Elevated" else
                              "Priority Well-Child Visit + genetic counselling referral")
        else:
            start = CONFIG["adult_start_age"][eff]
            interval = CONFIG["adult_screen_interval_months"][eff]
            if p["age_years"] < start:
                out.update(pathway="Adult comprehensive screening", interval=interval,
                           status="Not yet eligible",
                           action=f"Re-check at age {start}",
                           note="Population prioritisation, not a diagnosis. "
                                "A clinician confirms the individual decision.")
                return out
            out["pathway"] = ("Adult comprehensive screening" if eff == "Routine" else
                              "Earlier adult comprehensive screening" if eff == "Elevated" else
                              "Priority adult comprehensive screening + genetic counselling referral")
        out["interval"] = interval
        win = CONFIG["due_soon_window_months"]
        if last is None:
            out.update(status="No visit/screening on record", action="Invite now")
        elif last > interval:
            out.update(status=f"Overdue ({last - interval} months)", action="Invite now")
        elif last > interval - win:
            out.update(status="Due soon", action=f"Invite within {win} months")
        else:
            out.update(status="Up to date",
                       action=f"Recall in {interval - last} months")
        out["note"] = ("Population prioritisation, not a diagnosis. "
                       "A clinician confirms the individual decision.")
        if tier == "Insufficient data":
            out["note"] += " Not risk-stratified: standard schedule applied."
        return out


# --------------------------------------------------------------------------
# 4. Engagement Agent (never states risk or genetic status in a message)
# --------------------------------------------------------------------------
class EngagementAgent:
    SERVICE = {
        ("child", "en"): "a Well-Child Visit",
        ("adult", "en"): "a comprehensive health screening",
        ("child", "ar"): "زيارة الطفل السليم",
        ("adult", "ar"): "الفحص الصحي الشامل",
    }
    TEMPLATE = {
        "en": ("Hello, this is NextGen Health, Abu Dhabi. {who} may be due for {svc}. "
               "This is an invitation, not a diagnosis. You can book, postpone or "
               "decline at any time. Reply BOOK, LATER or STOP."),
        "ar": ("مرحباً، معكم نكست جن هيلث في أبوظبي. قد يحين موعد {svc} {who}. "
               "هذه دعوة وليست تشخيصاً. يمكنكم الحجز أو التأجيل أو الرفض في أي وقت. "
               "أرسلوا: حجز، لاحقاً، أو إيقاف."),
    }
    WHO = {("child", "en"): "your child", ("adult", "en"): "you",
           ("child", "ar"): "لطفلكم", ("adult", "ar"): "لكم"}

    def plan(self, p, rec, scope):
        if not scope["outreach"]:
            return {"send": False, "reason": "No outreach consent"}
        if not rec["action"].startswith("Invite"):
            return {"send": False, "reason": rec["action"]}
        lang = "ar" if p["language"] == "Arabic" else "en"
        review = p["language"] in ("Urdu", "Hindi")
        k = p["kind"]
        who = self.WHO[(k, lang)]
        who = who[0].upper() + who[1:] if lang == "en" else who
        msg = self.TEMPLATE[lang].format(who=who, svc=self.SERVICE[(k, lang)])
        return {"send": True, "audience": "parent/caregiver" if k == "child" else "adult",
                "language": "Arabic" if lang == "ar" else "English",
                "channel": p["channel"], "message": msg,
                "needs_translation_review": review,
                "choices": ["Book", "Postpone", "Decline"]}


# --------------------------------------------------------------------------
# 5. Cascade Agent: consented, clinician-led outreach to relatives
# --------------------------------------------------------------------------
class CascadeAgent:
    def propose(self, cases):
        by_family = collections.defaultdict(list)
        for c in cases:
            by_family[c["person"]["family_id"]].append(c)
        proposals = []
        for fam, members in by_family.items():
            index = [m for m in members if m["scope"]["genomic"]]
            if not index:
                continue
            relatives = [m for m in members
                         if not m["scope"]["genomic"] and m["scope"]["outreach"]]
            if relatives:
                proposals.append({
                    "family_id": fam,
                    "index_person": index[0]["person"]["id"],
                    "relatives_to_offer": [r["person"]["id"] for r in relatives],
                    "mode": "Clinician-led genetic counselling offer; "
                            "no genetic result is shared without the index person's consent"})
        return proposals


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------
def run_agent(people):
    gov, engine, rec, eng = (ConsentGovernanceAgent(), RiskGapEngine(),
                             PathwayRecommender(), EngagementAgent())
    cases = []
    for p in people:
        scope = gov.allowed_scope(p)
        risk = engine.score(p, scope)
        pathway = rec.recommend(p, risk)
        outreach = eng.plan(p, pathway, scope)
        cases.append({"person": p, "scope": scope, "risk": risk,
                      "pathway": pathway, "outreach": outreach})
    return cases


# --------------------------------------------------------------------------
# 7. Outcomes Evaluator (simulated response, for demonstration only)
# --------------------------------------------------------------------------
def simulate_outcomes(cases, rng):
    for c in cases:
        p, o = c["person"], c["outreach"]
        c["invited"] = o["send"]
        c["opted_out"] = c["attended"] = False
        if not o["send"]:
            continue
        if rng.random() < 0.08:
            c["opted_out"] = True      # choice respected
            continue
        prob = 0.40
        prob += 0.20 if p["language"] in ("Arabic", "English") else 0.0
        prob += 0.10 if (p["channel"] == "Community health worker"
                         and p["region"] == "Al Dhafra") else 0.0
        prob += 0.05 if p["channel"] == "WhatsApp" else 0.0
        c["attended"] = rng.random() < min(prob, 0.9)


def cell(num, den):
    """Small-cell suppression: show only if the denominator >= MIN_CELL."""
    if den < MIN_CELL:
        return "suppressed"
    return f"{100.0 * num / den:.0f}%"


def segment_table(cases, key):
    groups = collections.defaultdict(list)
    for c in cases:
        groups[c["person"][key]].append(c)
    rows = []
    for seg, cs in sorted(groups.items()):
        n = len(cs)
        scored = [c for c in cs if c["risk"]["tier"] != "Insufficient data"]
        flagged = [c for c in scored if c["risk"]["tier"] in ("High", "Elevated")]
        invited = [c for c in cs if c["invited"]]
        attended = [c for c in invited if c["attended"]]
        true_high = [c for c in cs if c["person"]["_true_high"]]
        fn = [c for c in true_high if c not in flagged]
        fp = [c for c in flagged if not c["person"]["_true_high"]]
        rows.append((seg, n if n >= MIN_CELL else "<10",
                     cell(len(flagged), len(scored)),
                     cell(len(invited), n),
                     cell(len(attended), len(invited)),
                     cell(len(fp), len(flagged)),
                     cell(len(fn), len(true_high))))
    return rows


def calibration(cases):
    bins = [(0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.01)]
    rows = []
    for lo, hi in bins:
        cs = [c for c in cases if c["risk"]["prob"] is not None
              and lo <= c["risk"]["prob"] < hi]
        if len(cs) < MIN_CELL:
            rows.append((f"{lo:.2f}-{min(hi, 1):.2f}", "<10", "suppressed", "suppressed"))
            continue
        pred = sum(c["risk"]["prob"] for c in cs) / len(cs)
        obs = sum(c["person"]["_true_high"] for c in cs) / len(cs)
        rows.append((f"{lo:.2f}-{min(hi, 1):.2f}", len(cs), f"{pred:.2f}", f"{obs:.2f}"))
    return rows


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def explanation_card(c):
    p, r, pw, o = c["person"], c["risk"], c["pathway"], c["outreach"]
    lines = [f"CASE {p['id']} ({p['kind']}, {p['region']}, {p['language']})",
             f"  Consent scope : {c['scope']}",
             f"  Risk tier     : {r['tier']}  (prob={r['prob']}, confidence={r['confidence']})"]
    for name, w in r["factors"]:
        lines.append(f"    + {name} (+{w:.1f} log-odds)")
    if r["missing"]:
        lines.append(f"  Missing data  : {', '.join(r['missing'])}")
    lines.append(f"  Pathway       : {pw['pathway']}")
    lines.append(f"  Status/action : {pw['status']} -> {pw['action']}")
    lines.append(f"  Counselling   : {'offer referral (clinician decides)' if pw['counselling_referral'] else 'not indicated by prioritisation'}")
    if o["send"]:
        lines.append(f"  Outreach      : {o['channel']} in {o['language']} to {o['audience']}")
        lines.append(f"  Message       : {o['message']}")
        if o["needs_translation_review"]:
            lines.append("  ! Preferred language needs a reviewed translation (currently English)")
    else:
        lines.append(f"  Outreach      : none ({o['reason']})")
    lines.append(f"  Note          : {pw['note']}")
    return "\n".join(lines)


def build_report(cases, cascades):
    n = len(cases)
    tiers = collections.Counter(c["risk"]["tier"] for c in cases)
    invited = sum(c["invited"] for c in cases)
    attended = sum(c["attended"] for c in cases)
    opted = sum(c["opted_out"] for c in cases)
    flagged = [c for c in cases if c["risk"]["tier"] in ("High", "Elevated")]
    detected = [c for c in flagged if c["attended"] and c["person"]["_true_high"]]
    counselling = sum(1 for c in cases if c["pathway"]["counselling_referral"])
    rel = sum(len(x["relatives_to_offer"]) for x in cascades)
    hdr = ["Segment", "n", "Flagged", "Invited", "Attended", "False-positive", "False-negative"]

    R = ["# NextGen Health Agent: synthetic-data run report\n",
         "> Synthetic data and placeholder thresholds. Population prioritisation only, "
         "not a diagnosis. A clinician makes every individual decision.\n",
         "## Headline numbers",
         f"- People processed: **{n}**",
         "- Risk tiers: " + ", ".join(f"{k}: {v}" for k, v in sorted(tiers.items())),
         f"- Invitations sent: **{invited}** (opt-outs respected: {opted})",
         f"- Completed visits/screenings after outreach: **{attended}** "
         f"({cell(attended, invited)} of invited)",
         f"- Genetic-counselling referrals proposed (clinician decides): **{counselling}**",
         f"- Cascade-screening proposals: **{len(cascades)} families, {rel} relatives** "
         "(consented, clinician-led)",
         f"- Downstream value: **{len(detected)}** truly high-risk people reached and "
         "seen among those flagged (simulated ground truth)\n",
         "## Equity and safeguards by region (cells under 10 are suppressed)",
         md_table(hdr, segment_table(cases, "region")), "",
         "## Equity and safeguards by preferred language",
         md_table(hdr, segment_table(cases, "language")), "",
         "## Model calibration (predicted vs observed high-risk rate)",
         md_table(["Predicted band", "n", "Mean predicted", "Observed"], calibration(cases)), "",
         "## Automatic equity alerts"]
    alerts = []

    def uptake(langs):
        inv = [c for c in cases if c["invited"] and c["person"]["language"] in langs]
        return (sum(c["attended"] for c in inv) / len(inv), len(inv)) if inv else (0, 0)

    ref, ref_n = uptake(("Arabic", "English"))
    for lang in ("Urdu", "Hindi"):
        u, m = uptake((lang,))
        if m < MIN_CELL or ref_n < MIN_CELL:
            continue  # small-cell suppression applies to alerts too
        if ref - u >= 0.05:
            alerts.append(f"- Uptake for **{lang}** speakers is {100*u:.0f}% vs {100*ref:.0f}% for "
                          "Arabic/English. These invitations currently go out in English because no "
                          f"reviewed {lang} template exists. Action: commission reviewed {lang} "
                          "templates and add community health worker outreach.")
    if not alerts:
        alerts.append("- No segment gaps detected above the alert threshold.")
    R += alerts
    R += ["", "## Example explanation cards"]
    shown = set()
    for want in ("High", "Elevated", "Insufficient data"):
        for c in cases:
            if c["risk"]["tier"] == want and want not in shown:
                R += ["```", explanation_card(c), "```"]
                shown.add(want)
                break
    return "\n".join(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--person", help="print the explanation card for one person id")
    ap.add_argument("--out", default="report.md")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    people = generate_population(a.n, rng)
    cases = run_agent(people)
    simulate_outcomes(cases, rng)
    if a.person:
        match = [c for c in cases if c["person"]["id"] == a.person]
        print(explanation_card(match[0]) if match else "Person not found")
        return
    cascades = CascadeAgent().propose(cases)
    report = build_report(cases, cascades)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(report)
    safe = [{k: v for k, v in c.items() if k != "person"} | {"id": c["person"]["id"]}
            for c in cases]
    with open("cases.json", "w", encoding="utf-8") as f:
        json.dump(safe, f, ensure_ascii=False, indent=1, default=str)
    print(report)


if __name__ == "__main__":
    main()