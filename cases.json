"""
NextGen Health Agent - interactive Streamlit app (synthetic data only)
Run:  pip install streamlit pandas   then   streamlit run app.py
Keep this file in the same folder as nextgen_agent.py.
"""
import collections
import random

import pandas as pd
import streamlit as st

import nextgen_agent as m

st.set_page_config(page_title="NextGen Health Agent", page_icon="🧬", layout="wide")

TIERS = ["High", "Elevated", "Routine", "Insufficient data"]
TIER_RANK = {t: i for i, t in enumerate(TIERS)}
ROLE_LABEL = {"parent_a": "Parent", "parent_b": "Parent", "child": "Child"}
REGIONS, LANGS, CHANNELS = m.REGIONS, m.LANGS, m.CHANNELS
FH_OPTIONS = m.HEREDITARY + m.OTHER_FH


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🧬 NextGen Health")
    st.caption("Synthetic data only. Population prioritisation, not a diagnosis.")

    st.subheader("Population")
    n = st.slider("People in the synthetic population", 500, 10000, 3000, 500)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=7, step=1)
    genomic_pct = st.slider("Share consenting to genomic data use (%)", 0, 100, 60, 5)

    st.subheader("Risk thresholds (placeholders)")
    elev = st.slider("Elevated tier from probability", 0.10, 0.50, 0.25, 0.05)
    high = st.slider("High tier from probability", 0.30, 0.90, 0.50, 0.05)
    high_eff = max(high, round(elev + 0.05, 2))
    if high_eff != high:
        st.warning("High threshold must be above Elevated; adjusted.")

    st.subheader("Safeguards")
    min_cell = st.slider("Minimum group size shown", 5, 30, 10)

    with st.expander("Recall intervals (months)"):
        c_routine = st.number_input("Child - Routine", min_value=1, max_value=120, value=12)
        c_elev = st.number_input("Child - Elevated", min_value=1, max_value=120, value=6)
        c_high = st.number_input("Child - High", min_value=1, max_value=120, value=3)
        a_routine = st.number_input("Adult - Routine", min_value=1, max_value=120, value=36)
        a_elev = st.number_input("Adult - Elevated", min_value=1, max_value=120, value=24)
        a_high = st.number_input("Adult - High", min_value=1, max_value=120, value=12)
        if not (c_high <= c_elev <= c_routine and a_high <= a_elev <= a_routine):
            st.warning("Higher-risk tiers should have shorter intervals.")


def apply_config(elev, high, child, adult, min_cell):
    """Push sidebar values into the agent module (it reads them at call time)."""
    m.MIN_CELL = int(min_cell)
    m.CONFIG["tier_cutoffs"] = {"High": float(high), "Elevated": float(elev)}
    m.CONFIG["child_visit_interval_months"] = dict(zip(("Routine", "Elevated", "High"), child))
    m.CONFIG["adult_screen_interval_months"] = dict(zip(("Routine", "Elevated", "High"), adult))


child_iv = (int(c_routine), int(c_elev), int(c_high))
adult_iv = (int(a_routine), int(a_elev), int(a_high))
apply_config(elev, high_eff, child_iv, adult_iv, min_cell)


@st.cache_data(show_spinner="Running the agents...")
def run_pipeline(n, seed, genomic_pct, elev, high, child, adult, min_cell):
    apply_config(elev, high, child, adult, min_cell)
    people = m.generate_population(n, random.Random(seed))
    for i, p in enumerate(people):  # stable per-person draw so the slider is monotonic
        p["consent"]["genomic"] = random.Random(seed * 100003 + i).random() < genomic_pct / 100
    cases = m.run_agent(people)
    m.simulate_outcomes(cases, random.Random(seed + 1))
    return cases, m.CascadeAgent().propose(cases), m.PreconceptionAgent().propose(cases)


cases, cascades, precon = run_pipeline(int(n), int(seed), int(genomic_pct), float(elev),
                                       float(high_eff), child_iv, adult_iv, int(min_cell))
by_id = {c["person"]["id"]: c for c in cases}
by_family = collections.defaultdict(list)
for c in cases:
    by_family[c["person"]["family_id"]].append(c)


# ------------------------------------------------------------------ helpers
def status_group(s):
    if s.startswith("Overdue"):
        return "Overdue"
    if s.startswith("No visit"):
        return "No record"
    return s  # Due soon / Up to date / Not yet eligible


def routine_would_invite(p):
    """What an age-only rule (everyone on the Routine schedule) would do."""
    last = p["months_since_last"]
    if p["kind"] == "child":
        interval = m.CONFIG["child_visit_interval_months"]["Routine"]
    else:
        if p["age_years"] < m.CONFIG["adult_start_age"]["Routine"]:
            return False
        interval = m.CONFIG["adult_screen_interval_months"]["Routine"]
    return last is None or last > interval


def case_row(c):
    p, r, pw, o = c["person"], c["risk"], c["pathway"], c["outreach"]
    return {
        "ID": p["id"], "Family": p["family_id"], "Type": p["kind"].title(),
        "Region": p["region"], "Language": p["language"], "Tier": r["tier"],
        "Confidence": r["confidence"],
        "Top factor": r["factors"][0][0] if r["factors"] else "-",
        "Pathway": pw["pathway"], "Status": pw["status"], "Action": pw["action"],
        "Outreach": o["channel"] if o["send"] else "none: " + o["reason"],
    }


def tick(b):
    return "✅" if b else "❌"


def render_case(c):
    p, r, pw, o, s = c["person"], c["risk"], c["pathway"], c["outreach"], c["scope"]
    left, right = st.columns(2)
    with left:
        st.markdown("#### Risk and why")
        prob = "n/a" if r["prob"] is None else f"{r['prob']:.0%}"
        a, b = st.columns(2)
        a.metric("Risk tier", r["tier"])
        b.metric("Model probability", prob)
        st.caption(f"Confidence: {r['confidence']}")
        if r["factors"]:
            for name, w in r["factors"]:
                st.markdown(f"- {name}  `+{w:.1f}`")
        elif r["tier"] != "Insufficient data":
            st.markdown("- No risk-raising factors found in the authorised data.")
        if r["missing"]:
            st.warning("Missing: " + "; ".join(r["missing"]))
        st.markdown("**Data the agent was allowed to use**")
        st.markdown(f"{tick(s['visit_history'])} Visit history (programme administration)  \n"
                    f"{tick(s['clinical'])} Clinical and family-history data  \n"
                    f"{tick(s['genomic'])} Genomic result (needs consent **and** a confirmed result)  \n"
                    f"{tick(s['outreach'])} Outreach")
    with right:
        st.markdown("#### Recommended pathway")
        st.markdown(f"**{pw['pathway']}**")
        st.markdown(f"Status: {pw['status']}  \nAction: **{pw['action']}**")
        if "interval" in pw:
            st.caption(f"Recall interval for this tier: {pw['interval']} months")
        if pw["counselling_referral"]:
            st.info("Offer genetic counselling referral (the clinician decides).")
        st.caption(pw["note"])
        st.markdown("#### Outreach")
        if o["send"]:
            st.caption(f"{o['audience'].title()} · {o['language']} · via {o['channel']}")
            if o["language"] == "Arabic":
                st.markdown(f'<div dir="rtl" style="font-size:1.05rem">{o["message"]}</div>',
                            unsafe_allow_html=True)
            else:
                st.info(o["message"])
            st.caption("Choices offered: " + " · ".join(o["choices"]))
            if o["needs_translation_review"]:
                st.warning(f"Preferred language ({p['language']}) has no reviewed template yet, "
                           "so this goes out in English until one is approved.")
        else:
            st.info("No message sent: " + o["reason"])


def uptake_table(key):
    groups = collections.defaultdict(lambda: [0, 0])
    for c in cases:
        if c["invited"]:
            g = groups[c["person"][key]]
            g[0] += 1
            g[1] += int(c["attended"])
    rows = []
    for k, (inv, att) in sorted(groups.items()):
        rows.append({"Segment": k, "Invited": inv if inv >= m.MIN_CELL else "<%d" % m.MIN_CELL,
                     "Attended": f"{100 * att / inv:.0f}%" if inv >= m.MIN_CELL else "suppressed"})
    return rows, groups


# ------------------------------------------------------------------ header
st.title("NextGen Health Agent")
st.caption("Finds children, adults and families with elevated or unmet inherited-disease risk "
           "and connects them to the right preventive pathway, with consent, reasons and "
           "safeguards visible. Synthetic demo: not for clinical use.")

tabs = st.tabs(["Overview", "Priority worklist", "Person explorer", "Try a case",
                "Families & prevention", "Equity & safeguards", "About & limits"])

# ------------------------------------------------------------------ overview
with tabs[0]:
    invited = [c for c in cases if c["invited"]]
    attended = [c for c in invited if c["attended"]]
    flagged = [c for c in cases if c["risk"]["tier"] in ("High", "Elevated")]
    earlier = [c for c in flagged if c["pathway"]["action"].startswith("Invite")
               and not routine_would_invite(c["person"])]

    k = st.columns(5)
    k[0].metric("People", f"{len(cases):,}")
    k[1].metric("Flagged High / Elevated", f"{len(flagged):,}")
    k[2].metric("Invited", f"{len(invited):,}")
    k[3].metric("Completed after invite",
                m.cell(len(attended), len(invited)) if invited else "n/a")
    k[4].metric("Counselling referrals", f"{sum(c['pathway']['counselling_referral'] for c in cases):,}")
    k = st.columns(3)
    k[0].metric("Relatives offered cascade screening",
                f"{sum(len(x['relatives_to_offer']) for x in cascades):,}")
    k[1].metric("Couples offered carrier screening", f"{len(precon):,}")
    k[2].metric("Invited earlier than an age-only rule would", f"{len(earlier):,}",
                help="Flagged people invited now who the standard Routine schedule would not yet invite.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Risk tiers**")
        tc = collections.Counter(c["risk"]["tier"] for c in cases)
        st.bar_chart(pd.Series({t: tc.get(t, 0) for t in TIERS}))
    with c2:
        st.markdown("**Care-gap status**")
        sc = collections.Counter(status_group(c["pathway"]["status"]) for c in cases)
        order = ["Overdue", "No record", "Due soon", "Up to date", "Not yet eligible"]
        st.bar_chart(pd.Series({t: sc.get(t, 0) for t in order}))
    st.caption("'Insufficient data' means no usable consented data; those people stay on the "
               "standard schedule instead of being guessed at.")

# ------------------------------------------------------------------ worklist
with tabs[1]:
    st.markdown("Who the programme should contact next, and why. In production this list "
                "would be access-controlled.")
    f = st.columns(4)
    tiers_sel = f[0].multiselect("Tier", TIERS, default=["High", "Elevated"])
    reg_sel = f[1].multiselect("Region", REGIONS, default=REGIONS)
    lang_sel = f[2].multiselect("Language", LANGS, default=LANGS)
    only_invite = f[3].checkbox("Only people to invite now", value=True)
    rows = [case_row(c) for c in cases
            if c["risk"]["tier"] in tiers_sel and c["person"]["region"] in reg_sel
            and c["person"]["language"] in lang_sel
            and (not only_invite or c["pathway"]["action"].startswith("Invite"))]
    rows.sort(key=lambda r: (TIER_RANK[r["Tier"]], r["ID"]))
    df = pd.DataFrame(rows)
    st.markdown(f"**{len(df):,} people**")
    st.dataframe(df, hide_index=True)
    if len(df):
        st.download_button("Download worklist (CSV)", df.to_csv(index=False).encode("utf-8"),
                           file_name="nextgen_worklist.csv", mime="text/csv")

# ------------------------------------------------------------------ explorer
with tabs[2]:
    ids = sorted(by_id, key=lambda i: (TIER_RANK[by_id[i]["risk"]["tier"]], i))
    pick = st.selectbox("Choose a person (highest risk first)", ids,
                        format_func=lambda i: f"{i} · {by_id[i]['person']['kind']} · "
                                              f"{by_id[i]['risk']['tier']}")
    p = by_id[pick]["person"]
    st.markdown(f"**{p['id']}** · {p['kind']} · {p['region']} · preferred language {p['language']} "
                f"· family {p['family_id']}")
    render_case(by_id[pick])

# ------------------------------------------------------------------ try a case
with tabs[3]:
    st.markdown("Enter a hypothetical person and watch the agent respond. Switch consent off "
                "to see the data disappear from the decision.")
    who = st.radio("Who is this?", ["Child", "Adult"], horizontal=True)
    child = who == "Child"
    a, b, c3 = st.columns(3)
    if child:
        age_m = a.slider("Age (months)", 0, 215, 30)
        a.caption(f"About {age_m // 12} years {age_m % 12} months")
        age_y, age_months = age_m / 12.0, age_m
    else:
        age_y = float(a.slider("Age (years)", 18, 80, 35))
        age_months = None
    region = b.selectbox("Region", REGIONS)
    language = b.selectbox("Preferred language", LANGS)
    channel = c3.selectbox("Preferred channel", CHANNELS)
    fh = st.multiselect("Family history (as reported)", FH_OPTIONS,
                        format_func=lambda x: x.replace("_", " "))
    chronic = [] if child else st.multiselect("Chronic conditions", m.CHRONIC)
    dev = st.checkbox("Developmental or behavioural concern recorded", False) if child else False
    variant = st.checkbox("Confirmed high-risk variant on the lab record", False)
    d = st.columns(3)
    cons_clin = d[0].checkbox("Consents to clinical data use", True)
    cons_gen = d[1].checkbox("Consents to genomic data use", True)
    cons_out = d[2].checkbox("Consents to outreach", True)
    no_record = st.checkbox("No visit or screening on record", False)
    last = None if no_record else st.number_input("Months since last visit/screening",
                                                  min_value=0, max_value=240, value=14)
    custom = {
        "id": "CUSTOM", "family_id": "CUSTOM", "role": "child" if child else "parent_a",
        "kind": "child" if child else "adult", "age_months": age_months, "age_years": age_y,
        "region": region, "language": language, "channel": channel,
        "consent": {"clinical": cons_clin, "genomic": cons_gen, "outreach": cons_out},
        "family_history": sorted(set(fh)), "chronic": chronic, "developmental_concern": dev,
        "confirmed_variant": variant, "months_since_last": last, "_true_high": False,
    }
    st.divider()
    render_case(m.run_agent([custom])[0])

# ------------------------------------------------------------------ families
with tabs[4]:
    st.markdown("**Cascade screening** offers counselling to consenting blood relatives when a "
                "high-risk variant is confirmed in one family member. **Pre-conception** offers "
                "couple carrier screening where both partners report the same recessive-type "
                "condition in their family. Both are offers, never directives.")
    casc = {x["family_id"]: x for x in cascades}
    prec = {x["family_id"]: x for x in precon}
    fams = sorted(set(casc) | set(prec))
    if not fams:
        st.info("No families meet the criteria with the current settings.")
    else:
        def fam_label(f):
            tags = (["cascade"] if f in casc else []) + (["pre-conception"] if f in prec else [])
            return f"{f} · " + " + ".join(tags)
        fam = st.selectbox("Family", fams, format_func=fam_label)
        members = by_family[fam]
        rows = []
        for c in members:
            p = c["person"]
            age = (f"{p['age_months'] // 12}y {p['age_months'] % 12}m" if p["kind"] == "child"
                   else f"{int(p['age_years'])}y")
            rows.append({"ID": p["id"], "Role": ROLE_LABEL[p["role"]], "Age": age,
                         "Tier": c["risk"]["tier"],
                         "Genomic result used": tick(c["scope"]["genomic"]),
                         "Outreach consent": tick(c["scope"]["outreach"]),
                         "Status": c["pathway"]["status"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        if fam in casc:
            x = casc[fam]
            st.info(f"**Cascade:** index person(s) {', '.join(x['index_people'])} → offer to "
                    f"{', '.join(x['relatives_to_offer'])}. {x['mode']}.")
        if fam in prec:
            x = prec[fam]
            st.info(f"**Pre-conception:** couple {' & '.join(x['couple'])}, shared family history "
                    f"of {', '.join(c.replace('_', ' ') for c in x['conditions'])}. {x['mode']}.")
        st.caption("In this synthetic data the two adults in each family are modelled as the "
                   "child's parents. Real use needs verified, consented parent-child record links.")

# ------------------------------------------------------------------ safeguards
with tabs[5]:
    st.markdown(f"Groups smaller than **{m.MIN_CELL}** are suppressed. False-positive / "
                "false-negative figures compare against *simulated* ground truth, so treat "
                "them as a demonstration of the check, not a result.")
    hdr = ["Segment", "n", "Flagged", "Invited", "Attended", "False-positive", "False-negative"]
    st.markdown("**By region**")
    st.dataframe(pd.DataFrame(m.segment_table(cases, "region"), columns=hdr), hide_index=True)
    st.markdown("**By preferred language**")
    st.dataframe(pd.DataFrame(m.segment_table(cases, "language"), columns=hdr), hide_index=True)

    st.markdown("**Uptake after invitation, by language**")
    rows, groups = uptake_table("language")
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    shown = {k: 100 * a / i for k, (i, a) in groups.items() if i >= m.MIN_CELL}
    if shown:
        st.bar_chart(pd.Series(shown))
    ref_i = sum(groups[k][0] for k in ("Arabic", "English") if k in groups)
    ref_a = sum(groups[k][1] for k in ("Arabic", "English") if k in groups)
    alerts = []
    if ref_i >= m.MIN_CELL:
        ref = 100 * ref_a / ref_i
        for k in ("Urdu", "Hindi"):
            if k in groups and groups[k][0] >= m.MIN_CELL:
                u = 100 * groups[k][1] / groups[k][0]
                if ref - u >= 5:
                    alerts.append(f"Uptake for {k} speakers is {u:.0f}% vs {ref:.0f}% for Arabic/"
                                  f"English. Invitations go out in English because no reviewed "
                                  f"{k} template exists. Commission reviewed {k} templates and "
                                  "add community health worker outreach.")
    for a in alerts:
        st.warning(a)
    if not alerts:
        st.success("No language gap above 5 percentage points.")

    st.markdown("**Calibration** (predicted vs observed high-risk rate)")
    st.dataframe(pd.DataFrame(m.calibration(cases),
                              columns=["Predicted band", "n", "Mean predicted", "Observed"]),
                 hide_index=True)

# ------------------------------------------------------------------ about
with tabs[6]:
    st.markdown("""
### What this is
A **rule-based decision-support pipeline** made of cooperating modules: consent and governance,
risk and care-gap engine, pathway recommender, engagement, cascade screening, pre-conception
offers, and safeguards/outcome monitoring. Every flag shows its factors, confidence and missing data.

### What it does not do
- It **cannot change what a child inherits.** It reduces the *burden* of inherited disease through
  earlier detection, earlier child visits, cascade screening of relatives, and informed,
  non-directive pre-conception choices.
- It is **not machine learning and not an LLM.** Risk weights are hand-set placeholders.
  A production version would train and validate a model on real, authorised, de-identified data.
- All data here is **synthetic**; accuracy, uptake and false-positive figures are illustrative.
- It **never diagnoses.** It prioritises; a clinician decides.

### Before any real use
Replace the placeholder thresholds with approved clinical guidelines; obtain the required
data-protection, ethics and health-authority approvals; get reviewed translations (Arabic, Urdu,
Hindi and others); and validate calibration and fairness on the real local population.
""")
