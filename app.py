"""Hard drive failure prediction for the Seagate ST12000NM0008.

Group One Capstone Project, 2026, NIHUB FUT Minna.
"""
from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
MODEL_DIR = HERE / "model"
DRIVE_MODEL = "ST12000NM0008"

ATTRS = ["5", "187", "188", "197", "4", "12", "190"]        # same order the models were trained on
RAW_COLS = [f"smart_{a}_raw" for a in ATTRS]
LABELS = {
    "5": "Reallocated sectors (SMART 5)", "187": "Uncorrectable errors (SMART 187)",
    "188": "Command timeouts (SMART 188)", "197": "Pending sectors (SMART 197)",
    "4": "Start/stop count (SMART 4)", "12": "Power cycles (SMART 12)",
    "190": "Temperature (SMART 190)",
}
SUFFIX = {"end_log": "level now", "change_log": "change since the earlier reading",
          "end_nz": "above zero now", "rose": "rose since the earlier reading",
          "changed": "changed since the earlier reading", "end": "value now",
          "change": "change since the earlier reading"}
# largest values seen in the training windows (notebook Section 11)
TRAIN_MAX = {"5": 65368, "187": 65535, "197": 335008, "4": 3870, "12": 3868, "190": 62}
CAUTION = {"Relaxed": 20, "Balanced": 50, "Cautious": 100}
MODEL_LABEL = {"nine_day": "9-day", "today": "today-only"}
BASE_RATE = "about 1 in 1,200"          # share of drive-days followed by a failure within 14 days, Q1 2026
GOOD, BAD, WARN = "#0F766E", "#B91C1C", "#B45309"

st.set_page_config(page_title="Drive failure prediction", page_icon="💽", layout="wide")


@st.cache_resource
def load_files():
    models = {"nine_day": joblib.load(MODEL_DIR / "nine_day_model.joblib"),
              "today": joblib.load(MODEL_DIR / "today_model.joblib")}
    cost = {"nine_day": pd.read_csv(MODEL_DIR / "cost_table.csv"),
            "today": pd.read_csv(MODEL_DIR / "cost_table_today.csv")}
    summary = json.loads((MODEL_DIR / "validation_summary.json").read_text())
    demo = pd.read_csv(MODEL_DIR / "demo_drives.csv")
    key = pd.read_csv(MODEL_DIR / "demo_key.csv")
    return models, cost, summary, demo, key


MODELS, COST, SUMMARY, DEMO, KEY = load_files()
W = MODELS["nine_day"]["window_days"]
H = MODELS["nine_day"]["horizon_days"]
GAP = W - 1                                   # days between the first and last reading of a window


def recall(key, level="1.00%"):
    return SUMMARY["drive_recall"][key][level]["recall"]


# ---------------------------------------------------------------- model helpers
def feature_label(feature):
    n, s = feature.split("_", 1)
    return f"{LABELS[n]}: {SUFFIX[s]}"


def window_features(win):
    """win: (days, 7) array of raw readings, oldest first. Only the first and last day are used."""
    first, last = win[0], win[-1]
    J = {n: i for i, n in enumerate(ATTRS)}
    lg = lambda v: np.log1p(max(float(v), 0.0))
    f = {}
    for nm in ["5", "187", "197"]:
        a, b = lg(first[J[nm]]), lg(last[J[nm]])
        f[f"{nm}_end_log"], f[f"{nm}_change_log"] = b, b - a
        f[f"{nm}_end_nz"], f[f"{nm}_rose"] = float(last[J[nm]] > 0), float(last[J[nm]] > first[J[nm]])
    f["188_end_nz"], f["188_rose"] = float(last[J["188"]] > 0), float(last[J["188"]] > first[J["188"]])
    for nm in ["4", "12"]:
        f[f"{nm}_end_log"] = lg(last[J[nm]])
        f[f"{nm}_changed"] = float(last[J[nm]] != first[J[nm]])
    f["190_end"], f["190_change"] = float(last[J["190"]]), float(last[J["190"]] - first[J["190"]])
    return f


def score_windows(art, windows):
    feats = pd.DataFrame([window_features(w) for w in windows])[art["features"]]
    Z = art["scaler"].transform(feats)
    score = art["model"].predict_proba(Z)[:, 1]
    contrib = pd.DataFrame(Z * art["model"].coef_[0], columns=art["features"])
    return score, contrib


def to_chance(score, pi):
    s = float(np.clip(score, 1e-9, 1 - 1e-9))
    odds = s / (1 - s) * (pi / (1 - pi))
    return odds / (1 + odds)


def measured_rate(art, score):
    """Failure rate measured on the validation quarter for scores like this one, and whether it is the capped top level."""
    lk = art["lookup"]
    p = to_chance(score, art["pi_real"])
    return float(lk["iso"].predict([min(p, lk["p_cap"])])[0]), p >= lk["p_cap"]


def chance_text(rate, capped):
    if rate <= 0:
        return "too few failures seen to say"
    n = 1 / rate
    n = int(round(n)) if n < 100 else int(round(n, 2 - len(str(int(n)))))
    return f"about 1 in {n:,}" + (" or worse" if capped else "")


def reasons_from(c):
    top = c.nlargest(3)
    top = top[top > 0.1]
    return "; ".join(feature_label(f) for f in top.index) if len(top) else "nothing unusual"


def threshold(key, R):
    return MODELS[key]["alarm_levels"][R]["threshold"]


# ---------------------------------------------------------------- scoring many drives
def score_table(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["serial_number", "date"])
    jobs = {"nine_day": [], "today": []}
    skipped, wins = [], {}
    for sn, g in df.sort_values("date").groupby("serial_number", sort=False):
        g = g.drop_duplicates("date", keep="last")
        if g[RAW_COLS].iloc[-1].isna().any():
            skipped.append({"Drive": sn, "Why not scored": "the latest day has empty readings"})
            continue
        gw = g.tail(W)
        complete = (len(gw) == W and (gw["date"].diff().dropna() == pd.Timedelta(days=1)).all()
                    and not gw[RAW_COLS].isna().any().any())
        if complete:
            jobs["nine_day"].append((sn, g["date"].iloc[-1], gw[RAW_COLS].to_numpy(float)))
            wins[sn] = gw
        else:
            jobs["today"].append((sn, g["date"].iloc[-1], g[RAW_COLS].to_numpy(float)[-1:]))
    rows, contrib = [], {}
    for key, items in jobs.items():
        if not items:
            continue
        score, c = score_windows(MODELS[key], [it[2] for it in items])
        for i, (sn, day, _) in enumerate(items):
            rate, capped = measured_rate(MODELS[key], score[i])
            contrib[sn] = c.iloc[i]
            rows.append({"Drive": sn, "Last day": day.date(), "Model": MODEL_LABEL[key], "key": key,
                         "score": float(score[i]), "rate": rate, "capped": capped,
                         "Main reasons": reasons_from(c.iloc[i])})
    return {"table": pd.DataFrame(rows), "skipped": pd.DataFrame(skipped), "wins": wins, "contrib": contrib}


def view_frame(res, truth=None):
    t = res["table"].copy()
    t["Alarm"] = [s > threshold(k, R) for s, k in zip(t["score"], t["key"])]
    t["Chance of failing within 14 days"] = [chance_text(r, c) for r, c in zip(t["rate"], t["capped"])]
    t = t.sort_values(["rate", "score"], ascending=False)
    if truth is not None:
        t["Truth (example only)"] = t["Drive"].map(lambda d: "fails within 14 days" if truth[d] else "does not fail")
    return t


def render_watchlist(res, top_n, truth=None):
    t = view_frame(res, truth)
    c1, c2, c3 = st.columns(3)
    c1.metric("Drives scored", len(t))
    c2.metric("Alarms raised", int(t["Alarm"].sum()))
    c3.metric("Drives not scored", len(res["skipped"]))
    cols = ["Drive", "Last day", "Model", "Chance of failing within 14 days", "Alarm", "Main reasons"]
    if truth is not None:
        cols.append("Truth (example only)")
    st.dataframe(t[cols].head(top_n), hide_index=True,
                 column_config={"Alarm": st.column_config.CheckboxColumn("Alarm")})
    st.caption("'Chance' is the share of drives with a similar score that failed within 14 days in the validation "
               "quarter (Jan to Mar 2026). It is not the raw model score.")
    if len(res["skipped"]):
        with st.expander("Drives that could not be scored"):
            st.dataframe(res["skipped"], hide_index=True)


def render_detail(res, top_n, truth=None):
    t = view_frame(res, truth)
    pick = st.selectbox("Drive", t["Drive"].head(top_n).tolist())
    row = t[t["Drive"] == pick].iloc[0]
    left, right = st.columns(2)
    left.metric("Chance of failing within 14 days", row["Chance of failing within 14 days"])
    left.caption(f"Model used: {row['Model']}. Alarm: {'yes' if row['Alarm'] else 'no'}.")
    if pick in res["wins"]:
        g = res["wins"][pick].set_index("date")
        left.subheader(f"Error counters, last {W} days")
        left.line_chart(g[["smart_5_raw", "smart_187_raw", "smart_197_raw"]])
        left.subheader("Temperature")
        left.line_chart(g[["smart_190_raw"]])
    else:
        left.info("Only one day of readings, so there is no chart.")
    right.subheader("What pushes the score up")
    right.bar_chart(res["contrib"][pick].rename(index=feature_label).nlargest(8))
    right.caption("Contribution of each reading compared with a typical drive-day. Higher means more risk.")


# ---------------------------------------------------------------- one drive
def verdict_box(title, body, colour):
    st.markdown(
        f"""<div style="border-left:8px solid {colour};background:{colour}14;padding:14px 18px;
        border-radius:6px;margin:8px 0 16px 0;"><div style="font-size:1.4rem;font-weight:700;color:{colour};">
        {title}</div><div style="margin-top:4px;">{body}</div></div>""", unsafe_allow_html=True)


def input_warnings(today, earlier):
    msgs = []
    for when, arr in (("today", today), ("8 days ago", earlier)):
        if arr is None:
            continue
        for a, v in zip(ATTRS, arr):
            if a in TRAIN_MAX and v > TRAIN_MAX[a]:
                msgs.append(f"{LABELS[a]} {when} ({v:,.0f}) is above anything the model saw in training "
                            f"({TRAIN_MAX[a]:,}). Treat the result with care.")
    if earlier is not None:
        for a in ("5", "187", "197"):
            i = ATTRS.index(a)
            if earlier[i] > today[i]:
                msgs.append(f"{LABELS[a]} is lower today than 8 days ago. These counters normally only go up; "
                            "check the two sets of readings are the right way round.")
    return msgs


def show_result(today, earlier):
    key = "nine_day" if earlier is not None else "today"
    art = MODELS[key]
    win = np.array([earlier, today] if earlier is not None else [today], dtype=float)
    score, contrib = score_windows(art, [win])
    score, c = float(score[0]), contrib.iloc[0]
    rate, capped = measured_rate(art, score)
    alarm = score > threshold(key, R)
    txt = chance_text(rate, capped)
    body = (f"In the validation quarter (Jan to Mar 2026), the chance of failing within {H} days for drives "
            f"scoring like this was <b>{txt}</b>. An average drive-day was {BASE_RATE}.")
    if alarm:
        verdict_box("Alarm: this drive is flagged", body, BAD)
    else:
        verdict_box("No alarm", body + " No alarm does not mean safe: many failures give no warning in these "
                    "readings (see 'How well does it work?').", GOOD)
    if key == "today":
        st.warning(f"Only today's readings were used, so this is the less accurate version: on the validation "
                   f"quarter it caught about {recall('today'):.0%} of failing drives, against "
                   f"{recall('nine_day'):.0%} for the version that also sees the readings from {GAP} days ago "
                   "(both at 1% false alarms). Add the earlier readings for a better answer.")
    else:
        st.caption(f"9-day model, using today's readings and the readings from {GAP} days ago "
                   f"(the gap it was trained on). Alarm level: {CHOICE}.")
    for m in input_warnings(today, earlier):
        st.warning(m)
    left, right = st.columns(2)
    left.subheader("Main reasons")
    left.write(reasons_from(c))
    right.subheader("What pushes the score up")
    right.bar_chart(c.rename(index=feature_label).nlargest(8))


PRESETS = {
    "Typical healthy drive": {
        "today": {"5": 0, "187": 0, "197": 0, "188": False, "4": 11, "12": 10, "190": 31},
        "earlier": {"5": 0, "187": 0, "197": 0, "188": False, "4": 11, "12": 10, "190": 31}},
    "Early warning signs": {
        "today": {"5": 40, "187": 10, "197": 0, "188": False, "4": 11, "12": 10, "190": 33},
        "earlier": {"5": 8, "187": 2, "197": 0, "188": False, "4": 11, "12": 10, "190": 31}},
    "Serious warning signs": {
        "today": {"5": 6900, "187": 160, "197": 96, "188": True, "4": 14, "12": 12, "190": 36},
        "earlier": {"5": 3000, "187": 60, "197": 12, "188": False, "4": 14, "12": 12, "190": 34}},
}
DEFAULTS = {"5": 0, "187": 0, "197": 0, "188": False, "4": 11, "12": 10, "190": 31}


def apply_preset(name):
    for pre, vals in (("t_", PRESETS[name]["today"]), ("e_", PRESETS[name]["earlier"])):
        for a, v in vals.items():
            st.session_state[pre + a] = v
    st.session_state["use_earlier"] = True


def reading_inputs(pre, disabled=False):
    st.number_input(LABELS["5"], min_value=0, step=1, key=pre + "5", disabled=disabled,
                    help="Bad spots the drive has already moved to spare areas. Healthy drives are usually at 0.")
    st.number_input(LABELS["187"], min_value=0, step=1, key=pre + "187", disabled=disabled,
                    help="Read errors the drive could not fix. Usually 0.")
    st.number_input(LABELS["197"], min_value=0, step=1, key=pre + "197", disabled=disabled,
                    help="Spots the drive is unsure about and waiting to re-test. Usually 0.")
    st.checkbox("Command timeouts reported (SMART 188 above zero)", key=pre + "188", disabled=disabled,
                help="Tick if the raw value of SMART 188 is above zero.")
    st.slider("Temperature in °C (SMART 190)", 10, 70, key=pre + "190", disabled=disabled,
              help="Most drives run between 25 and 45 °C.")
    with st.expander("Optional: start/stop and power cycle counts"):
        st.number_input(LABELS["4"], min_value=0, step=1, key=pre + "4", disabled=disabled,
                        help="Most drives show roughly 10 to 20. Leave as it is if you do not know.")
        st.number_input(LABELS["12"], min_value=0, step=1, key=pre + "12", disabled=disabled,
                        help="Most drives show roughly 10 to 20. Leave as it is if you do not know.")


def values_from_state(pre):
    s = st.session_state
    return [float(s[pre + "5"]), float(s[pre + "187"]), 1.0 if s[pre + "188"] else 0.0, float(s[pre + "197"]),
            float(s[pre + "4"]), float(s[pre + "12"]), float(s[pre + "190"])]


def one_drive_typed():
    for pre in ("t_", "e_"):
        for a, v in DEFAULTS.items():
            st.session_state.setdefault(pre + a, v)
    st.session_state.setdefault("use_earlier", False)

    st.markdown("**Start from an example** (illustrative values, then change them as you like):")
    cols = st.columns(3)
    for col, name in zip(cols, PRESETS):
        col.button(name, on_click=apply_preset, args=(name,), use_container_width=True)
    st.checkbox(f"Add the readings from {GAP} days ago (recommended: much more accurate)", key="use_earlier")
    use_earlier = st.session_state["use_earlier"]
    left, right = st.columns(2)
    with left:
        st.subheader("Today's readings")
        reading_inputs("t_")
    with right:
        st.subheader(f"Readings from {GAP} days ago")
        if not use_earlier:
            st.caption("Tick the box above to use these.")
        reading_inputs("e_", disabled=not use_earlier)
    st.divider()
    st.header("Result")
    show_result(values_from_state("t_"), values_from_state("e_") if use_earlier else None)


# ---------------------------------------------------------------- smartctl paste
LINE_RE = re.compile(r"^\s*(\d+)\s+(\S+)\s+0x[0-9a-fA-F]+\s+\d+\s+\d+\s+\d+\s+\S+\s+\S+\s+\S+\s+(.*\S)\s*$")
REQUIRED = {"5": "Reallocated_Sector_Ct", "187": "Reported_Uncorrect", "188": "Command_Timeout",
            "197": "Current_Pending_Sector", "4": "Start_Stop_Count", "12": "Power_Cycle_Count",
            "190": "Airflow_Temperature_Cel"}


def first_int(s):
    m = re.search(r"\d+", s or "")
    return int(m.group()) if m else None


def parse_smartctl(text):
    """Read the output of `smartctl -A` (plain text) or `smartctl -A -j` (JSON). Returns ({attr: number}, notes)."""
    rows, text = {}, text.strip()
    if text.startswith("{"):
        try:
            table = json.loads(text)["ata_smart_attributes"]["table"]
        except (ValueError, KeyError, TypeError):
            raise ValueError("That looks like JSON but has no SMART attribute table. Use: sudo smartctl -A -j /dev/sda")
        for a in table:
            raw = a.get("raw", {})
            rows[str(a["id"])] = (raw.get("value"), raw.get("string", "") or "")
    else:
        for line in text.splitlines():
            m = LINE_RE.match(line)
            if m:
                rows[m.group(1)] = (None, m.group(3))
    if not rows:
        raise ValueError("No SMART attribute table found. Paste the output of: sudo smartctl -A /dev/sda "
                         "(replace sda with your drive).")
    out, notes = {}, []
    for a in ATTRS:
        src = a
        if a == "190" and "190" not in rows and "194" in rows:
            src = "194"
            notes.append("This drive has no attribute 190, so attribute 194 (temperature) was used instead.")
        if src not in rows:
            continue
        value, string = rows[src]
        if a == "190":
            n = first_int(string)
            if n is None and value is not None:
                n = int(value) & 0xFF
        elif a == "188":
            n = int(value) if value is not None else sum(int(x) for x in re.findall(r"\d+", string))
        else:
            n = int(value) if value is not None else first_int(string)
        if n is not None:
            out[a] = float(n)
    missing = [f"{a} ({REQUIRED[a]})" for a in ATTRS if a not in out]
    if missing:
        raise ValueError("These attributes are missing: " + ", ".join(missing) +
                         ". This tool reads SATA drives; the output must contain these lines.")
    return out, notes


SAMPLE = """ID# ATTRIBUTE_NAME          FLAG     VALUE WORST THRESH TYPE      UPDATED  WHEN_FAILED RAW_VALUE
  1 Raw_Read_Error_Rate     0x000f   080   064   044    Pre-fail  Always       -       95433432
  3 Spin_Up_Time            0x0003   094   094   000    Pre-fail  Always       -       0
  4 Start_Stop_Count        0x0032   100   100   020    Old_age   Always       -       11
  5 Reallocated_Sector_Ct   0x0033   100   100   010    Pre-fail  Always       -       {r5}
  7 Seek_Error_Rate         0x000f   088   060   045    Pre-fail  Always       -       635428701
  9 Power_On_Hours          0x0032   070   070   000    Old_age   Always       -       26311
 10 Spin_Retry_Count        0x0013   100   100   097    Pre-fail  Always       -       0
 12 Power_Cycle_Count       0x0032   100   100   020    Old_age   Always       -       10
187 Reported_Uncorrect      0x0032   100   100   000    Old_age   Always       -       {r187}
188 Command_Timeout         0x0032   100   100   000    Old_age   Always       -       0
190 Airflow_Temperature_Cel 0x0022   069   058   040    Old_age   Always       -       33 (Min/Max 29/35)
194 Temperature_Celsius     0x0022   033   042   000    Old_age   Always       -       33 (0 20 0 0 0)
197 Current_Pending_Sector  0x0012   100   100   000    Old_age   Always       -       {r197}
"""


def load_sample():
    st.session_state["sc_today"] = SAMPLE.format(r5=112, r187=12, r197=8)
    st.session_state["sc_earlier"] = SAMPLE.format(r5=16, r187=2, r197=0)


def one_drive_smartctl():
    st.session_state.setdefault("sc_today", "")
    st.session_state.setdefault("sc_earlier", "")
    st.markdown("Run this on the computer the drive is in (replace `sda` with your drive; "
                "on Windows use smartmontools with the same command):")
    st.code("sudo smartctl -A /dev/sda", language="bash")
    st.caption("Paste the output below, then click outside the box (or press Ctrl+Enter). "
               f"For a more accurate answer, also paste the output from {GAP} days ago.")
    st.button("Load a sample", on_click=load_sample)
    left, right = st.columns(2)
    today_txt = left.text_area("Today's smartctl output", key="sc_today", height=280)
    earlier_txt = right.text_area(f"smartctl output from {GAP} days ago (optional, recommended)",
                                  key="sc_earlier", height=280)
    if not today_txt.strip():
        st.info("Paste today's smartctl output to begin.")
        return
    try:
        today_vals, notes = parse_smartctl(today_txt)
        earlier_vals = None
        if earlier_txt.strip():
            earlier_vals, notes2 = parse_smartctl(earlier_txt)
            notes += notes2
    except ValueError as e:
        st.error(str(e))
        return
    m = re.search(r"Device Model:\s*(.+)", today_txt)
    if m and DRIVE_MODEL not in m.group(1):
        st.warning(f"This drive is a {m.group(1).strip()}. The model was trained only on {DRIVE_MODEL} drives, "
                   "so do not trust the result.")
    for n in notes:
        st.info(n)
    seen = pd.DataFrame({"Reading": [LABELS[a] for a in ATTRS],
                         "Today": [today_vals[a] for a in ATTRS]})
    if earlier_vals is not None:
        seen[f"{GAP} days ago"] = [earlier_vals[a] for a in ATTRS]
    with st.expander("What was read from your text (check these)", expanded=True):
        st.dataframe(seen, hide_index=True)
    st.divider()
    st.header("Result")
    show_result([today_vals[a] for a in ATTRS], [earlier_vals[a] for a in ATTRS] if earlier_vals else None)


def screen_one_drive():
    method = st.radio("How do you want to enter the readings?",
                      ["Type or slide the values", "Paste smartctl output"], horizontal=True)
    st.divider()
    if method == "Type or slide the values":
        one_drive_typed()
    else:
        one_drive_smartctl()


# ---------------------------------------------------------------- many drives
def screen_many_drives():
    st.write(f"Upload one CSV with one row per drive per day: `serial_number`, `date` and the 7 SMART columns "
             f"(`{'`, `'.join(RAW_COLS)}`). Drives with {W} consecutive days of complete readings are scored with the "
             f"9-day model; drives with fewer days are scored from their latest day with the less accurate "
             f"today-only model. Backblaze daily files with many days also work (other drive models are dropped).")
    st.download_button("Download an example file", DEMO.to_csv(index=False), "example_drives.csv", "text/csv")
    up = st.file_uploader("Drag and drop your CSV file here", type="csv")
    if up is None:
        return
    wanted = {"serial_number", "date", "model"} | set(RAW_COLS)
    try:
        data = pd.read_csv(up, usecols=lambda c: c in wanted)
    except Exception as e:
        st.error(f"Could not read this file as a CSV: {e}")
        return
    missing = [c for c in ["serial_number", "date"] + RAW_COLS if c not in data.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        return
    if "model" in data.columns:
        n_before = data["serial_number"].nunique()
        data = data[data["model"] == DRIVE_MODEL]
        st.caption(f"Kept {data['serial_number'].nunique():,} of {n_before:,} drives that are {DRIVE_MODEL}.")
    res = score_table(data)
    if res["table"].empty:
        st.error("No drive could be scored.")
        if len(res["skipped"]):
            st.dataframe(res["skipped"], hide_index=True)
        return
    n_today = int((res["table"]["key"] == "today").sum())
    if n_today:
        st.info(f"{n_today:,} drive(s) did not have {W} consecutive days and were scored from their latest day "
                "only (less accurate).")
    top_n = st.select_slider("Watch-list length", options=[5, 10, 25, 50, 100], value=25, key="top_n_many")
    st.subheader("Watch-list")
    render_watchlist(res, top_n)
    st.subheader("Drive detail")
    render_detail(res, top_n)


# ---------------------------------------------------------------- shared pages
def render_limits():
    st.subheader("Validation: first quarter of 2026")
    ceiling = SUMMARY["catchable"] / SUMMARY["failed_drives"]
    st.write(f"{SUMMARY['failed_drives']} drives failed in the quarter. {SUMMARY['no_window']} had no usable "
             f"{W}-day window and {SUMMARY['quiet']} looked healthy on all four core counters, so at most "
             f"{SUMMARY['catchable']} ({ceiling:.0%}) can be caught from these readings.")
    rows = []
    for lv in ["1.00%", "0.10%", "0.01%"]:
        r = {"Healthy drive-days flagged by mistake": lv}
        for key, name in [("nine_day", "9-day model: failed drives caught"),
                          ("today", "Today-only model: failed drives caught")]:
            v = SUMMARY["drive_recall"][key][lv]
            r[name] = f"{v['recall']:.0%} (range {v['low']:.0%} to {v['high']:.0%})"
        rows.append(r)
    st.table(pd.DataFrame(rows))
    st.caption("The today-only model catches clearly fewer failing drives: about 18 points fewer at 1% false alarms "
               "(range 11 to 25).")
    st.subheader("Cost of each alarm level (9-day model)")
    st.dataframe(COST["nine_day"], hide_index=True)
    st.subheader("Cost of each alarm level (today-only model)")
    st.dataframe(COST["today"], hide_index=True)
    st.caption("R is how many false alarms one missed failure counts as. Alarm levels and the 'about 1 in N' figures "
               "were chosen and measured on the same quarter, so real results can be somewhat lower. The Q2 2026 "
               "quarter has been held out for a final test.")


def render_about():
    st.markdown(f"""
**What it does.** It estimates whether a Seagate {DRIVE_MODEL} drive will fail within the next {H} days, from its
SMART readings. A logistic regression trained on Backblaze data for 2023 to 2025 reads the drive's readings today and
{GAP} days earlier (9 days of history; only the first and last day matter).

**Chance of failing.** The figure shown ("about 1 in N") is the share of drives with a similar score that really
failed within {H} days in the first quarter of 2026. It is not the raw model score.

**Alarm.** The alarm level depends on how costly a missed failure is compared with a false alarm. The sidebar offers
Relaxed, Balanced and Cautious.

**Getting readings.** Run `sudo smartctl -A /dev/sda` (install with `sudo apt install smartmontools`), or upload a
CSV of daily readings.

**Limits.** One drive model only; many failures show no warning in these readings; results come from one validation
quarter; with only today's readings the answer is clearly less accurate. This is a decision aid, not a replacement
for monitoring.
""")
    st.caption("Group One Capstone Project, 2026, NIHUB FUT Minna.")


# ---------------------------------------------------------------- pages
def page_example():
    st.subheader("Try an example")
    st.write("Twenty real drive histories from the first quarter of 2026: 6 failed within 14 days, 14 did not. "
             "See how the tool ranks them.")
    st.warning(f"These 20 drives were picked to show how the tool reads, not at random. Real performance is on the "
               f"'How well does it work?' tab: about {recall('nine_day'):.0%} of failed drives caught at 1% false alarms.")
    res = score_table(DEMO)
    truth = KEY.set_index("serial_number")["fails_within_14_days"]
    top_n = st.select_slider("Watch-list length", options=[5, 10, 20], value=20, key="top_n_example")
    t1, t2, t3, t4 = st.tabs(["Watch-list", "Drive detail", "How well does it work?", "About"])
    with t1:
        render_watchlist(res, top_n, truth)
    with t2:
        render_detail(res, top_n, truth)
    with t3:
        render_limits()
    with t4:
        render_about()


def page_predict():
    t1, t2, t3 = st.tabs(["Get a prediction", "How well does it work?", "About"])
    with t1:
        st.info(f"This tool is trained only on Seagate {DRIVE_MODEL} (12 TB) drives. Readings from other drive "
                "models are outside what it has seen.")
        screen = st.radio("What do you want to check?", ["One drive", "Many drives (CSV file)"], horizontal=True)
        st.divider()
        if screen == "One drive":
            screen_one_drive()
        else:
            screen_many_drives()
    with t2:
        render_limits()
    with t3:
        render_about()


# ---------------------------------------------------------------- sidebar and layout
st.title("Hard drive failure prediction")
st.caption(f"Seagate {DRIVE_MODEL} drives. Estimates which drives are likely to fail within the next {H} days.")

page = st.sidebar.radio("Go to", ["Try an example", "Get prediction"])
st.sidebar.header("Settings")
CHOICE = st.sidebar.radio("How cautious should the alarm be?", list(CAUTION), index=1)
R = CAUTION[CHOICE]
row = COST["nine_day"][COST["nine_day"]["R"] == R].iloc[0]
st.sidebar.caption(f"A missed failure counts as {R} false alarms. On the validation quarter (about 18,600 drives a "
                   f"day, 9-day model) this setting raised about {row['false alarms per day']} false alarms a day "
                   f"and caught {row['failed drives caught']} failed drives.")
st.sidebar.caption("Group One Capstone Project, 2026, NIHUB FUT Minna.")

if page == "Try an example":
    page_example()
else:
    page_predict()