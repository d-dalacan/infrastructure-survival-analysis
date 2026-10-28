"""Disk failure prediction for the Seagate ST12000NM0008.

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
MODEL_LABEL = {"nine_day": "9-day", "today": "one reading"}
GOOD, BAD, WARN = "#0F766E", "#B91C1C", "#B45309"

LEVEL_INFO = {"red": ("AT RISK", "🔴", BAD), "yellow": ("KEEP AN EYE ON IT", "🟡", WARN),
              "green": ("GOOD", "🟢", GOOD)}
LEVEL_SHORT = {"red": "🔴 At risk", "yellow": "🟡 Keep an eye on it", "green": "🟢 Good"}
LEVEL_RANK = {"red": 0, "yellow": 1, "green": 2}
CARD = {
    "red": ("This drive shows signs that often come before a failure.", "Back up your data now."),
    "yellow": ("A few readings are worse than normal, but nothing alarming yet.", "Check again in a few days."),
    "green": ("No signs of trouble in these readings.", "Nothing to do. Keep normal monitoring."),
}
LEVEL_DEF = {
    "red": "Readings worse than those of 99 out of 100 healthy drives.",
    "yellow": "Readings worse than those of 95 out of 100 healthy drives, but not among the worst 1%.",
    "green": "Readings no worse than those of 95 out of 100 healthy drives.",
}

st.set_page_config(page_title="Disk failure prediction", page_icon="💽", layout="wide")


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
for _k, _a in MODELS.items():
    if "levels" not in _a:
        st.error("The saved model files have no warning levels. Re-run the notebook cell that adds them.")
        st.stop()
W = MODELS["nine_day"]["window_days"]
H = MODELS["nine_day"]["horizon_days"]


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


def level_of(art, score):
    lv = art["levels"]
    if score > lv["red"]["threshold"]:
        return "red"
    if score > lv["yellow"]["threshold"]:
        return "yellow"
    return "green"


def n_text(rate):
    """1 / rate, rounded to something easy to read (24, 200, 5,500)."""
    n = 1 / rate
    n = int(round(n)) if n < 100 else int(round(n, 2 - len(str(int(n)))))
    return f"{n:,}"


PLAIN = {"5": "Reallocated sectors", "187": "Uncorrectable read errors", "197": "Pending sectors"}


def warning_signs(c, win, limit=4):
    """Plain-language list of the readings that push the model's score up (largest first)."""
    first, last = dict(zip(ATTRS, win[0])), dict(zip(ATTRS, win[-1]))
    order = []
    for f in c.sort_values(ascending=False).index:
        if c[f] <= 0.1:
            break
        a = f.split("_")[0]
        if a not in order:
            order.append(a)
    out = []
    for a in order[:limit]:
        v0, v1 = first[a], last[a]
        if a in PLAIN:
            if len(win) > 1 and v1 > v0:
                out.append(f"{PLAIN[a]} rose from {v0:,.0f} to {v1:,.0f}")
            else:
                out.append(f"{PLAIN[a]}: {v1:,.0f} (healthy drives are usually at 0)")
        elif a == "188":
            out.append("Command timeouts have been reported")
        elif a == "190":
            out.append(f"Temperature is higher than most healthy drives ({v1:.0f} °C)")
        elif a == "4":
            out.append(f"Start/stop count is high ({v1:,.0f})")
        elif a == "12":
            out.append(f"Power cycle count is high ({v1:,.0f})")
    return out


def verdict_box(title, body, colour):
    st.markdown(
        f"""<div style="border-left:8px solid {colour};background:{colour}14;padding:14px 18px;
        border-radius:6px;margin:8px 0 16px 0;"><div style="font-size:1.4rem;font-weight:700;color:{colour};">
        {title}</div><div style="margin-top:4px;">{body}</div></div>""", unsafe_allow_html=True)


def input_warnings(values):
    msgs = []
    for a, v in zip(ATTRS, values):
        if a in TRAIN_MAX and v > TRAIN_MAX[a]:
            msgs.append(f"{LABELS[a]} ({v:,.0f}) is above anything the model saw in training "
                        f"({TRAIN_MAX[a]:,}). Treat the result with care.")
    return msgs


# ---------------------------------------------------------------- the prediction card (one reading)
def predict_card(values):
    art = MODELS["today"]
    win = np.array([values], dtype=float)
    score, contrib = score_windows(art, [win])
    c = contrib.iloc[0]
    lvl = level_of(art, float(score[0]))
    title, icon, colour = LEVEL_INFO[lvl]
    headline, advice = CARD[lvl]
    signs = warning_signs(c, win)
    if not signs:
        signs = (["Nothing unusual in the readings"] if lvl == "green"
                 else ["No single reading stands out; the combination is what raised the score"])
    items = "".join(f"<li>{s}</li>" for s in signs)
    body = (f"{headline}<br><br><b>Out of every {n_text(art['levels'][lvl]['rate'])} drives with readings like "
            f"these, about 1 failed within the next {H} days.</b><br><br>"
            f"<b>Warning signs</b><ul style='margin:4px 0 10px 18px;'>{items}</ul><b>Advice:</b> {advice}")
    verdict_box(f"{icon} {title}", body, colour)
    st.caption("Based on drives in Jan to Mar 2026. This uses one set of readings, so it is less accurate than the "
               "9-day history used on the Many drives screen. Some failures give no warning in these readings.")
    for m in input_warnings(values):
        st.warning(m)
    with st.expander("More detail"):
        st.write(LEVEL_DEF[lvl])
        st.write("A normal drive-day was followed by a failure within 14 days about 1 time in 1,200.")
        st.write("Checked on Jan to Mar 2026 with one reading: 48% of the drives that failed were red at some point, "
                 "and 67% were red or yellow (counting a drive if it was flagged on any day of its last 14 days). "
                 "About 1 in 20 healthy drives shows yellow or red.")
        st.write("What pushes the score up:")
        st.bar_chart(c.rename(index=feature_label).nlargest(8), horizontal=True)


# ---------------------------------------------------------------- typed readings
PRESETS = {
    "Typical healthy drive": {"5": 0, "187": 0, "197": 0, "188": False, "4": 11, "12": 10, "190": 31},
    "Some warning signs": {"5": 40, "187": 10, "197": 0, "188": False, "4": 11, "12": 10, "190": 33},
    "Serious warning signs": {"5": 6900, "187": 160, "197": 96, "188": True, "4": 14, "12": 12, "190": 36},
}
DEFAULTS = {"5": 0, "187": 0, "197": 0, "188": False, "4": 11, "12": 10, "190": 31}


def apply_preset(name):
    for a, v in PRESETS[name].items():
        st.session_state["t_" + a] = v


def values_from_state():
    s = st.session_state
    return [float(s["t_5"]), float(s["t_187"]), 1.0 if s["t_188"] else 0.0, float(s["t_197"]),
            float(s["t_4"]), float(s["t_12"]), float(s["t_190"])]


def run_typed():
    st.session_state["typed_result"] = values_from_state()


def reading_inputs():
    st.number_input(LABELS["5"], min_value=0, step=1, key="t_5",
                    help="Bad spots the drive has already moved to spare areas. Healthy drives are usually at 0.")
    st.number_input(LABELS["187"], min_value=0, step=1, key="t_187",
                    help="Read errors the drive could not fix. Usually 0.")
    st.number_input(LABELS["197"], min_value=0, step=1, key="t_197",
                    help="Spots the drive is unsure about and waiting to re-test. Usually 0.")
    st.checkbox("Command timeouts reported (SMART 188 above zero)", key="t_188",
                help="Tick if the raw value of SMART 188 is above zero.")
    st.slider("Temperature in °C (SMART 190)", 10, 70, key="t_190",
              help="Most drives run between 25 and 45 °C.")
    with st.expander("Optional: start/stop and power cycle counts"):
        st.number_input(LABELS["4"], min_value=0, step=1, key="t_4",
                        help="Most drives show roughly 10 to 20. Leave as it is if you do not know.")
        st.number_input(LABELS["12"], min_value=0, step=1, key="t_12",
                        help="Most drives show roughly 10 to 20. Leave as it is if you do not know.")


def one_drive_typed():
    for a, v in DEFAULTS.items():
        st.session_state.setdefault("t_" + a, v)
    st.markdown("**Start from an example** (typical values; change them as you like):")
    cols = st.columns(3)
    for col, name in zip(cols, PRESETS):
        col.button(name, on_click=apply_preset, args=(name,), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Readings")
        reading_inputs()
        st.button("Predict", type="primary", on_click=run_typed)
    with right:
        st.subheader("Prediction")
        res = st.session_state.get("typed_result")
        if res is None:
            st.info("Enter the readings on the left and click Predict.")
        else:
            predict_card(res)
            if res != values_from_state():
                st.caption("The readings have changed since this prediction. Click Predict again.")


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


def load_sample(r5, r187, r197):
    st.session_state["sc_text"] = SAMPLE.format(r5=r5, r187=r187, r197=r197)
    st.session_state.pop("sc_result", None)


def run_smartctl():
    text = st.session_state.get("sc_text", "")
    if not text.strip():
        st.session_state["sc_result"] = ("empty", None, [], None)
        return
    try:
        vals, notes = parse_smartctl(text)
    except ValueError as e:
        st.session_state["sc_result"] = ("err", str(e), [], None)
        return
    m = re.search(r"Device Model:\s*(.+)", text)
    warn = None
    if m and DRIVE_MODEL not in m.group(1):
        warn = (f"This drive is a {m.group(1).strip()}. The model was trained only on {DRIVE_MODEL} drives, "
                "so do not trust the result.")
    st.session_state["sc_result"] = ("ok", [vals[a] for a in ATTRS], notes, warn)


def one_drive_smartctl():
    st.session_state.setdefault("sc_text", "")
    st.markdown("Run this on the computer the drive is in (replace `sda` with your drive), then paste the output:")
    st.code("sudo smartctl -A /dev/sda", language="bash")
    b1, b2, _ = st.columns([1, 1.3, 2])
    b1.button("Load a healthy sample", on_click=load_sample, args=(0, 0, 0))
    b2.button("Load a sample with warning signs", on_click=load_sample, args=(112, 12, 8))
    left, right = st.columns(2)
    with left:
        st.text_area("Paste the smartctl output here", key="sc_text", height=320)
        st.button("Predict", type="primary", on_click=run_smartctl, key="sc_predict")
    with right:
        st.subheader("Prediction")
        res = st.session_state.get("sc_result")
        if res is None:
            st.info("Paste the smartctl output on the left, or load a sample, then click Predict.")
        elif res[0] == "empty":
            st.info("The box is empty. Paste the smartctl output first.")
        elif res[0] == "err":
            st.error(res[1])
        else:
            _, values, notes, warn = res
            if warn:
                st.warning(warn)
            for n in notes:
                st.info(n)
            predict_card(values)
            with st.expander("What was read from your text (check these)"):
                st.dataframe(pd.DataFrame({"Reading": [LABELS[a] for a in ATTRS], "Value": values}),
                             hide_index=True)


def screen_one_drive():
    method = st.radio("How do you want to enter the readings?",
                      ["Type or slide the values", "Paste smartctl output"], horizontal=True)
    st.divider()
    if method == "Type or slide the values":
        one_drive_typed()
    else:
        one_drive_smartctl()


# ---------------------------------------------------------------- lists (example and many drives)
def alarm_dial():
    choice = st.radio("Alarm setting for the Alarm column", list(CAUTION), index=1, horizontal=True, key="dial")
    R = CAUTION[choice]
    row = COST["nine_day"][COST["nine_day"]["R"] == R].iloc[0]
    st.caption(f"A missed failure counts as {R} false alarms. On the validation quarter (about 18,600 drives a day, "
               f"9-day model) this raised about {row['false alarms per day']} false alarms a day and caught "
               f"{row['failed drives caught']} failed drives.")
    return R


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
        art = MODELS[key]
        score, c = score_windows(art, [it[2] for it in items])
        for i, (sn, day, win) in enumerate(items):
            lvl = level_of(art, float(score[i]))
            signs = warning_signs(c.iloc[i], win, limit=3)
            contrib[sn] = c.iloc[i]
            rows.append({"Drive": sn, "Last day": day.date(), "key": key, "Model": MODEL_LABEL[key],
                         "score": float(score[i]), "level": lvl, "rate": art["levels"][lvl]["rate"],
                         "Main warning signs": "; ".join(signs) if signs else "nothing unusual"})
    return {"table": pd.DataFrame(rows), "skipped": pd.DataFrame(skipped), "wins": wins, "contrib": contrib}


def view_frame(res, R, truth=None):
    t = res["table"].copy()
    t["Level"] = t["level"].map(LEVEL_SHORT)
    t["Failed within 14 days (similar drives)"] = ["1 in " + n_text(r) for r in t["rate"]]
    t["Alarm"] = [s > MODELS[k]["alarm_levels"][R]["threshold"] for s, k in zip(t["score"], t["key"])]
    t["rank"] = t["level"].map(LEVEL_RANK)
    t = t.sort_values(["rank", "rate", "score"], ascending=[True, False, False])
    if truth is not None:
        t["Truth (example only)"] = t["Drive"].map(lambda d: "fails within 14 days" if truth[d] else "does not fail")
    return t


def render_watchlist(res, R, top_n, truth=None, download=False):
    t = view_frame(res, R, truth)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 At risk", int((t["level"] == "red").sum()))
    c2.metric("🟡 Keep an eye on it", int((t["level"] == "yellow").sum()))
    c3.metric("🟢 Good", int((t["level"] == "green").sum()))
    c4.metric("Not scored", len(res["skipped"]))
    cols = ["Drive", "Last day", "Level", "Failed within 14 days (similar drives)", "Main warning signs",
            "Model", "Alarm"] + (["Truth (example only)"] if truth is not None else [])
    st.dataframe(t[cols].head(top_n), hide_index=True,
                 column_config={"Alarm": st.column_config.CheckboxColumn("Alarm")})
    st.caption("'Failed within 14 days' is how often drives at this level failed within 14 days in the validation "
               "quarter (Jan to Mar 2026). The 'Alarm' column follows the alarm setting above.")
    if download:
        st.download_button("Download all results as CSV", t[cols].to_csv(index=False), "drive_predictions.csv",
                           "text/csv")
    if len(res["skipped"]):
        with st.expander("Drives that could not be scored"):
            st.dataframe(res["skipped"], hide_index=True)


def render_detail(res, R, top_n, truth=None):
    t = view_frame(res, R, truth)
    pick = st.selectbox("Drive", t["Drive"].head(top_n).tolist())
    row = t[t["Drive"] == pick].iloc[0]
    left, right = st.columns(2)
    left.metric("Level", row["Level"])
    left.write(f"Drives at this level failed within 14 days about {row['Failed within 14 days (similar drives)']} "
               f"of the time (model: {row['Model']}).")
    left.write("Main warning signs: " + row["Main warning signs"])
    if pick in res["wins"]:
        g = res["wins"][pick].set_index("date")
        left.subheader(f"Error counters, last {W} days")
        left.line_chart(g[["smart_5_raw", "smart_187_raw", "smart_197_raw"]])
        left.subheader("Temperature")
        left.line_chart(g[["smart_190_raw"]])
    else:
        left.info("Only one day of readings, so there is no chart.")
    right.subheader("What pushes the score up")
    right.bar_chart(res["contrib"][pick].rename(index=feature_label).nlargest(8), horizontal=True)
    right.caption("Contribution of each reading compared with a typical drive-day. Higher means more risk.")


def read_upload(up):
    """Read a big CSV in pieces, keeping only the needed columns and the right drive model."""
    wanted = {"serial_number", "date", "model"} | set(RAW_COLS)
    parts, rows_read = [], 0
    for chunk in pd.read_csv(up, usecols=lambda c: c in wanted, chunksize=200_000):
        rows_read += len(chunk)
        if "model" in chunk.columns:
            chunk = chunk[chunk["model"] == DRIVE_MODEL].drop(columns=["model"])
        parts.append(chunk)
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()), rows_read


def screen_many_drives():
    st.write(f"Upload one CSV with one row per drive per day: `serial_number`, `date` and the 7 SMART columns "
             f"(`{'`, `'.join(RAW_COLS)}`). Drives with {W} consecutive days of complete readings are scored with the "
             "9-day history, which is more accurate. Drives with fewer days are scored from their latest day only. "
             "Backblaze daily files also work (other drive models are dropped). For several big daily files, combine "
             "them first with `prepare_upload.py`.")
    st.download_button("Download an example file", DEMO.to_csv(index=False), "example_drives.csv", "text/csv")
    up = st.file_uploader("Drag and drop your CSV file here", type="csv")
    if up is None:
        return
    try:
        with st.spinner("Reading the file..."):
            data, rows_read = read_upload(up)
    except Exception as e:
        st.error(f"Could not read this file as a CSV: {e}")
        return
    missing = [c for c in ["serial_number", "date"] + RAW_COLS if c not in data.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        return
    st.caption(f"Read {rows_read:,} rows; kept {len(data):,} rows from {data['serial_number'].nunique():,} "
               f"{DRIVE_MODEL} drives.")
    with st.spinner("Scoring..."):
        res = score_table(data)
    if res["table"].empty:
        st.error("No drive could be scored.")
        if len(res["skipped"]):
            st.dataframe(res["skipped"], hide_index=True)
        return
    n_one = int((res["table"]["key"] == "today").sum())
    if n_one:
        st.info(f"{n_one:,} drive(s) did not have {W} consecutive days and were scored from their latest day only "
                "(less accurate).")
    R = alarm_dial()
    top_n = st.select_slider("Watch-list length", options=[5, 10, 25, 50, 100], value=25, key="top_n_many")
    st.subheader("Watch-list")
    render_watchlist(res, R, top_n, download=True)
    st.subheader("Drive detail")
    render_detail(res, R, top_n)


# ---------------------------------------------------------------- shared tabs
FEATURE_HELP = [
    ("Reallocated sectors (SMART 5)", "Bad spots on the disk that the drive has already moved to spare areas. A rising count means the surface is wearing out."),
    ("Uncorrectable errors (SMART 187)", "Read errors the drive could not fix. Normally 0; a rising count is a warning."),
    ("Pending sectors (SMART 197)", "Spots the drive is unsure about and waiting to re-test. Normally 0; a rising count is a warning."),
    ("Command timeouts (SMART 188)", "Times the drive failed to answer in time. Only 'zero or not zero' (and, with a history, 'did it rise') is used."),
    ("Start/stop and power cycles (SMART 4, 12)", "How often the drive was started and stopped."),
    ("Temperature (SMART 190)", "Drive temperature in °C."),
]


def render_features():
    st.subheader("What the model looks at")
    st.write("It reads 7 SMART counters. With one reading it builds 10 numbers from them: how big each counter is and "
             "whether it is above zero. With a 9-day history (the Many drives screen) it also uses how much each "
             "counter changed or rose between the first and the last day, which makes it more accurate: a counter "
             "that has just jumped matters more than one that has sat still for years.")
    st.table(pd.DataFrame(FEATURE_HELP, columns=["Reading", "What it means"]))
    art = MODELS["nine_day"]
    coef = pd.Series(art["model"].coef_[0], index=art["features"])
    st.markdown("**How much each of the 20 numbers of the 9-day model pushes the score**")
    st.bar_chart(coef.rename(index=feature_label).sort_values(), horizontal=True)
    st.caption("Bars to the right raise the score; bars to the left lower it. The numbers were put on a common scale "
               "before fitting, so the lengths can be compared. Several numbers overlap, so read the chart as a "
               "rough guide, not as the exact effect of one reading.")


def render_limits():
    st.subheader("Validation: first quarter of 2026")
    ceiling = SUMMARY["catchable"] / SUMMARY["failed_drives"]
    st.write(f"{SUMMARY['failed_drives']} drives failed in the quarter. {SUMMARY['no_window']} had no usable "
             f"{W}-day window and {SUMMARY['quiet']} looked healthy on all four core counters, so at most "
             f"{SUMMARY['catchable']} ({ceiling:.0%}) can be caught from these readings.")
    st.markdown("**The warning levels**")
    rows = []
    for key, name in [("today", "One reading"), ("nine_day", "9-day history")]:
        lv = MODELS[key]["levels"]
        rows.append({"Model": name,
                     "🔴 At risk (worst 1% of healthy drive-days)": "1 in " + n_text(lv["red"]["rate"]),
                     "🟡 Keep an eye on it (next 4%)": "1 in " + n_text(lv["yellow"]["rate"]),
                     "🟢 Good (the other 95%)": "1 in " + n_text(lv["green"]["rate"]),
                     "Failing drives red at some point": "62 of 129 (48%)" if key == "today" else "85 of 129 (66%)",
                     "Failing drives red or yellow": "87 of 129 (67%)" if key == "today" else "91 of 129 (71%)"})
    st.table(pd.DataFrame(rows))
    st.caption("'1 in N' is how often drive-days at that level failed within 14 days; an average drive-day was about "
               "1 in 1,200. About 1 in 20 healthy drives shows yellow or red. A failing drive is counted if it was "
               "flagged on any day of its last 14 days. These levels were set and measured on the same quarter, so "
               "real results may be somewhat lower.")
    st.markdown("**Drive recall at fixed false-alarm levels**")
    rows = []
    for lv in ["1.00%", "0.10%", "0.01%"]:
        r = {"Healthy drive-days flagged by mistake": lv}
        for key, name in [("nine_day", "9-day history: failed drives caught"),
                          ("today", "One reading: failed drives caught")]:
            v = SUMMARY["drive_recall"][key][lv]
            r[name] = f"{v['recall']:.0%} (range {v['low']:.0%} to {v['high']:.0%})"
        rows.append(r)
    st.table(pd.DataFrame(rows))
    st.subheader("Cost of each alarm level (9-day history)")
    st.dataframe(COST["nine_day"], hide_index=True)
    st.subheader("Cost of each alarm level (one reading)")
    st.dataframe(COST["today"], hide_index=True)
    st.caption("R is how many false alarms one missed failure counts as. Q2 2026 (April to June) was kept for a final test. It was used only to draw 12 random drive-days for the live demonstration, and the model was not changed afterwards.")


def render_about():
    st.markdown(f"""
**What it does.** It estimates whether a Seagate {DRIVE_MODEL} drive will fail within the next {H} days, from its
SMART readings. A logistic regression trained on Backblaze data for 2023 to 2025 reads the readings.

**One drive.** Type in one set of readings, or paste the output of `sudo smartctl -A /dev/sda` (install with
`sudo apt install smartmontools`), and click Predict. You get a level (at risk, keep an eye on it, good), how often
drives at that level failed within {H} days, and the warning signs.

**Many drives.** Upload a CSV of daily readings. Drives with {W} days of history are scored more accurately than
drives with one reading.

**Limits.** One drive model only; many failures show no warning in these readings; the levels come from one
validation quarter; a single reading is less accurate than a {W}-day history. This is a decision aid, not a
replacement for monitoring.
""")
    st.caption("Group One Capstone Project, 2026, NIHUB FUT Minna.")


# ---------------------------------------------------------------- pages
def page_example():
    st.subheader("Try an example")
    st.write("Twenty real drive histories from the first quarter of 2026: 6 failed within 14 days, 14 did not. "
             "See how the tool ranks them.")
    st.warning(f"These 20 drives were picked to show how the tool reads, not at random. Real performance is on the "
               f"'How well does it work?' tab: about {recall('nine_day'):.0%} of failed drives caught at 1% false "
               "alarms.")
    res = score_table(DEMO)
    truth = KEY.set_index("serial_number")["fails_within_14_days"]
    R = alarm_dial()
    top_n = st.select_slider("Watch-list length", options=[5, 10, 20], value=20, key="top_n_example")
    t1, t2, t3, t4, t5 = st.tabs(["Watch-list", "Drive detail", "What the model looks at",
                                  "How well does it work?", "About"])
    with t1:
        render_watchlist(res, R, top_n, truth)
    with t2:
        render_detail(res, R, top_n, truth)
    with t3:
        render_features()
    with t4:
        render_limits()
    with t5:
        render_about()


def page_predict():
    t1, t2, t3, t4 = st.tabs(["Get a prediction", "What the model looks at", "How well does it work?", "About"])
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
        render_features()
    with t3:
        render_limits()
    with t4:
        render_about()


# ---------------------------------------------------------------- layout
st.title("Disk failure prediction")
st.caption(f"Seagate {DRIVE_MODEL} drives. Estimates which drives are likely to fail within the next {H} days.")
page = st.sidebar.radio("Go to", ["Try an example", "Get prediction"])
st.sidebar.caption("Group One Capstone Project, 2026, NIHUB FUT Minna.")

if page == "Try an example":
    page_example()
else:
    page_predict()