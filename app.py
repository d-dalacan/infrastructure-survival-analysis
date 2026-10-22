"""Drive failure watch-list: scores each drive's last 14 days with a logistic regression and explains the score."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
MODEL_DIR = HERE / "model"
RAW_COLS = ["smart_5_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw",
            "smart_4_raw", "smart_12_raw", "smart_190_raw"]
SMART_NAMES = {"5": "Reallocated sectors (SMART 5)", "187": "Uncorrectable errors (SMART 187)",
               "188": "Command timeouts (SMART 188)", "197": "Pending sectors (SMART 197)",
               "4": "Start/stop count (SMART 4)", "12": "Power cycles (SMART 12)",
               "190": "Airflow temperature (SMART 190)"}
SUFFIX = {"end_log": "level now", "change_log": "change over the window", "end_nz": "above zero now",
          "rose": "rose during the window", "changed": "changed during the window",
          "end": "value now", "change": "change over the window"}

st.set_page_config(page_title="Drive failure watch-list", layout="wide")


@st.cache_resource
def load_files():
    art = joblib.load(MODEL_DIR / "logistic_model.joblib")
    cost = pd.read_csv(MODEL_DIR / "cost_table.csv")
    summary = json.loads((MODEL_DIR / "validation_summary.json").read_text())
    demo = pd.read_csv(MODEL_DIR / "demo_drives.csv")
    key = pd.read_csv(MODEL_DIR / "demo_key.csv")
    return art, cost, summary, demo, key


def feature_label(feature):
    n, s = feature.split("_", 1)
    return f"{SMART_NAMES[n]}: {SUFFIX[s]}"


def window_features(win):
    """win: (days, 7) array of raw readings, oldest day first -> the 20 model features."""
    first, last = win[0], win[-1]
    J = {n: i for i, n in enumerate(["5", "187", "188", "197", "4", "12", "190"])}
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


def score_drives(df, art):
    W = art["window_days"]
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    keep, skipped, wins = [], [], {}
    for sn, g in df.sort_values("date").groupby("serial_number"):
        g = g.tail(W)
        if len(g) < W:
            skipped.append({"Drive": sn, "Why not scored": f"fewer than {W} days of readings"}); continue
        if not (g["date"].diff().dropna() == pd.Timedelta(days=1)).all():
            skipped.append({"Drive": sn, "Why not scored": f"missing days in the last {W} days"}); continue
        if g[RAW_COLS].isna().any().any():
            skipped.append({"Drive": sn, "Why not scored": "empty readings in the window"}); continue
        keep.append((sn, g["date"].iloc[-1])); wins[sn] = g
    if not keep:
        return pd.DataFrame(), pd.DataFrame(skipped), wins, None
    feats = pd.DataFrame([window_features(wins[sn][RAW_COLS].to_numpy(dtype=float)) for sn, _ in keep])[art["features"]]
    Z = art["scaler"].transform(feats)
    score = art["model"].predict_proba(Z)[:, 1]
    contrib = pd.DataFrame(Z * art["model"].coef_[0], columns=art["features"])
    out = pd.DataFrame({"Drive": [k[0] for k in keep], "Last day": [k[1].date() for k in keep], "Risk score": score})
    return out, pd.DataFrame(skipped), wins, contrib


art, cost, summary, demo, key = load_files()
levels = art["alarm_levels"]

st.title("Drive failure watch-list")
st.caption("Seagate ST12000NM0008 drives. Looks at each drive's last 14 days of SMART readings and ranks the drives "
           "most likely to fail within the next 14 days. A decision aid, not a replacement for monitoring.")

# ---------------- sidebar ----------------
st.sidebar.header("Settings")
source = st.sidebar.radio("Drives to score", ["Demo drives", "Upload a CSV"])
R = st.sidebar.select_slider("A missed failure costs as much as ... false alarms", options=sorted(levels),
                             value=100 if 100 in levels else sorted(levels)[0])
top_n = st.sidebar.select_slider("Watch-list length", options=[5, 10, 25, 50, 100], value=25)
row = cost[cost["R"] == R].iloc[0]
st.sidebar.caption(f"At this setting, across the ~18,600 drives of the validation quarter, the model raised about "
                   f"{row['false alarms per day']} false alarms per day and caught {row['failed drives caught']} failed drives.")

if source == "Demo drives":
    data, truth = demo, key.set_index("serial_number")["fails_within_14_days"]
    st.sidebar.warning("These 20 demo drives were picked to show how the tool reads. Real performance is on the "
                       "'How well does it work?' tab: about 65% of failed drives caught at 1% false alarms.")
else:
    up = st.sidebar.file_uploader("CSV: serial_number, date and the 7 SMART columns", type="csv")
    if up is None:
        st.info("Upload a CSV, or switch to the demo drives in the sidebar. The About tab shows the required columns.")
        st.stop()
    data, truth = pd.read_csv(up), None
    missing = [c for c in ["serial_number", "date"] + RAW_COLS if c not in data.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing)); st.stop()

scores, skipped, wins, contrib = score_drives(data, art)
if scores.empty:
    st.error("No drive had 14 consecutive days of complete readings."); st.stop()

thr = levels[R]["threshold"]
scores["Alarm"] = scores["Risk score"] > thr


def reasons(i):
    top = contrib.iloc[i].nlargest(3)
    top = top[top > 0.1]
    return "; ".join(feature_label(f) for f in top.index) if len(top) else "nothing unusual"


scores["Main reasons"] = [reasons(i) for i in range(len(scores))]
view = scores.sort_values("Risk score", ascending=False).head(top_n)
if truth is not None:
    view = view.assign(**{"Truth (demo only)": view["Drive"].map(lambda d: "fails within 14 days" if truth[d] else "does not fail")})

tab_list, tab_drive, tab_limits, tab_about = st.tabs(["Watch-list", "Drive detail", "How well does it work?", "About"])

with tab_list:
    c1, c2, c3 = st.columns(3)
    c1.metric("Drives scored", len(scores)); c2.metric("Alarms raised", int(scores["Alarm"].sum()))
    c3.metric("Drives not scored", len(skipped))
    st.dataframe(view, hide_index=True, column_config={
        "Risk score": st.column_config.ProgressColumn("Risk score", min_value=0.0, max_value=1.0, format="%.3f"),
        "Alarm": st.column_config.CheckboxColumn("Alarm")})
    st.caption("The risk score ranks drives; it is not the real chance of failure, because failures were weighted up in training. "
               "An alarm means the score is above the level chosen with the slider.")
    if len(skipped):
        with st.expander("Drives that could not be scored"):
            st.dataframe(skipped, hide_index=True)

with tab_drive:
    pick = st.selectbox("Drive", view["Drive"].tolist())
    g = wins[pick].set_index("date")
    i = scores.index[scores["Drive"] == pick][0]
    left, right = st.columns(2)
    left.subheader("Error counters, last 14 days")
    left.line_chart(g[["smart_5_raw", "smart_187_raw", "smart_197_raw"]])
    left.subheader("Temperature")
    left.line_chart(g[["smart_190_raw"]])
    right.subheader("What pushes the score up")
    right.bar_chart(contrib.iloc[i].rename(index=feature_label).nlargest(8))
    right.caption("Contribution of each reading compared with a typical drive-day. Higher means more risk.")

with tab_limits:
    st.subheader("Validation: first quarter of 2026")
    ceiling = summary["catchable"] / summary["failed_drives"]
    st.write(f"{summary['failed_drives']} drives failed in the quarter. {summary['no_window']} had no usable 14-day window and "
             f"{summary['quiet']} looked healthy on all four core counters, so at most {summary['catchable']} "
             f"({ceiling:.0%}) can be caught from these readings.")
    st.table(pd.DataFrame([{"Healthy drive-days flagged by mistake": k,
                            "Failed drives caught": f"{v['recall']:.0%} (range {v['low']:.0%} to {v['high']:.0%})"}
                           for k, v in summary["drive_recall"].items()]))
    st.subheader("Cost of each alarm level")
    st.dataframe(cost, hide_index=True)
    st.caption("Alarm levels were chosen and measured on the same quarter, so real results can be somewhat lower.")

with tab_about:
    st.markdown("""
**What it does.** For every drive it takes the last 14 days of readings, summarises them into 20 numbers and scores them
with a logistic regression trained on Backblaze data for 2023 to 2025.

**Your CSV needs one row per drive per day** with these columns: `serial_number`, `date`, `smart_5_raw`, `smart_187_raw`,
`smart_188_raw`, `smart_197_raw`, `smart_4_raw`, `smart_12_raw`, `smart_190_raw`. A drive is scored only if its last 14 days are
consecutive and complete.

**Limits.** One drive model only; many failures show no warning in these readings; the alarm level depends on how costly
a missed failure is for you, which is why it is a slider.
""")

    st.caption("Group One Capstone Project, 2026, NIHUB FUT Minna.")