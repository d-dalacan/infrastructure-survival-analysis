# Disk failure prediction

**Group One Capstone Project, 2026, NIHUB FUT Minna**

Estimates whether a Seagate ST12000NM0008 hard drive will fail within the next 14 days, from its daily SMART readings (Backblaze data), with an interactive Streamlit app.

**Live app:** (paste your Streamlit link here)

## What the app does

- **Try an example:** 20 real drive histories from Q1 2026, ranked by risk.
- **Get prediction, One drive:** type the 7 readings, or paste the output of `smartctl -A`, and click Predict. The result is a colour level (at risk, keep an eye on it, good), how often drives at that level failed within 14 days, and the warning signs. This uses one reading.
- **Get prediction, Many drives:** upload a CSV of daily readings. Drives with 9 consecutive days are scored with the more accurate 9-day history; drives with fewer days from their latest day. The result is a ranked list that can be downloaded.

## How the levels work

Levels are set on the validation quarter (Q1 2026) from how a drive's score compares with healthy drive-days: red is the worst 1%, yellow the next 4%, green the other 95%.

| Level | One reading: drive-days that failed within 14 days | 9-day history |
|---|---|---|
| At risk | 1 in 24 | 1 in 19 |
| Keep an eye on it | 1 in 200 | 1 in 330 |
| Good | 1 in 5,500 | 1 in 6,500 |

An average drive-day was about 1 in 1,200. Of the 129 drives that failed in Q1 2026, the one-reading model had 62 (48%) in red at some point and 87 (67%) in red or yellow; the 9-day model 85 (66%) and 91 (71%). About 1 in 20 healthy drives shows yellow or red.

## Results (validation: Q1 2026, 129 failed drives)

| False alarms | 9-day history | One reading |
|---|---|---|
| 1% of healthy drive-days | 65.9% (57-74%) | 48.1% (39-56%) |
| 0.1% | 38.0% (29-47%) | 24.8% (17-32%) |
| 0.01% | 9.3% (4-14%) | 10.1% (5-15%) |

- At most 92 of the 129 failed drives (71.3%) can be caught from these readings: 25 have no usable window and 12 look healthy on all four core counters.
- Gradient boosting performed about the same as logistic regression, so logistic regression is used (simpler, explainable).
- Ranges come from resampling drives 1,000 times. A drive counts as caught if any of its failing windows is flagged.

## Method

- One example is one drive-day. The input is the last 9 days of readings (only the first and last day are used); the label is 1 if the drive fails 1 to 14 days later.
- Split by date with a 14-day gap: training to 2025-12-17, validation Q1 2026.
- 20 summary features for the 9-day model, 10 for the one-reading model; logistic regression with failures weighted up.
- Q2 2026 was kept for a final test. It was used only to draw 12 random drive-days for a live demonstration; the model was not changed afterwards, and the full Q2 evaluation has not been run.

## Repository

| File | What it is |
|---|---|
| `app.py` | The Streamlit app |
| `02_modelling.ipynb` | Label design, features, models, levels, saving the model |
| `02b_cnn_experiments.ipynb` | Extra experiments with other model types (no improvement shown) |
| `parquet_maker.py` | Builds the training data from Backblaze daily files |
| `prepare_upload.py` | Combines Backblaze daily CSVs into one small CSV for the app: `python prepare_upload.py FOLDER OUT.csv` |
| `model/` | Saved models, levels, cost tables, validation summary, example drives |
| `.streamlit/config.toml` | App theme and upload limit |
| `requirements.txt` | Pinned library versions |

## Run the app

    pip install -r requirements.txt
    streamlit run app.py

## Limits

- One drive model only.
- Many failures show no warning in these readings.
- Levels and the "1 in N" figures were set and measured on the same quarter, so real results may be somewhat lower.
- Whether failure patterns drift over time was not checked.
- A single reading is less accurate than a 9-day history.

## Data

Backblaze Drive Stats, https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data