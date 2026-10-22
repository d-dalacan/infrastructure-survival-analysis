# Hard drive failure prediction

Predicts which hard drives are likely to fail within the next 14 days, from their daily SMART readings, and ranks them in a watch-list whose alarm level can be set by how costly a missed failure is. Built on Backblaze Drive Stats data for one drive model (Seagate ST12000NM0008).

**Group One Capstone Project**
** NIHUB FUT Minna, 2026**

Instructor: [SIR COLLINS]

**Live app:** [https://infrastructure-survival-analysis-project.streamlit.app/]

## What it does

For each drive, the model takes the last 14 days of readings, summarises them into 20 numbers and gives a risk score. Drives above the chosen alarm level are flagged, with the readings that pushed the score up. The app lets you upload your own drive readings or try demo drives, and move a cost slider ("a missed failure costs as much as N false alarms").

## Results (validation, first quarter of 2026)

129 drives failed in the quarter. At most 92 of them (71.3%) can be caught from these readings: 25 have no usable 14-day window and 12 look healthy on all four core counters.

| False alarms allowed (healthy drive-days flagged) | Failed drives caught (95% range) |
|---|---|
| 1% (10 per 1,000 drive-days) | 65% (56% to 73%) |
| 0.1% (1 per 1,000) | 36% (29% to 45%) |
| 0.01% (0.1 per 1,000) | 9% (4% to 15%) |

With a missed failure costing 100 false alarms, the cheapest alarm level raised about 18.5 false alarms per day and caught 48 of the 129 failed drives. The alarm levels were chosen and measured on the same quarter, so real results can be somewhat lower. The second quarter of 2026 has been held out and not opened.

## Method

- One example is one drive on one day. The input is that drive's last 14 consecutive days of readings; the label is 1 if the drive fails within the next 14 days.
- The split is by date, never random: training on 2023 to 2025, validation on Q1 2026, with a 14-day gap so no label looks past the end of its period.
- Healthy examples are sampled for training because there are too many; validation keeps every window, so its failure share is real. Failures are weighted up in training.
- Scores are reported per drive (is a failed drive caught at least once?), with ranges from resampling drives, not windows.
- Logistic regression was chosen over gradient boosting: the two perform alike, and logistic regression is simpler and explainable.

## Repository

| File | What it is |
|---|---|
| `01_data_loading_and_eda.ipynb` | Data loading and exploratory analysis |
| `02_modelling.ipynb` | Label design, features, models, drive-level scores, alarm levels, files for the app |
| `02b_cnn_experiments.ipynb` | Additional experiments with a small neural network; not used in the app |
| `survival_analysis.ipynb` | Earlier exploratory notebook |
| `app.py` | The Streamlit app |
| `model/` | The saved model, alarm levels, validation summary and demo drives |
| `requirements.txt` | Python packages (pinned versions) |

## Run the app

```
pip install -r requirements.txt
streamlit run app.py
```

## Rebuild the model

The raw data is not in this repository. Download the Backblaze Drive Stats files, keep the Seagate ST12000NM0008 rows, and save one parquet file per quarter in a `Dataset` folder. The cleaned training file `clean_train.parquet` is produced by notebook 01. In notebook 02, set `DATA_ROOT` to the folder holding `clean_train.parquet` and the `Dataset` folder, then run all cells.

## Limits

- One drive model only; the model was trained on 2023 to 2025.
- Many failures show no warning in these readings, so no model can catch them all.
- Only 129 failed drives in validation, so small differences between models cannot be told from chance.
- The alarm level depends on how costly a missed failure is for the user. A decision aid, not a replacement for monitoring.

## Data

Backblaze Drive Stats, https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data. Credit to Backblaze for publishing the data.