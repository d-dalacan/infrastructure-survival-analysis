import glob
import os
import zipfile
import pandas as pd

MODEL = "ST12000NM0008"

KEEP = {
    "date", "serial_number", "model", "failure",
    "vault_id", "pod_id", "pod_slot_num",
    "smart_1_raw", "smart_4_raw", "smart_5_raw", "smart_7_raw",
    "smart_9_raw", "smart_12_raw", "smart_187_raw", "smart_188_raw",
    "smart_190_raw", "smart_192_raw", "smart_193_raw", "smart_194_raw",
    "smart_195_raw", "smart_197_raw", "smart_198_raw", "smart_199_raw",
    "smart_240_raw", "smart_241_raw", "smart_242_raw",
    "smart_3_raw", "smart_10_raw", "smart_183_raw", "smart_184_raw",
    "smart_189_raw", "smart_191_raw", "smart_196_raw",
    "smart_5_normalized", "smart_187_normalized", "smart_188_normalized",
    "smart_197_normalized", "smart_198_normalized",
}

DATA = os.path.expanduser("~/FailurePrediction/Dataset")
OUT = os.path.expanduser("~/FailurePrediction/parquet")
os.makedirs(OUT, exist_ok=True)

def read_day(f):
    df = pd.read_csv(f, usecols=lambda c: c in KEEP)
    df = df[df["model"] == MODEL]
    return df.drop(columns=["model"])

def save(parts, name):
    out = f"{OUT}/seagate_{name}.parquet"
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(out)
    print(name, "->", df.shape, "failures:", int(df["failure"].sum()))

# Quarters that are unzipped folders
for folder in sorted(glob.glob(DATA + "/data_*/")):
    name = os.path.basename(folder.rstrip("/"))
    files = sorted(glob.glob(folder + "**/*.csv", recursive=True))
    save([read_day(f) for f in files], name)

# Quarters that are still zip files
for zpath in sorted(glob.glob(DATA + "/*.zip")):
    name = os.path.basename(zpath).replace(".zip", "")
    parts = []
    with zipfile.ZipFile(zpath) as z:
        for n in sorted(z.namelist()):
            if n.endswith(".csv") and "__MACOSX" not in n:
                with z.open(n) as f:
                    parts.append(read_day(f))
    save(parts, name)