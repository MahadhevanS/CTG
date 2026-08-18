"""
Parses clinical outcome/descriptor fields from CTU-CHB WFDB header (.hea)
comment blocks into a single clinical_metadata.csv.

The CTU-CHB headers embed clinical fields as '#Name    Value' comment lines
(see PhysioNet database documentation). Missing values appear as the literal
string 'NaN', or as -1 for II.stage ("unknown duration").

Usage:
    python scripts/extract_metadata.py
"""
import os
import re
import glob

import pandas as pd

RAW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw", "ctu-chb-intrapartum-cardiotocography-database-1.0.0",
)
OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "raw", "clinical_metadata.csv",
)

# Maps the header comment label -> output column name
FIELD_MAP = {
    "pH": "ph",
    "BDecf": "bdecf",
    "pCO2": "pco2",
    "BE": "be",
    "Apgar1": "apgar1",
    "Apgar5": "apgar5",
    "Gest. weeks": "gest_weeks",
    "Weight(g)": "weight_g",
    "Sex": "sex",
    "Age": "maternal_age",
    "Gravidity": "gravidity",
    "Parity": "parity",
    "Diabetes": "diabetes",
    "Hypertension": "hypertension",
    "Preeclampsia": "preeclampsia",
    "Liq. praecox": "liq_praecox",
    "Pyrexia": "pyrexia",
    "Meconium": "meconium",
    "Presentation": "presentation",
    "Induced": "induced",
    "I.stage": "stage1_min",
    "NoProgress": "no_progress",
    "CK/KP": "ck_kp",
    "II.stage": "stage2_min",
    "Deliv. type": "delivery_type",
    "Rec. type": "rec_type",
    "Pos. II.st.": "pos_stage2_sample",
    "Sig2Birth": "sig2birth_min",
}

# '#Name    Value' with arbitrary internal whitespace before the value
LINE_RE = re.compile(r"^#([A-Za-z0-9\./]+(?:[ \t]+[A-Za-z0-9\./]+)*?)\s{2,}(\S+)\s*$")


def parse_header(path: str) -> dict:
    record_id = os.path.splitext(os.path.basename(path))[0]
    row = {"record_id": record_id}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = LINE_RE.match(line)
            if not m:
                continue
            label, value = m.group(1).strip(), m.group(2).strip()
            if label not in FIELD_MAP:
                continue
            col = FIELD_MAP[label]
            if value.upper() == "NAN":
                row[col] = float("nan")
            else:
                try:
                    row[col] = float(value) if "." in value else int(value)
                except ValueError:
                    row[col] = value
    return row


def main():
    hea_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.hea")))
    print(f"Found {len(hea_files)} .hea files in {RAW_DIR}")
    rows = [parse_header(p) for p in hea_files]
    df = pd.DataFrame(rows)

    # II.stage: -1 encodes "unknown duration" in the source docs, not a real 0.
    if "stage2_min" in df.columns:
        df.loc[df["stage2_min"] < 0, "stage2_min"] = float("nan")

    df = df.sort_values("record_id").reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows -> {OUT_PATH}")
    print(f"  Missing pH: {df['ph'].isna().sum()}")
    print(f"  Missing BDecf: {df['bdecf'].isna().sum()}")
    print(f"  Missing both pH and BDecf: {(df['ph'].isna() & df['bdecf'].isna()).sum()}")
    print(f"  pH <= 7.05 (primary positive label): {(df['ph'] <= 7.05).sum()}")
    print(f"  pH <= 7.15 (old repo's threshold, for reference): {(df['ph'] <= 7.15).sum()}")
    print(f"  BDecf > 12 (secondary label): {(df['bdecf'] > 12).sum()}")
    print(f"  Unknown stage-II duration: {df['stage2_min'].isna().sum()}")


if __name__ == "__main__":
    main()
