from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SOURCE_CSV = BASE_DIR / "stock_risk_dataset.csv"
OUTPUT_CSV = BASE_DIR / "stock_risk_dataset_synthetic.csv"

TARGET_ROWS = 3000
RANDOM_SEED = 42
RISK_LEVELS = ["Low", "Medium", "High"]

NUMERIC_COLUMNS = [
    "stock_quantity",
    "weekly_sales",
    "supplier_delay_days",
    "min_stock_level",
    "price",
    "last_sale_days",
    "seasonality",
]

COLUMN_ORDER = [
    "product_id",
    "product_name",
    "category",
    "stock_quantity",
    "weekly_sales",
    "supplier_delay_days",
    "min_stock_level",
    "price",
    "last_sale_days",
    "seasonality",
    "risk_level",
]


def clip_round(values, lower, upper):
    return np.rint(np.clip(values, lower, upper)).astype(int)


def risk_score(row):
    stock_gap = (row["min_stock_level"] - row["stock_quantity"]) / max(row["min_stock_level"], 1)
    sales_pressure = row["weekly_sales"] / 70
    delay_pressure = row["supplier_delay_days"] / 15
    stale_stock_relief = row["last_sale_days"] / 120
    seasonal_pressure = 0.12 if row["seasonality"] == 1 else 0

    return (
        0.42 * stock_gap
        + 0.28 * sales_pressure
        + 0.22 * delay_pressure
        + seasonal_pressure
        - 0.12 * stale_stock_relief
    )


def matches_target_risk(row, target_risk):
    score = risk_score(row)

    if target_risk == "High":
        return (
            score >= 0.34
            and row["stock_quantity"] <= row["min_stock_level"] * 1.1
            and row["weekly_sales"] >= 12
        )

    if target_risk == "Low":
        return (
            score <= 0.12
            and row["stock_quantity"] >= row["min_stock_level"] * 1.25
            and row["supplier_delay_days"] <= 7
        )

    return 0.10 < score < 0.38


def adjust_for_target_risk(rows, target_risk, rng):
    rows = rows.copy()

    if target_risk == "High":
        rows["stock_quantity"] = np.minimum(
            rows["stock_quantity"],
            rng.uniform(0.05, 1.05, len(rows)) * rows["min_stock_level"],
        )
        rows["weekly_sales"] = np.maximum(rows["weekly_sales"], rng.normal(38, 12, len(rows)))
        rows["supplier_delay_days"] = np.maximum(rows["supplier_delay_days"], rng.normal(10, 3, len(rows)))
        rows["last_sale_days"] = np.minimum(rows["last_sale_days"], rng.normal(8, 5, len(rows)))
        rows["seasonality"] = rng.binomial(1, 0.62, len(rows))

    elif target_risk == "Low":
        rows["stock_quantity"] = np.maximum(
            rows["stock_quantity"],
            rng.uniform(1.4, 9.0, len(rows)) * rows["min_stock_level"],
        )
        rows["weekly_sales"] = np.minimum(rows["weekly_sales"], rng.normal(5, 4, len(rows)))
        rows["supplier_delay_days"] = np.minimum(rows["supplier_delay_days"], rng.normal(3, 2, len(rows)))
        rows["seasonality"] = rng.binomial(1, 0.22, len(rows))

    else:
        rows["stock_quantity"] = np.clip(
            rows["stock_quantity"],
            rng.uniform(0.45, 3.2, len(rows)) * rows["min_stock_level"],
            rng.uniform(1.0, 4.2, len(rows)) * rows["min_stock_level"],
        )
        rows["weekly_sales"] = np.clip(rows["weekly_sales"], 4, 38)
        rows["supplier_delay_days"] = np.clip(rows["supplier_delay_days"], 2, 10)
        rows["seasonality"] = rng.binomial(1, 0.25, len(rows))

    return rows


def sample_class_rows(source, target_risk, count, rng):
    class_source = source[source["risk_level"] == target_risk].reset_index(drop=True)
    sampled = class_source.sample(n=count * 3, replace=True, random_state=int(rng.integers(1_000_000)))
    sampled = sampled.reset_index(drop=True)

    for column in NUMERIC_COLUMNS:
        if column == "seasonality":
            continue

        std = max(class_source[column].std(ddof=0), 1)
        sampled[column] = sampled[column] + rng.normal(0, std * 0.18, len(sampled))

    sampled = adjust_for_target_risk(sampled, target_risk, rng)

    sampled["min_stock_level"] = clip_round(sampled["min_stock_level"], 10, 80)
    sampled["stock_quantity"] = clip_round(sampled["stock_quantity"], 0, 800)
    sampled["weekly_sales"] = clip_round(sampled["weekly_sales"], 0, 75)
    sampled["supplier_delay_days"] = clip_round(sampled["supplier_delay_days"], 1, 15)
    sampled["price"] = clip_round(sampled["price"], 35, 2500)
    sampled["last_sale_days"] = clip_round(sampled["last_sale_days"], 0, 120)
    sampled["seasonality"] = sampled["seasonality"].astype(int)
    sampled["risk_level"] = target_risk

    valid_mask = sampled.apply(lambda row: matches_target_risk(row, target_risk), axis=1)
    valid = sampled[valid_mask]

    if len(valid) < count:
        extra = sample_class_rows(source, target_risk, count - len(valid), rng)
        valid = pd.concat([valid, extra], ignore_index=True)

    return valid.head(count)


def generate_synthetic_dataset(source_csv=SOURCE_CSV, output_csv=OUTPUT_CSV, total_rows=TARGET_ROWS):
    rng = np.random.default_rng(RANDOM_SEED)
    source = pd.read_csv(source_csv)

    if list(source.columns) != COLUMN_ORDER:
        raise ValueError("Source CSV columns do not match the expected ERP stock schema.")

    rows_per_class = total_rows // len(RISK_LEVELS)
    remainder = total_rows % len(RISK_LEVELS)

    synthetic_parts = []
    for index, risk in enumerate(RISK_LEVELS):
        count = rows_per_class + (1 if index < remainder else 0)
        synthetic_parts.append(sample_class_rows(source, risk, count, rng))

    synthetic = pd.concat(synthetic_parts, ignore_index=True)
    synthetic = synthetic.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    synthetic["product_id"] = np.arange(1, len(synthetic) + 1)
    synthetic = synthetic[COLUMN_ORDER]
    synthetic.to_csv(output_csv, index=False)

    return synthetic


if __name__ == "__main__":
    dataset = generate_synthetic_dataset()
    print(f"Saved {len(dataset)} rows to {OUTPUT_CSV}")
    print(dataset["risk_level"].value_counts().sort_index().to_string())
