"""
CS611 Assignment 1 — loan default data pipeline.

Run with:  python main.py

    RAW CSV  ->  BRONZE      ->  SILVER        ->  GOLD
                 raw, as-is      typed+cleaned     feature_store
                 partitioned     validated         label_store
"""

import sys
import time

import os
from utils.gold_label_store import build_label_store

from utils import config
from utils.bronze_processing import run_bronze_layer
from utils.silver_processing import run_silver_layer
from utils.spark_utils import generate_snapshot_dates, get_spark
from utils.gold_feature_store import build_feature_store
from utils.data_quality import validate


def main() -> int:
    started = time.time()

    print("=" * 78)
    print("CS611 ASSIGNMENT 1 | LOAN DEFAULT DATA PIPELINE")
    print("=" * 78)

    snapshot_dates = generate_snapshot_dates(config.START_DATE, config.END_DATE)
    print(f"Batch window : {config.START_DATE} -> {config.END_DATE} "
          f"({len(snapshot_dates)} monthly partitions)")
    print(f"Label        : {config.LABEL_DEFINITION}")

    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")     # hide Spark's INFO noise

    try:
        run_bronze_layer(spark, snapshot_dates)
        run_silver_layer(spark, snapshot_dates)

        print("\n[3/4] GOLD LAYER - business-level aggregates")
        os.makedirs(config.GOLD_DIR, exist_ok=True)
        build_label_store(spark, snapshot_dates)
        build_feature_store(spark, snapshot_dates)
        all_checks_passed = validate(spark)
    finally:
        spark.stop()

    print(f"\nDONE in {time.time() - started:.1f}s")
    # non-zero exit lets a scheduler treat a validation failure as a failed run
    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    sys.exit(main())
