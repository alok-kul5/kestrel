"""Paths, split boundaries and the cost matrix in one place."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

# IEEE-CIS TransactionDT is seconds from an undisclosed reference point.
# The usual reading is 2017-12-01, which we use for display only.
EPOCH = "2017-12-01"
DAY = 86_400

# Time-based split (days since the reference). Training never sees a later day.
TRAIN_END_DAY = 120   # days   1-119  -> fit
VALID_END_DAY = 152   # days 120-151  -> calibration, threshold, model choice
#                       days 152-182  -> untouched test month

# Cost matrix in dollars.
COST_REVIEW = 5.0        # analyst review / customer friction for every flagged txn
COST_CHARGEBACK_FEE = 20.0  # fee on top of the lost amount when fraud gets through

SEED = 7
