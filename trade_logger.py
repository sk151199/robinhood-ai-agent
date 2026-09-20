import csv
import logging
import os

from config import settings

os.makedirs(settings.log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(settings.log_dir, "agent.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("robinhood_ai_agent")

TRADE_LOG = os.path.join(settings.log_dir, "trades.csv")
FIELDS = [
    "timestamp",
    "asset_class",
    "symbol",
    "side",
    "requested_dollars",
    "dollars",
    "price",
    "spread_pct",
    "decision",
    "reason",
    "dry_run",
    "ref_id",
    "order_result",
]


def log_trade(row: dict):
    write_header = not os.path.exists(TRADE_LOG)
    with open(TRADE_LOG, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDS})
