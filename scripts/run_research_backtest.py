"""Entry point for the crypto-native residual ridge research backtest."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.research.config import load_research_settings
from src.research.runner import run_research_backtest, summarize_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run research backtest")
    parser.add_argument("--mode", choices=["fast", "accurate"], default="fast")
    parser.add_argument("--data-path", default="data/research")
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--train-days", type=int, default=None)
    parser.add_argument("--val-days", type=int, default=None)
    args = parser.parse_args()

    settings = load_research_settings()
    if args.train_days is not None:
        settings.model.training.train_window_days = args.train_days
    if args.val_days is not None:
        settings.model.training.validation_window_days = args.val_days

    logger.info("Running %s mode backtest", args.mode)
    logger.info("Universe target: top %s coins", settings.universe.liquidity.top_n)
    logger.info(
        "Model windows: train=%sd val=%sd",
        settings.model.training.train_window_days,
        settings.model.training.validation_window_days,
    )

    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error("Data path %s does not exist. Run data fetcher first.", data_path)
        logger.info("Usage: uv run python scripts/fetch_research_data.py")
        return 1

    try:
        prepared, report = run_research_backtest(
            data_path=data_path,
            settings=settings,
            mode=args.mode,
            max_symbols=args.max_symbols,
        )
    except Exception as exc:
        logger.exception("Research backtest failed: %s", exc)
        return 1

    logger.info("Prepared %s tradable symbols and %s observations", len(prepared.tradable_symbols), len(prepared.features))
    logger.info("Backtest summary\n%s", summarize_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
