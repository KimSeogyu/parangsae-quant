from __future__ import annotations

import logging
import sys

from src.engine.backtest import run_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/settings.yaml"
    engine = run_backtest(config_path)

    for report in engine.trader.generate_order_fills_report():
        print(report)
    for report in engine.trader.generate_positions_report():
        print(report)
    for report in engine.trader.generate_account_report(venue=None):
        print(report)


if __name__ == "__main__":
    main()
