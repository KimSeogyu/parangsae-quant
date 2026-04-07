"""Create Nautilus instruments from CCXT market data."""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from pathlib import Path

from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.enums import CurrencyType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoPerpetual, CurrencyPair
from nautilus_trader.model.objects import Currency, Money, Price, Quantity

logger = logging.getLogger(__name__)

VENUE = Venue("BINANCE")


def ccxt_symbol_to_file_name(ccxt_symbol: str, market_type: str) -> str:
    """Convert CCXT symbol to our file naming convention.

    Spot:    "BTC/USDT"       → "BTCUSDT"
    Futures: "BTC/USDT:USDT"  → "BTCUSDT-PERP"
    """
    if market_type == "futures":
        base_quote = ccxt_symbol.split(":")[0]
        return base_quote.replace("/", "") + "-PERP"
    return ccxt_symbol.replace("/", "")


def _get_or_create_currency(code: str, precision: int = 8) -> Currency:
    """Get a built-in currency or create a new crypto currency."""
    try:
        return Currency.from_str(code)
    except Exception:
        return Currency(
            code=code,
            precision=precision,
            iso4217=0,
            name=code,
            currency_type=CurrencyType.CRYPTO,
        )


def _precision_to_increment(precision: int) -> str:
    """Convert decimal precision to increment string. e.g. 2 → '0.01', 5 → '0.00001'."""
    if precision <= 0:
        return "1"
    return f"0.{'0' * (precision - 1)}1"


def save_market_info(markets: dict, market_type: str, storage_path: str) -> None:
    """Save CCXT market info to JSON for later instrument creation."""
    out_dir = Path(storage_path) / market_type
    out_dir.mkdir(parents=True, exist_ok=True)

    info = {}
    for ccxt_symbol, market in markets.items():
        if not market["active"] or market["quote"] != "USDT":
            continue
        if market_type == "spot" and not market.get("spot"):
            continue
        if market_type == "futures" and not (market.get("swap") and market.get("linear")):
            continue

        file_name = ccxt_symbol_to_file_name(ccxt_symbol, market_type)
        info[file_name] = {
            "ccxt_symbol": ccxt_symbol,
            "base": market["base"],
            "quote": market["quote"],
            "settle": market.get("settle", ""),
            "price_precision": market["precision"].get("price", 2),
            "amount_precision": market["precision"].get("amount", 6),
            "min_amount": market["limits"].get("amount", {}).get("min", 0.001),
            "max_amount": market["limits"].get("amount", {}).get("max", 100000),
            "min_cost": market["limits"].get("cost", {}).get("min", 10.0),
            "min_price": market["limits"].get("price", {}).get("min", 0.01),
            "max_price": market["limits"].get("price", {}).get("max", 1000000),
            "taker_fee": market.get("taker", 0.001),
            "maker_fee": market.get("maker", 0.001),
        }

    meta_path = out_dir / "_instruments.json"
    with open(meta_path, "w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"Saved {len(info)} instrument specs to {meta_path}")


def load_instruments(storage_path: str, market_type: str) -> dict[str, object]:
    """Load instruments from saved market info. Returns {file_name: Nautilus Instrument}."""
    meta_path = Path(storage_path) / market_type / "_instruments.json"
    if not meta_path.exists():
        logger.warning(f"No instrument metadata at {meta_path}")
        return {}

    with open(meta_path) as f:
        info = json.load(f)

    instruments = {}
    for file_name, spec in info.items():
        try:
            instrument = _create_instrument_from_spec(file_name, spec, market_type)
            if instrument:
                instruments[file_name] = instrument
        except Exception as e:
            logger.debug(f"Skipping {file_name}: {e}")
    logger.info(f"Loaded {len(instruments)} {market_type} instruments")
    return instruments


def _create_instrument_from_spec(file_name: str, spec: dict, market_type: str):
    """Create a Nautilus instrument from saved CCXT market spec."""
    base_currency = _get_or_create_currency(spec["base"])
    quote_currency = _get_or_create_currency(spec["quote"])

    price_prec = int(spec.get("price_precision", 2))
    amount_prec = int(spec.get("amount_precision", 6))
    price_inc = _precision_to_increment(price_prec)
    size_inc = _precision_to_increment(amount_prec)

    min_qty = spec.get("min_amount") or 0.001
    max_qty = spec.get("max_amount") or 100000
    min_price = spec.get("min_price") or 0.01
    max_price = spec.get("max_price") or 1000000
    min_cost = spec.get("min_cost") or 10.0

    if market_type == "futures":
        settle_currency = _get_or_create_currency(spec.get("settle", "USDT"))
        return CryptoPerpetual(
            instrument_id=InstrumentId(Symbol(file_name), VENUE),
            raw_symbol=Symbol(file_name.replace("-PERP", "")),
            base_currency=base_currency,
            quote_currency=quote_currency,
            settlement_currency=settle_currency,
            is_inverse=False,
            price_precision=price_prec,
            size_precision=amount_prec,
            price_increment=Price.from_str(price_inc),
            size_increment=Quantity.from_str(size_inc),
            max_quantity=Quantity.from_str(f"{max_qty:.{amount_prec}f}"),
            min_quantity=Quantity.from_str(f"{min_qty:.{amount_prec}f}"),
            max_notional=None,
            min_notional=Money(min_cost, quote_currency),
            max_price=Price.from_str(f"{max_price:.{price_prec}f}"),
            min_price=Price.from_str(f"{min_price:.{price_prec}f}"),
            margin_init=Decimal("0.05"),
            margin_maint=Decimal("0.025"),
            maker_fee=Decimal(str(spec.get("maker_fee", 0.001))),
            taker_fee=Decimal(str(spec.get("taker_fee", 0.001))),
            ts_event=0,
            ts_init=0,
        )
    else:
        return CurrencyPair(
            instrument_id=InstrumentId(Symbol(file_name), VENUE),
            raw_symbol=Symbol(file_name),
            base_currency=base_currency,
            quote_currency=quote_currency,
            price_precision=price_prec,
            size_precision=amount_prec,
            price_increment=Price.from_str(price_inc),
            size_increment=Quantity.from_str(size_inc),
            lot_size=None,
            max_quantity=Quantity.from_str(f"{max_qty:.{amount_prec}f}"),
            min_quantity=Quantity.from_str(f"{min_qty:.{amount_prec}f}"),
            max_notional=None,
            min_notional=Money(min_cost, quote_currency),
            max_price=Price.from_str(f"{max_price:.{price_prec}f}"),
            min_price=Price.from_str(f"{min_price:.{price_prec}f}"),
            margin_init=Decimal("0"),
            margin_maint=Decimal("0"),
            maker_fee=Decimal(str(spec.get("maker_fee", 0.001))),
            taker_fee=Decimal(str(spec.get("taker_fee", 0.001))),
            ts_event=0,
            ts_init=0,
        )
