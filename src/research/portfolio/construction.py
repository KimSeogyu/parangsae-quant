"""Long/short portfolio construction for the research pipeline.

PRD Section 7: Portfolio Construction
- Cross-sectional long/short: top 20% long, bottom 20% short
- Long 6 / Short 6 from ~30 coin universe
- Hysteresis: entry top/bottom 20%, exit top/bottom 35%
- Weight: clip(score_z, ±2) / rvol_1d, normalized to gross 1.0
- Position caps: single-name 12%, sector 30%
- Beta cap: abs(portfolio beta) ≤ 0.10
- Net exposure cap: abs(net) ≤ 0.15
- Rebalance every 20 minutes
"""

from __future__ import annotations



def select_long_short(
    scores: dict[str, float],
    current_longs: set[str],
    current_shorts: set[str],
    long_count: int = 6,
    short_count: int = 6,
    entry_bucket_pct: float = 0.20,
    exit_bucket_pct: float = 0.35,
) -> tuple[set[str], set[str]]:
    """Select long and short holdings with hysteresis.

    Args:
        scores: {symbol: combined_score}
        current_longs: Currently held long positions.
        current_shorts: Currently held short positions.
        long_count: Target number of longs.
        short_count: Target number of shorts.
        entry_bucket_pct: Top/bottom % for entry.
        exit_bucket_pct: Exit when outside this %.

    Returns:
        (new_longs, new_shorts) sets.
    """
    if not scores:
        return set(), set()

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    n = len(ranked)
    entry_n = max(1, int(n * entry_bucket_pct))
    exit_n = max(1, int(n * exit_bucket_pct))

    # Entry candidates: top entry_n for long, bottom entry_n for short
    entry_long_candidates = {s for s, _ in ranked[:entry_n] if scores[s] > 0}
    entry_short_candidates = {s for s, _ in ranked[-entry_n:] if scores[s] < 0}

    # Exit zones: outside top exit_n for long, outside bottom exit_n for short
    stay_long_zone = {s for s, _ in ranked[:exit_n]}
    stay_short_zone = {s for s, _ in ranked[-exit_n:]}

    # Hysteresis: keep existing holdings if still in stay zone
    new_longs = current_longs & stay_long_zone
    new_shorts = current_shorts & stay_short_zone

    # Add new entries if below target count
    for sym in entry_long_candidates:
        if len(new_longs) >= long_count:
            break
        if sym not in new_shorts:
            new_longs.add(sym)

    for sym in entry_short_candidates:
        if len(new_shorts) >= short_count:
            break
        if sym not in new_longs:
            new_shorts.add(sym)

    return new_longs, new_shorts


def compute_weights(
    longs: set[str],
    shorts: set[str],
    scores: dict[str, float],
    volatilities: dict[str, float],
    gross: float = 1.0,
    score_clip: float = 2.0,
) -> dict[str, float]:
    """Compute position weights: clip(score_z, ±2) / rvol, normalized.

    Positive weights = long, negative weights = short.
    Total abs(weights) = gross.
    """
    raw = {}
    for sym in longs:
        s = min(max(scores.get(sym, 0.0), -score_clip), score_clip)
        vol = volatilities.get(sym, 1.0)
        if vol < 1e-10:
            vol = 1.0
        raw[sym] = abs(s) / vol  # Long: positive

    for sym in shorts:
        s = min(max(scores.get(sym, 0.0), -score_clip), score_clip)
        vol = volatilities.get(sym, 1.0)
        if vol < 1e-10:
            vol = 1.0
        raw[sym] = -(abs(s) / vol)  # Short: negative

    total_abs = sum(abs(v) for v in raw.values())
    if total_abs < 1e-10:
        return {}

    scale = gross / total_abs
    return {sym: w * scale for sym, w in raw.items()}


def apply_position_caps(
    weights: dict[str, float],
    single_name_cap: float = 0.12,
    sector_map: dict[str, str] | None = None,
    sector_soft_cap: float = 0.30,
) -> dict[str, float]:
    """Apply single-name and sector caps.

    Excess weight is redistributed proportionally.
    """
    if not weights:
        return {}

    # Single-name cap
    capped = {}
    for sym, w in weights.items():
        if abs(w) > single_name_cap:
            capped[sym] = single_name_cap * (1 if w > 0 else -1)
        else:
            capped[sym] = w

    # Sector cap
    if sector_map:
        sector_weights: dict[str, float] = {}
        for sym, w in capped.items():
            sector = sector_map.get(sym, "other")
            sector_weights[sector] = sector_weights.get(sector, 0.0) + abs(w)

        for sector, total in sector_weights.items():
            if total > sector_soft_cap:
                ratio = sector_soft_cap / total
                for sym in capped:
                    if sector_map.get(sym, "other") == sector:
                        capped[sym] *= ratio

    return capped


def apply_beta_hedge(
    weights: dict[str, float],
    betas: dict[str, float],
    btc_symbol: str = "BTCUSDT",
    max_beta: float = 0.10,
) -> dict[str, float]:
    """Apply BTC hedge to enforce portfolio beta neutrality.

    If abs(portfolio_beta) > max_beta, add/reduce BTC perp position.
    """
    if not weights:
        return weights

    portfolio_beta = sum(w * betas.get(sym, 1.0) for sym, w in weights.items())

    if abs(portfolio_beta) <= max_beta:
        return weights

    # Hedge: need to add BTC position to offset excess beta
    btc_beta = betas.get(btc_symbol, 1.0)
    if abs(btc_beta) < 1e-10:
        return weights

    hedge_weight = -portfolio_beta / btc_beta
    result = dict(weights)
    result[btc_symbol] = result.get(btc_symbol, 0.0) + hedge_weight
    return result


def apply_net_cap(
    weights: dict[str, float],
    net_cap: float = 0.15,
) -> dict[str, float]:
    """Cap net exposure to ±net_cap.

    Scales long and short sides proportionally if net is too large.
    """
    if not weights:
        return weights

    net = sum(weights.values())
    if abs(net) <= net_cap:
        return weights

    # Scale the side causing imbalance
    long_total = sum(w for w in weights.values() if w > 0)
    short_total = sum(w for w in weights.values() if w < 0)

    if net > net_cap:
        # Too long: scale down longs
        if long_total > 0:
            target_long = short_total * -1 + net_cap  # short_total is negative
            if target_long > 0:
                ratio = target_long / long_total
                return {
                    s: (w * ratio if w > 0 else w) for s, w in weights.items()
                }
    else:
        # Too short: scale down shorts (make less negative)
        if short_total < 0:
            target_short = -(long_total - net_cap)
            if target_short < 0:
                ratio = target_short / short_total
                return {
                    s: (w * ratio if w < 0 else w) for s, w in weights.items()
                }

    return weights


def build_portfolio(
    scores: dict[str, float],
    volatilities: dict[str, float],
    betas: dict[str, float],
    current_longs: set[str] | None = None,
    current_shorts: set[str] | None = None,
    sector_map: dict[str, str] | None = None,
    long_count: int = 6,
    short_count: int = 6,
    entry_bucket_pct: float = 0.20,
    exit_bucket_pct: float = 0.35,
    gross: float = 1.0,
    score_clip: float = 2.0,
    single_name_cap: float = 0.12,
    sector_soft_cap: float = 0.30,
    btc_beta_cap: float = 0.10,
    net_cap: float = 0.15,
) -> tuple[dict[str, float], set[str], set[str]]:
    """Full portfolio construction pipeline.

    Returns:
        (weights, longs, shorts)
    """
    if current_longs is None:
        current_longs = set()
    if current_shorts is None:
        current_shorts = set()

    longs, shorts = select_long_short(
        scores, current_longs, current_shorts,
        long_count, short_count, entry_bucket_pct, exit_bucket_pct,
    )

    weights = compute_weights(longs, shorts, scores, volatilities, gross, score_clip)
    weights = apply_position_caps(weights, single_name_cap, sector_map, sector_soft_cap)
    weights = apply_beta_hedge(weights, betas, max_beta=btc_beta_cap)
    weights = apply_net_cap(weights, net_cap)

    return weights, longs, shorts
