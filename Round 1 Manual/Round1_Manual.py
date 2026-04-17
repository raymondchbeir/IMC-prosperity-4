from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class Order:
    price: int
    qty: int
    is_mine: bool = False


# =========================
# Core auction mechanics
# =========================

def aggregate_orders(levels: List[Tuple[int, int]]) -> List[Order]:
    return [Order(price=p, qty=q, is_mine=False) for p, q in levels if q > 0]


def candidate_prices(
    bids: List[Order],
    asks: List[Order],
    extra_prices: Optional[List[int]] = None,
) -> List[int]:
    prices = {o.price for o in bids} | {o.price for o in asks}
    if extra_prices:
        prices |= set(extra_prices)
    return sorted(prices)


def eligible_bid_volume(bids: List[Order], p: int) -> int:
    return sum(o.qty for o in bids if o.price >= p)


def eligible_ask_volume(asks: List[Order], p: int) -> int:
    return sum(o.qty for o in asks if o.price <= p)


def clearing_price_and_volume(
    bids: List[Order],
    asks: List[Order],
    extra_prices: Optional[List[int]] = None,
) -> Tuple[Optional[int], int]:
    prices = candidate_prices(bids, asks, extra_prices)
    if not prices:
        return None, 0

    best_price = None
    best_volume = -1

    for p in prices:
        traded = min(eligible_bid_volume(bids, p), eligible_ask_volume(asks, p))
        if traded > best_volume or (traded == best_volume and (best_price is None or p > best_price)):
            best_volume = traded
            best_price = p

    return best_price, best_volume


# =========================
# Fill logic for your order
# =========================

def my_fill_if_buy(
    bids: List[Order],
    asks: List[Order],
    my_order: Order,
    extra_prices: Optional[List[int]] = None,
) -> Tuple[int, Optional[int], int, int]:
    all_bids = bids + [my_order]
    p_star, traded = clearing_price_and_volume(all_bids, asks, extra_prices)

    if p_star is None or my_order.price < p_star:
        return 0, p_star, traded, 0

    ahead = 0
    for o in bids:
        if o.price > my_order.price:
            ahead += o.qty
        elif o.price == my_order.price:
            ahead += o.qty

    fill = max(0, min(my_order.qty, traded - ahead))
    return fill, p_star, traded, ahead


def my_fill_if_sell(
    bids: List[Order],
    asks: List[Order],
    my_order: Order,
    extra_prices: Optional[List[int]] = None,
) -> Tuple[int, Optional[int], int, int]:
    all_asks = asks + [my_order]
    p_star, traded = clearing_price_and_volume(bids, all_asks, extra_prices)

    if p_star is None or my_order.price > p_star:
        return 0, p_star, traded, 0

    ahead = 0
    for o in asks:
        if o.price < my_order.price:
            ahead += o.qty
        elif o.price == my_order.price:
            ahead += o.qty

    fill = max(0, min(my_order.qty, traded - ahead))
    return fill, p_star, traded, ahead


# =========================
# One-order simulation
# =========================

def simulate_order(
    bid_levels: List[Tuple[int, int]],
    ask_levels: List[Tuple[int, int]],
    side: str,
    my_price: int,
    my_qty: int,
    current_position: int = 0,
    target_position_after_fill: Optional[int] = None,
    fair_value: Optional[float] = None,
    future_exit_price: Optional[float] = None,
    fee_per_unit: float = 0.0,
    extra_price_levels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    side = side.upper().strip()
    bids = aggregate_orders(bid_levels)
    asks = aggregate_orders(ask_levels)
    my_order = Order(price=my_price, qty=my_qty, is_mine=True)

    base_clearing_price, base_traded = clearing_price_and_volume(bids, asks, extra_price_levels)

    if side == "BUY":
        fill, new_clearing_price, new_traded, ahead = my_fill_if_buy(
            bids, asks, my_order, extra_price_levels
        )
        signed_fill = fill
        new_position = current_position + fill
    elif side == "SELL":
        fill, new_clearing_price, new_traded, ahead = my_fill_if_sell(
            bids, asks, my_order, extra_price_levels
        )
        signed_fill = -fill
        new_position = current_position - fill
    else:
        raise ValueError("side must be 'BUY' or 'SELL'")

    result: Dict[str, Any] = {
        "side": side,
        "order_price": my_price,
        "order_qty": my_qty,
        "current_position": current_position,
        "base_clearing_price": base_clearing_price,
        "base_traded_volume": base_traded,
        "new_clearing_price": new_clearing_price,
        "new_traded_volume": new_traded,
        "clearing_price_change": None if base_clearing_price is None or new_clearing_price is None else new_clearing_price - base_clearing_price,
        "fill_qty": fill,
        "signed_fill": signed_fill,
        "queue_ahead_at_my_price": ahead,
        "new_position": new_position,
        "position_gap_to_target": None,
        "edge_vs_fair_per_unit": None,
        "mark_edge_total": None,
        "expected_unwind_pnl_per_unit": None,
        "expected_unwind_pnl_total": None,
    }

    if target_position_after_fill is not None:
        result["position_gap_to_target"] = target_position_after_fill - new_position

    if fair_value is not None and new_clearing_price is not None:
        if side == "BUY":
            edge_per_unit = fair_value - new_clearing_price - fee_per_unit
        else:
            edge_per_unit = new_clearing_price - fair_value - fee_per_unit
        result["edge_vs_fair_per_unit"] = edge_per_unit
        result["mark_edge_total"] = fill * edge_per_unit

    if future_exit_price is not None and new_clearing_price is not None:
        if side == "BUY":
            pnl_per_unit = future_exit_price - new_clearing_price - fee_per_unit
        else:
            pnl_per_unit = new_clearing_price - future_exit_price - fee_per_unit
        result["expected_unwind_pnl_per_unit"] = pnl_per_unit
        result["expected_unwind_pnl_total"] = fill * pnl_per_unit

    return result


# =========================
# Brute-force optimizer
# =========================

def _score_result(result: Dict[str, Any], objective: str) -> Optional[float]:
    if objective == "fill_qty":
        return float(result["fill_qty"])

    if objective == "mark_edge_total":
        value = result["mark_edge_total"]
        return None if value is None else float(value)

    if objective == "expected_unwind_pnl_total":
        value = result["expected_unwind_pnl_total"]
        return None if value is None else float(value)

    if objective == "closest_to_target":
        gap = result["position_gap_to_target"]
        return None if gap is None else -float(abs(gap))

    raise ValueError(
        "objective must be one of: fill_qty, mark_edge_total, expected_unwind_pnl_total, closest_to_target"
    )


def brute_force_optimal_order(
    bid_levels: List[Tuple[int, int]],
    ask_levels: List[Tuple[int, int]],
    allowed_sides: List[str],
    price_candidates: List[int],
    qty_candidates: List[int],
    current_position: int = 0,
    target_position_after_fill: Optional[int] = None,
    fair_value: Optional[float] = None,
    future_exit_price: Optional[float] = None,
    fee_per_unit: float = 0.0,
    extra_price_levels: Optional[List[int]] = None,
    objective: str = "expected_unwind_pnl_total",
    require_fill: bool = True,
) -> Dict[str, Any]:
    best_result: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[float, int, int, int]] = None

    for raw_side in allowed_sides:
        side = raw_side.upper().strip()
        if side not in {"BUY", "SELL"}:
            raise ValueError("allowed_sides must only contain BUY and/or SELL")

        for p in price_candidates:
            for q in qty_candidates:
                if q <= 0:
                    continue

                result = simulate_order(
                    bid_levels=bid_levels,
                    ask_levels=ask_levels,
                    side=side,
                    my_price=p,
                    my_qty=q,
                    current_position=current_position,
                    target_position_after_fill=target_position_after_fill,
                    fair_value=fair_value,
                    future_exit_price=future_exit_price,
                    fee_per_unit=fee_per_unit,
                    extra_price_levels=extra_price_levels,
                )

                if require_fill and result["fill_qty"] <= 0:
                    continue

                score = _score_result(result, objective)
                if score is None:
                    continue

                if side == "BUY":
                    price_tiebreak = p
                else:
                    price_tiebreak = -p

                key = (
                    score,
                    result["fill_qty"],
                    -q,
                    price_tiebreak,
                )

                if best_key is None or key > best_key:
                    best_key = key
                    best_result = result

    if best_result is None:
        raise ValueError("No valid order found. Widen your price/qty ranges or relax require_fill.")

    return best_result


# =========================
# Output helpers
# =========================

def print_book(bid_levels: List[Tuple[int, int]], ask_levels: List[Tuple[int, int]]) -> None:
    print("\nORDER BOOK")
    print("----------")
    print("Bids:")
    for p, q in sorted(bid_levels, key=lambda x: (-x[0], -x[1])):
        print(f"  {p:>5} x {q}")
    print("Asks:")
    for p, q in sorted(ask_levels, key=lambda x: (x[0], -x[1])):
        print(f"  {p:>5} x {q}")


def print_optimal_order(result: Dict[str, Any], objective: str) -> None:
    print("\nOPTIMAL ORDER")
    print("-------------")
    print(f"Objective:               {objective}")
    print(f"Side:                    {result['side']}")
    print(f"Price:                   {result['order_price']}")
    print(f"Quantity:                {result['order_qty']}")
    print(f"Expected fill:           {result['fill_qty']}")
    print(f"Base clearing price:     {result['base_clearing_price']}")
    print(f"New clearing price:      {result['new_clearing_price']}")
    print(f"Clearing price change:   {result['clearing_price_change']}")
    print(f"Queue ahead at price:    {result['queue_ahead_at_my_price']}")
    print(f"New position:            {result['new_position']}")
    print(f"Gap to target:           {result['position_gap_to_target']}")
    print(f"Mark edge total:         {result['mark_edge_total']}")
    print(f"Expected unwind pnl:     {result['expected_unwind_pnl_total']}")


# =========================
# Plug-and-play area
# =========================
if __name__ == "__main__":
    bid_levels = [
        (10995, 9),
        (10992, 25),
    ]
    ask_levels = [
        (11009, 25),
    ]

    allowed_sides = ["BUY", "SELL"]
    price_candidates = [10992, 10995, 11009]
    qty_candidates = list(range(1, 26))

    current_position = 0
    target_position_after_fill = 0
    fair_value = 11000.0
    future_exit_price = 11000.0
    fee_per_unit = 0.0
    extra_price_levels = [10992, 10995, 11009]

    objective = "expected_unwind_pnl_total"

    best = brute_force_optimal_order(
        bid_levels=bid_levels,
        ask_levels=ask_levels,
        allowed_sides=allowed_sides,
        price_candidates=price_candidates,
        qty_candidates=qty_candidates,
        current_position=current_position,
        target_position_after_fill=target_position_after_fill,
        fair_value=fair_value,
        future_exit_price=future_exit_price,
        fee_per_unit=fee_per_unit,
        extra_price_levels=extra_price_levels,
        objective=objective,
        require_fill=True,
    )

    print_book(bid_levels, ask_levels)
    print_optimal_order(best, objective)