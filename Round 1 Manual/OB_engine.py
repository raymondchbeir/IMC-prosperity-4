from dataclasses import dataclass
from typing import List, Tuple, Optional, Literal

Side = Literal["BUY", "SELL"]


@dataclass
class Order:
    price: int
    qty: int
    is_mine: bool = False


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
        if traded > best_volume or (
            traded == best_volume and (best_price is None or p > best_price)
        ):
            best_volume = traded
            best_price = p

    return best_price, best_volume


def exact_fill_for_order(
    bids: List[Order],
    asks: List[Order],
    my_side: Side,
    my_price: int,
    my_qty: int,
    extra_prices: Optional[List[int]] = None,
) -> Tuple[int, Optional[int], int]:
    my_order = Order(price=my_price, qty=my_qty, is_mine=True)

    if my_side == "BUY":
        all_bids = bids + [my_order]
        all_asks = asks
    else:
        all_bids = bids
        all_asks = asks + [my_order]

    p_star, traded = clearing_price_and_volume(all_bids, all_asks, extra_prices)

    if p_star is None:
        return 0, None, 0

    if my_side == "BUY":
        if my_price < p_star:
            return 0, p_star, traded

        opposite_capacity = sum(o.qty for o in asks if o.price <= p_star)

        ahead_qty = 0
        for o in bids:
            if o.price > my_price:
                ahead_qty += o.qty
            elif o.price == my_price:
                ahead_qty += o.qty

        remaining = opposite_capacity - ahead_qty
        fill = max(0, min(my_qty, remaining))
        return fill, p_star, traded

    else:
        if my_price > p_star:
            return 0, p_star, traded

        opposite_capacity = sum(o.qty for o in bids if o.price >= p_star)

        ahead_qty = 0
        for o in asks:
            if o.price < my_price:
                ahead_qty += o.qty
            elif o.price == my_price:
                ahead_qty += o.qty

        remaining = opposite_capacity - ahead_qty
        fill = max(0, min(my_qty, remaining))
        return fill, p_star, traded


def candidate_quantities(
    bids: List[Order],
    asks: List[Order],
    max_qty: int,
) -> List[int]:
    sizes = {1, 5, 10, 20, 50, 100, max_qty}
    for o in bids + asks:
        if 1 <= o.qty <= max_qty:
            sizes.add(o.qty)
    return sorted(sizes)


def score_candidate(
    side: Side,
    fill: int,
    clearing_price: Optional[int],
    fair_value: float,
    my_price: int,
    inventory_penalty_per_unit: float = 0.0,
    aggression_penalty_per_unit: float = 0.0,
) -> float:
    if fill <= 0 or clearing_price is None:
        return float("-inf")

    if side == "BUY":
        edge_per_unit = fair_value - clearing_price
        aggression = max(0.0, my_price - fair_value)
    else:
        edge_per_unit = clearing_price - fair_value
        aggression = max(0.0, fair_value - my_price)

    return (
        fill * edge_per_unit
        - fill * inventory_penalty_per_unit
        - fill * aggression * aggression_penalty_per_unit
    )


def optimal_order_v2(
    bid_levels: List[Tuple[int, int]],
    ask_levels: List[Tuple[int, int]],
    fair_value: float,
    max_qty: int,
    allow_buy: bool = True,
    allow_sell: bool = True,
    extra_price_levels: Optional[List[int]] = None,
    inventory_penalty_per_unit: float = 0.0,
    aggression_penalty_per_unit: float = 0.0,
) -> dict:
    bids = aggregate_orders(bid_levels)
    asks = aggregate_orders(ask_levels)

    prices = candidate_prices(bids, asks, extra_price_levels)
    quantities = candidate_quantities(bids, asks, max_qty)

    best = {
        "side": None,
        "price": None,
        "qty": 0,
        "fill": 0,
        "clearing_price": None,
        "traded_volume": 0,
        "score": float("-inf"),
    }

    sides: List[Side] = []
    if allow_buy:
        sides.append("BUY")
    if allow_sell:
        sides.append("SELL")

    for side in sides:
        for p in prices:
            for q in quantities:
                fill, p_star, traded = exact_fill_for_order(
                    bids=bids,
                    asks=asks,
                    my_side=side,
                    my_price=p,
                    my_qty=q,
                    extra_prices=extra_price_levels,
                )

                score = score_candidate(
                    side=side,
                    fill=fill,
                    clearing_price=p_star,
                    fair_value=fair_value,
                    my_price=p,
                    inventory_penalty_per_unit=inventory_penalty_per_unit,
                    aggression_penalty_per_unit=aggression_penalty_per_unit,
                )

                if score > best["score"]:
                    best = {
                        "side": side,
                        "price": p,
                        "qty": q,
                        "fill": fill,
                        "clearing_price": p_star,
                        "traded_volume": traded,
                        "score": score,
                    }

    return best