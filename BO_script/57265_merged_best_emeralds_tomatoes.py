from datamodel import Order, TradingState, OrderDepth
from typing import Dict, List, Any
import json
import math


class Trader:
    EMERALDS = "EMERALDS"
    TOMATOES = "TOMATOES"

    POSITION_LIMITS = {
        "EMERALDS": 80,
        "TOMATOES": 80,
    }

    # =============================
    # EMERALDS params
    # =============================
    EMERALDS_FAIR = 10000
    EMERALDS_BASE_SIZE = 10
    EMERALDS_AGGR_THRESHOLD = 60
    EMERALDS_AGGR_SIZE = 20

    # =============================
    # TOMATOES fair-value params
    # =============================
    TOM_FAIR_ALPHA = 0.18
    TOM_PRESSURE_BETA = 0.90
    TOM_DEPTH_LEVELS = 3

    TOM_TAKE_EDGE = 1.0
    TOM_MAX_TAKE_SIZE = 25
    TOM_PASSIVE_SIZE = 12
    TOM_PASSIVE_EDGE = 1
    TOM_MIN_SPREAD_FOR_MAKING = 3

    TOM_INVENTORY_SKEW = 0.03
    TOM_HEAVY_POS = 40

    # -------------------------------------------------
    # State helpers
    # -------------------------------------------------
    def _default_state(self) -> Dict[str, Any]:
        return {
            "em_long_qty": 0,
            "em_long_cost": 0.0,
            "em_short_qty": 0,
            "em_short_proceeds": 0.0,
            "em_last_trade_ts": -1,
            "tom_fair": None,
        }

    def _load_state(self, traderData: str) -> Dict[str, Any]:
        if not traderData:
            return self._default_state()
        try:
            loaded = json.loads(traderData)
            base = self._default_state()
            base.update(loaded)
            return base
        except Exception:
            return self._default_state()

    def _dump_state(self, s: Dict[str, Any]) -> str:
        return json.dumps(s, separators=(",", ":"))

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def _best_prices(self, order_depth: OrderDepth):
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def _top_depth_totals(self, depth: OrderDepth, levels: int):
        bid_levels = sorted(depth.buy_orders.items(), key=lambda x: x[0], reverse=True)[:levels]
        ask_levels = sorted(depth.sell_orders.items(), key=lambda x: x[0])[:levels]

        bid_total = sum(qty for _, qty in bid_levels)
        ask_total = sum(-qty for _, qty in ask_levels)
        return bid_total, ask_total


    def _update_emerald_inventory_from_trades(
        self,
        state: TradingState,
        mem: Dict[str, Any],
        fallback_buy_price: int,
        fallback_sell_price: int,
    ) -> None:
        long_qty = int(mem.get("em_long_qty", 0))
        long_cost = float(mem.get("em_long_cost", 0.0))
        short_qty = int(mem.get("em_short_qty", 0))
        short_proceeds = float(mem.get("em_short_proceeds", 0.0))
        last_ts = int(mem.get("em_last_trade_ts", -1))

        trades = []
        if hasattr(state, "own_trades") and self.EMERALDS in state.own_trades:
            trades = state.own_trades[self.EMERALDS]

        max_seen_ts = last_ts

        try:
            trades = sorted(trades, key=lambda t: getattr(t, "timestamp", -1))
        except Exception:
            pass

        for trade in trades:
            ts = int(getattr(trade, "timestamp", -1))
            if ts <= last_ts:
                continue

            price = float(getattr(trade, "price", 0))
            qty = int(getattr(trade, "quantity", 0))
            buyer = getattr(trade, "buyer", None)
            seller = getattr(trade, "seller", None)

            if buyer == "SUBMISSION":
                cover_qty = min(short_qty, qty)
                if cover_qty > 0:
                    avg_short = short_proceeds / short_qty if short_qty > 0 else 0.0
                    short_proceeds -= avg_short * cover_qty
                    short_qty -= cover_qty
                    qty -= cover_qty
                    if short_qty == 0:
                        short_proceeds = 0.0

                if qty > 0:
                    long_cost += price * qty
                    long_qty += qty

            elif seller == "SUBMISSION":
                reduce_long = min(long_qty, qty)
                if reduce_long > 0:
                    avg_long = long_cost / long_qty if long_qty > 0 else 0.0
                    long_cost -= avg_long * reduce_long
                    long_qty -= reduce_long
                    qty -= reduce_long
                    if long_qty == 0:
                        long_cost = 0.0

                if qty > 0:
                    short_proceeds += price * qty
                    short_qty += qty

            if ts > max_seen_ts:
                max_seen_ts = ts

        actual_pos = state.position.get(self.EMERALDS, 0)

        if actual_pos > 0:
            short_qty = 0
            short_proceeds = 0.0
            if long_qty > actual_pos:
                avg = long_cost / long_qty if long_qty > 0 else 0.0
                long_qty = actual_pos
                long_cost = avg * long_qty
            elif long_qty < actual_pos:
                missing = actual_pos - long_qty
                long_cost += fallback_buy_price * missing
                long_qty = actual_pos

        elif actual_pos < 0:
            long_qty = 0
            long_cost = 0.0
            needed_short = -actual_pos
            if short_qty > needed_short:
                avg = short_proceeds / short_qty if short_qty > 0 else 0.0
                short_qty = needed_short
                short_proceeds = avg * short_qty
            elif short_qty < needed_short:
                missing = needed_short - short_qty
                short_proceeds += fallback_sell_price * missing
                short_qty = needed_short

        else:
            long_qty = 0
            long_cost = 0.0
            short_qty = 0
            short_proceeds = 0.0

        mem["em_long_qty"] = long_qty
        mem["em_long_cost"] = long_cost
        mem["em_short_qty"] = short_qty
        mem["em_short_proceeds"] = short_proceeds
        mem["em_last_trade_ts"] = max_seen_ts

    def _emerald_buy_allowed(self, pos: int, avg_long: float, price: int, qty: int) -> bool:
        if qty <= 0:
            return False

        if pos > 60 and price >= self.EMERALDS_FAIR:
            return False

        if pos <= 0 or avg_long is None:
            return True

        if avg_long > 9994:
            return price < avg_long

        projected_avg = ((avg_long * pos) + (price * qty)) / (pos + qty)
        return projected_avg <= 9994

    # -------------------------------------------------
    # TOMATOES fair value helpers
    # -------------------------------------------------
    def _instant_tom_fair_value(self, depth: OrderDepth) -> float | None:
        if not depth.buy_orders or not depth.sell_orders:
            return None

        best_bid, best_ask = self._best_prices(depth)
        if best_bid is None or best_ask is None:
            return None

        best_bid_vol = depth.buy_orders.get(best_bid, 0)
        best_ask_vol = -depth.sell_orders.get(best_ask, 0)

        if best_bid_vol <= 0 or best_ask_vol <= 0:
            return (best_bid + best_ask) / 2.0

        microprice = (
            best_bid * best_ask_vol + best_ask * best_bid_vol
        ) / (best_bid_vol + best_ask_vol)

        bid_depth, ask_depth = self._top_depth_totals(depth, self.TOM_DEPTH_LEVELS)
        depth_total = bid_depth + ask_depth

        imbalance = 0.0
        if depth_total > 0:
            imbalance = (bid_depth - ask_depth) / depth_total

        spread = best_ask - best_bid
        half_spread = spread / 2.0 if spread > 0 else 1.0

        fair = microprice + self.TOM_PRESSURE_BETA * imbalance * half_spread
        return fair

    def _update_tom_fair(self, mem: Dict[str, Any], instant_fair: float) -> float:
        prev_fair = mem.get("tom_fair", None)

        if prev_fair is None:
            fair = float(instant_fair)
        else:
            fair = (1.0 - self.TOM_FAIR_ALPHA) * float(prev_fair) + self.TOM_FAIR_ALPHA * float(instant_fair)

        mem["tom_fair"] = fair
        return fair

    # -------------------------------------------------
    # EMERALDS strategy
    # -------------------------------------------------
    def _trade_emeralds(self, state: TradingState, mem: Dict[str, Any]) -> List[Order]:
        orders: List[Order] = []

        if self.EMERALDS not in state.order_depths:
            return orders

        depth: OrderDepth = state.order_depths[self.EMERALDS]
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid, best_ask = self._best_prices(depth)
        pos = state.position.get(self.EMERALDS, 0)
        limit = self.POSITION_LIMITS[self.EMERALDS]

        max_buy_allowed = max(0, limit - pos)
        max_sell_allowed = max(0, limit + pos)

        buy_price = best_bid + 1
        sell_price = best_ask - 1

        if buy_price >= sell_price:
            buy_price = best_bid
            sell_price = best_ask

        self._update_emerald_inventory_from_trades(state, mem, buy_price, sell_price)

        em_long_qty = int(mem.get("em_long_qty", 0))
        em_long_cost = float(mem.get("em_long_cost", 0.0))
        avg_long = (em_long_cost / em_long_qty) if em_long_qty > 0 else None

        if self.EMERALDS_FAIR in depth.sell_orders:
            fair_ask_qty = -depth.sell_orders[self.EMERALDS_FAIR]

            if pos <= 40:
                buy_qty = min(fair_ask_qty, max_buy_allowed)
            elif pos <= 60:
                buy_qty = min(max(1, fair_ask_qty // 2), max_buy_allowed)
            else:
                buy_qty = 0

            if avg_long is not None and avg_long > 9994 and self.EMERALDS_FAIR >= avg_long:
                buy_qty = 0

            if buy_qty > 0:
                orders.append(Order(self.EMERALDS, self.EMERALDS_FAIR, buy_qty))

        if self.EMERALDS_FAIR in depth.buy_orders:
            fair_bid_qty = depth.buy_orders[self.EMERALDS_FAIR]
            sell_qty = min(fair_bid_qty, max_sell_allowed)
            if sell_qty > 0:
                orders.append(Order(self.EMERALDS, self.EMERALDS_FAIR, -sell_qty))

        buy_qty = self.EMERALDS_BASE_SIZE
        sell_qty = self.EMERALDS_BASE_SIZE

        if pos >= self.EMERALDS_AGGR_THRESHOLD:
            sell_qty = self.EMERALDS_AGGR_SIZE
            sell_price = max(best_bid + 1, sell_price - 1)
            if buy_price >= sell_price:
                sell_price = best_ask - 1 if best_ask - 1 > buy_price else best_ask

        elif pos <= -self.EMERALDS_AGGR_THRESHOLD:
            buy_qty = self.EMERALDS_AGGR_SIZE
            buy_price = min(best_ask - 1, buy_price + 1)
            if buy_price >= sell_price:
                buy_price = best_bid + 1 if best_bid + 1 < sell_price else best_bid

        buy_used = sum(o.quantity for o in orders if o.quantity > 0)
        sell_used = sum(-o.quantity for o in orders if o.quantity < 0)

        remaining_buy = max(0, max_buy_allowed - buy_used)
        remaining_sell = max(0, max_sell_allowed - sell_used)

        buy_qty = min(buy_qty, remaining_buy)
        sell_qty = min(sell_qty, remaining_sell)

        if buy_qty > 0 and not self._emerald_buy_allowed(pos, avg_long, int(buy_price), int(buy_qty)):
            buy_qty = 0

        if sell_price < self.EMERALDS_FAIR:
            sell_qty = 0

        if buy_qty > 0:
            orders.append(Order(self.EMERALDS, int(buy_price), int(buy_qty)))

        if sell_qty > 0:
            orders.append(Order(self.EMERALDS, int(sell_price), -int(sell_qty)))

        return orders

    # -------------------------------------------------
    # TOMATOES fair-value strategy
    # -------------------------------------------------
    def _trade_tomatoes(self, state: TradingState, mem: Dict[str, Any]) -> List[Order]:
        orders: List[Order] = []

        if self.TOMATOES not in state.order_depths:
            return orders

        depth = state.order_depths[self.TOMATOES]
        if not depth.buy_orders or not depth.sell_orders:
            return orders

        best_bid, best_ask = self._best_prices(depth)
        if best_bid is None or best_ask is None:
            return orders

        pos = state.position.get(self.TOMATOES, 0)
        limit = self.POSITION_LIMITS[self.TOMATOES]

        max_buy_allowed = max(0, limit - pos)
        max_sell_allowed = max(0, limit + pos)

        best_bid_vol = depth.buy_orders.get(best_bid, 0)
        best_ask_vol = -depth.sell_orders.get(best_ask, 0)
        spread = best_ask - best_bid

        instant_fair = self._instant_tom_fair_value(depth)
        if instant_fair is None:
            return orders

        rolling_fair = self._update_tom_fair(mem, instant_fair)

        # Inventory-adjusted fair: if long, lower fair; if short, raise fair
        inv_fair = rolling_fair - self.TOM_INVENTORY_SKEW * pos

        # 1) Take favorable ask/bid if it crosses fair enough
        buy_edge = inv_fair - best_ask
        sell_edge = best_bid - inv_fair

        if buy_edge >= self.TOM_TAKE_EDGE and max_buy_allowed > 0:
            extra = max(0, int(math.floor(buy_edge - self.TOM_TAKE_EDGE)))
            buy_qty = min(self.TOM_MAX_TAKE_SIZE + 5 * extra, best_ask_vol, max_buy_allowed)

            if buy_qty > 0:
                orders.append(Order(self.TOMATOES, int(best_ask), int(buy_qty)))
                max_buy_allowed -= buy_qty

        elif sell_edge >= self.TOM_TAKE_EDGE and max_sell_allowed > 0:
            extra = max(0, int(math.floor(sell_edge - self.TOM_TAKE_EDGE)))
            sell_qty = min(self.TOM_MAX_TAKE_SIZE + 5 * extra, best_bid_vol, max_sell_allowed)

            if sell_qty > 0:
                orders.append(Order(self.TOMATOES, int(best_bid), -int(sell_qty)))
                max_sell_allowed -= sell_qty

        # 2) Passive quotes around fair
        if spread >= self.TOM_MIN_SPREAD_FOR_MAKING:
            fair_floor = math.floor(inv_fair)
            fair_ceil = math.ceil(inv_fair)

            passive_bid = min(best_bid + 1, fair_floor - self.TOM_PASSIVE_EDGE)
            passive_ask = max(best_ask - 1, fair_ceil + self.TOM_PASSIVE_EDGE)

            if passive_bid < passive_ask:
                buy_qty = min(self.TOM_PASSIVE_SIZE, max_buy_allowed)
                sell_qty = min(self.TOM_PASSIVE_SIZE, max_sell_allowed)

                if pos > self.TOM_HEAVY_POS:
                    buy_qty = min(buy_qty, 4)
                elif pos < -self.TOM_HEAVY_POS:
                    sell_qty = min(sell_qty, 4)

                if buy_qty > 0 and passive_bid < inv_fair:
                    orders.append(Order(self.TOMATOES, int(passive_bid), int(buy_qty)))

                if sell_qty > 0 and passive_ask > inv_fair:
                    orders.append(Order(self.TOMATOES, int(passive_ask), -int(sell_qty)))

        return orders

    # -------------------------------------------------
    # Main entrypoint
    # -------------------------------------------------
    def run(self, state: TradingState):
        mem = self._load_state(state.traderData)

        result: Dict[str, List[Order]] = {}
        result[self.EMERALDS] = self._trade_emeralds(state, mem)
        result[self.TOMATOES] = self._trade_tomatoes(state, mem)

        traderData = self._dump_state(mem)
        conversions = 0
        return result, conversions, traderData