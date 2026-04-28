"""v10 — v3 (491870.py) + asymmetric voucher overlay layer.

VELVET logic is unchanged. Same detector, same target inventory executor,
same fair trim, same NEVER_SELL_BELOW_FAIR price protection, same adverse
stop, same late-day caps.

What's NEW: when a peak/valley signal fires, we ALSO take a parallel
position in multiple vouchers. Core strikes trade both directions; extra strikes
are PEAK-only short-call overlays so we use unused option capacity without
accidentally buying expensive/wide calls near valleys.

Default voucher pair: VEV_5000 and VEV_5100 (highest empirical correlation
to VELVET, moderate price for clean fills, full 300-unit cap usable).

Voucher position is sized as a fraction of the VELVET target. Same trim,
same adverse stop, same execution rules — just on the voucher's own book.
Each voucher has its own NEVER-trade-against-fair gate keyed off VELVET's
mid (not the voucher mid), since the voucher tracks VELVET.

Configurable voucher pair via VOUCHER_OVERLAY_STRIKES — try (4000, 4500),
(5000, 5100), or (5400, 5500) depending on your risk preference.
"""
from datamodel import Order, TradingState
from typing import Dict, List
import json
import math


class Trader:
    USE_DV          = True
    USE_MARK        = True
    USE_CHOP_FILTER = True
    USE_TRIM        = True
    USE_VOUCHER_OVERLAY = True   # set False to revert to v3 behavior
    USE_CHOP_FOLLOW  = False     # v11 disabled: body-strike vouchers have no Mark flow
    USE_HG_TRADER    = True      # v35 MERGE: SIGNAL-ONLY scanner (no executor).
                                  # Updates st['hg_mark_target']; HG_MM reads it
                                  # as offset to L2 target.  Single executor → no
                                  # dual-fight problem that lost -$159k in v34.
    USE_HG_MM        = True      # v13: HG passive MM (post bid+1/ask-1)
    USE_VEV4000_MM   = True      # v16: VEV_4000 passive MM (huge spread + Mark flow)

    # ============================================================
    UNDERLYING       = "VELVETFRUIT_EXTRACT"
    ALL_STRIKES      = (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500)
    SIGNAL_VOUCHERS  = (5000, 6000, 5100)
    CHOP_VOUCHER     = 5100

    FAIR             = 5250.0
    # === v31 FIX: dynamic VELVET fair (mirrors HG clip-range fair) ===
    # All 84 day-2 losses + 26 day-3 losses were SHORTS at +$12-30 dev.
    # Day means drift up to $5265 on some days. Static FAIR=5250 measures
    # those entries as +$15 deeper than they actually are vs local mean.
    # Dynamic fair = 5250 + clip(cum_mid - 5250, ±VELV_CLIP_RANGE).
    # v35 MERGE EXP: re-enable dynamic VELVET fair to recover eval performance.
    # Teammate disabled because it caused -$36k on training day 2.  But your
    # eval (513705 100k) showed +$5.9k from having it ON.  This hybrid keeps
    # teammate's signal-layer changes (asym override, dead-zone MM, etc.) AND
    # restores dynamic fair.  Risk: brings back the day-2 regression that
    # teammate's testing flagged.  Test all 4 days carefully before shipping.
    USE_DYNAMIC_VELV_FAIR = True       # was False (teammate); now True (your v31)
    VELV_CLIP_RANGE       = 10.0
    USE_VELV_FAIR_UNLOAD  = True       # was False (teammate); now True (your v31)
    VELV_FAIR_UNLOAD_SIZE = 60
    VELV_FAIR_UNLOAD_MIN_POS = 8

    MIN_DIST_PEAK    = 0.0   # v20: was 8 — distance gate killed ~50% of in-between extrema
    MIN_DIST_VALLEY  = 0.0   # v20: was 8 — sweep showed monotonic gain as gate→0
    POS_LIMIT        = 200
    TRADE_CAP        = 190
    COOLDOWN_TS      = 6_000  # v17: was 20k
    SIGNAL_LOG_CAP   = 800

    # ============================================================
    #   VELVET TARGET / EXECUTION  (unchanged from v3)
    # ============================================================
    TARGET_DV         = 55
    TARGET_MARK       = 75
    TARGET_DEEP       = 115
    TARGET_ADD        = 28
    DIST_SIZE_PER_TICK = 4.0
    Z_SIZE_PER_POINT   = 10.0
    TARGET_EXTRA_FAR   = 25
    FAR_SIZE_DIST      = 30.0
    MAX_TAKE_PER_TICK    = 34
    MAX_PASSIVE_PER_TICK = 85
    TAKE_FRACTION        = 0.10  # v25: was 0.50 — VELVET passive entries save spread (+$3k)
    MIN_TAKER_NEED       = 10
    TRIM_NEAR_FAIR_DIST = 1.0
    TRIM_HARD_FAIR_DIST = 10.0
    TRIM_KEEP_FRACTION  = 0.65
    NEVER_SELL_BELOW_FAIR = True
    MIN_SELL_EDGE         = 0.5
    LATE_CAP_AFTER_TS      = 900_000
    LATE_TRADE_CAP         = 95
    VERY_LATE_CAP_AFTER_TS = 960_000
    VERY_LATE_TRADE_CAP    = 55
    ADVERSE_STOP_TICKS     = 14.0
    HARD_FLATTEN_AFTER_TS  = None

    Z_THRESHOLD       = 2.96  # v19: optimal — sweep across 6 days found plateau 2.95-2.97 (+$9.7k vs 3.0)
    ESCALATION_EPS    = 0.10
    REVERSAL_DIST     = 8.0
    REGIME_TIMEOUT_TS = 50_000  # v17: was 200k
    WARMUP_SAMPLES    = 100
    MARK_EXCEPTION_Z  = 2.0
    DEEP_DIST         = 20.0
    DEEP_MIN_Z        = 0.3

    # === v21: price-deviation override ===
    # ~50% of missed extrema have price moving far but dV signal didn't fire.
    # When |S - FAIR| >= PRICE_OVERRIDE_DIST, fire a peak/valley signal even
    # without dV confirmation. Sweep on 6 days found optimum plateau 21-25.
    USE_PRICE_OVERRIDE     = True
    PRICE_OVERRIDE_DIST    = 22.0   # legacy symmetric (kept for back-compat)

    # === v31 EXP 8: ASYMMETRIC price thresholds ===
    # 6-day data: price spends 52.8% below fair (vs 46.7% above), reaches -$58 min
    # vs +$50 max, and r4_d3 has 72.5% of ticks below fair. Down-side has fatter
    # tail and more time-spent.
    # Implication: long entries should wait for DEEPER moves (price has more
    # downside room before reversing). Short entries can fire at moderate up-dev.
    # Avg current long entry: -$18 dev. Actual day lows: -$36 to -$58 dev.
    # Lifting long threshold pulls avg long entries closer to actual valleys.
    # v31 EXP 8 (revised after 100-day MC): wider on BOTH sides outperformed
    # asymmetric in MC (mean PnL $6,179 vs $4,224 — 46% better). Going with the
    # MC-validated config. This still preserves the asymmetry-aware design but
    # widens both thresholds to require meaningful price extension on either side.
    USE_ASYM_PRICE_OVERRIDE = True
    PRICE_OVERRIDE_DIST_PEAK   = 25.0   # short fires at +$25 (was +$22 → +$18 → +$25)
    PRICE_OVERRIDE_DIST_VALLEY = 35.0   # long fires at  -$35 (was -$22 → -$28 → -$35)
    DV_MIN_DIST_PEAK   = 18.0   # dV-path short: keep slightly below price-override
    DV_MIN_DIST_VALLEY = 28.0   # dV-path long:  keep slightly below price-override

    # === v24: VELVET passive MM ===
    # VELVET spread is ~$5 with 460 marks/day on 5 days. Existing executor only
    # quotes when target != cur_pos. During stale zones (e.g. ts 50K-200K on
    # r4_d3) we sit idle. Add two-sided bid+1/ask-1 quotes when main strategy
    # is converged, with inventory skew.
    USE_VELVET_MM           = True
    VELVET_MM_QUOTE_SIZE    = 6        # tuned: plateau around 5-10 quote_size
    VELVET_MM_MIN_SPREAD    = 4
    VELVET_MM_IMPROVE_TICKS = 1
    VELVET_MM_POS_SOFT      = 80
    VELVET_MM_POS_HARD      = 160
    VELVET_MM_CONVERGED_TOL = 8

    # === v31 EXP 7 LAYER A: Dead-zone VELVET MM amplification ===
    # PnL trajectory: strategy makes ~all PnL by tick 42K, gives back $1.6k after.
    # Mark 38 keeps trading throughout — we're under-engaging in the dead zone.
    # When no peak/valley signal has fired in DEAD_ZONE_LOOKBACK ticks, scale
    # VELVET MM size 3x to capture more Mark spread while main strategy idle.
    USE_DEAD_ZONE_MM        = True
    DEAD_ZONE_LOOKBACK      = 8000     # ticks since last peak/valley signal
    DEAD_ZONE_MM_SIZE_MULT  = 3        # 6 → 18 quote size during dead zone

    # === v31 EXP 7 LAYER B: Mark-driven micro-entries ===
    # When Mark sentiment is clear in dead zone (no main signal active),
    # fire small directional VELVET position. Mark 14 buys cluster at valleys
    # (long signal); Mark 38 buys cluster at peaks (short signal — fade).
    # The existing strategy uses these as multipliers; this elevates them to
    # primary signal sources during dormant periods.
    USE_MARK_MICRO_ENTRY    = True
    MARK_MICRO_TARGET       = 30       # small position size on micro signal
    MARK_MICRO_BULL_THRESH  = 2.0      # mark_bull score required
    MARK_MICRO_BEAR_THRESH  = 2.0
    MARK_MICRO_COOLDOWN_TS  = 5000     # min ticks between micro entries

    # === v26: voucher passive MM (body strikes) ===
    # User insight: keep aggressive taker on entry, ALSO leave a passive bid/ask
    # alongside to scoop up opportunistic flow at better prices.
    USE_VOUCHER_MM            = True
    VOUCHER_MM_STRIKES        = (5400, 5500)  # body strikes have ~0 flow; 5400/5500 have 80-90 marks/day
    VOUCHER_MM_QUOTE_SIZE     = 4
    VOUCHER_MM_MIN_SPREAD     = 1   # spread is tight here; need to allow it
    VOUCHER_MM_IMPROVE_TICKS  = 1
    VOUCHER_MM_POS_SOFT       = 100
    VOUCHER_MM_POS_HARD       = 200
    VOUCHER_MM_CONVERGED_TOL  = 5

    # === v27: Order book imbalance signal (#3) ===
    # When bid_volume >> ask_volume, sellers are exhausting — small long bias
    # Apply as a tiny overlay only when main target is near zero
    USE_OB_IMBALANCE          = True
    OB_IMBALANCE_THRESH       = 0.10       # 99th-pct threshold for VELVET (book is always balanced)
    OB_IMBALANCE_TARGET       = 20         # small VELVET target
    OB_IMBALANCE_VOUCHER      = 25         # small voucher target
    OB_IMBALANCE_MAX_MAIN     = 50         # loosened: was 30
    OB_IMBALANCE_COOLDOWN_TS  = 3_000

    # === v27: Mark counterparty COMBINATIONS (#2) ===
    # DISABLED — backtest showed -$69k regression (noise on r4_d1/r4_d2).
    USE_MARK_COMBO            = False
    MARK_COMBO_INFORMED       = ("Mark 14", "Mark 01", "Mark 67")
    MARK_COMBO_UNINFORMED     = ("Mark 22",)
    MARK_COMBO_VELVET_TARGET  = 80        # bigger size on combo confirmation
    MARK_COMBO_VOUCHER_TARGET = 100
    MARK_COMBO_COOLDOWN_TS    = 5_000

    # === v31 EXPERIMENT 6: IV smile arb implementation ===
    # v30 author left flag without code. We implement minimal additive layer:
    #   When a body voucher's |z| >= IV_SMILE_ARB_Z, trade THAT voucher directly.
    #   - z signed positive → voucher OVERPRICED vs smile → SHORT it
    #   - z signed negative → voucher UNDERPRICED → LONG it
    #   Conservative: only nudge target if existing target is in same direction
    #   or zero. Never fight the main strategy.
    #   Only on body strikes (5000, 5100) where signal is cleanest.
    USE_IV_SMILE_ARB          = True
    IV_SMILE_ARB_Z            = 3.5      # higher threshold than 2.96 main signal
    IV_SMILE_ARB_TARGET       = 40       # small additive size
    IV_SMILE_ARB_STRIKES      = (5000, 5100)   # body only

    INFORMED_BUYERS   = ("Mark 01", "Mark 14", "Mark 67")
    INFORMED_SELLERS  = ("Mark 14", "Mark 01")
    FADE_SELLERS      = ("Mark 22",)
    MARK_PRODUCTS     = ("VELVETFRUIT_EXTRACT",
                         "VEV_5000", "VEV_5400", "VEV_5500",
                         "VEV_6000", "VEV_6500")
    CHOP_Z_THRESH       = 2.5
    CHOP_LOOKBACK_TS    = 10_000
    CHOP_OVERRIDE_DIST  = 20.0
    CHOP_DETECT_ZONE    = 12.0

    # ============================================================
    #   VOUCHER OVERLAY (v6 addition)
    # ============================================================
    # Best empirical correlation to VELVET across 6 days is 5000/5100 (0.76).
    # Try (4000, 4500) for the literal "expensive most correlated" interpretation
    # or (5400, 5500) for a convex (cheap, leveraged) version.
    # v10: asymmetric overlay.
    # Core strikes are the proven liquid/high-correlation layer.
    # Extra peak-only strikes add capacity when the detector says "short calls";
    # on valley signals their targets go to 0, so they cover but do not flip long.
    # v31 EXPERIMENT 1 REVERTED: VEV_5500 has corr=0.33 with VELVET (vs 0.52-0.76
    # for body strikes). Backtest showed 27 wins / 63 losses, net -$20. Reverted.
    #
    # v31 EXPERIMENT 2 SHIPPED (+$15,093 / +9.4% on day 3):
    #   VEV_4500 (corr 0.60) and VEV_5300 (corr 0.62) moved to bidirectional CORE.
    #   4500: $5,168 → $14,002 (+$8,834).  5300: $5,114 → $11,373 (+$6,259).
    #
    # v31 EXPERIMENT 3 REVERTED: VEV_4000 bidirectional was net-mixed.
    #   Day 3: +$2,529 (works on big-drift days)
    #   Day 2: -$2,988 (loses on chop days)
    #   High-delta strike fights mean-reversion when going long at moderate dist.
    #   Reverted back to peak-only. KEEP 4500 and 5300 — those are clean wins.
    VOUCHER_CORE_STRIKES      = (4500, 5000, 5100, 5200, 5300)
    VOUCHER_PEAK_EXTRA_STRIKES= (4000, 5400)
    VOUCHER_OVERLAY_STRIKES   = VOUCHER_CORE_STRIKES + VOUCHER_PEAK_EXTRA_STRIKES

    VOUCHER_POS_LIMIT       = 300
    VOUCHER_TRADE_CAP       = 280

    VOUCHER_BASE_TARGET     = 90
    VOUCHER_DIST_PER_TICK   = 5.0
    VOUCHER_TARGET_FAR_BONUS= 50
    VOUCHER_MARK_FRACTION   = 0.75

    # v31 EXPERIMENT 5 REVERTED: body sizing 90→115 hurt body strike PnL by
    # $1-2k/day. Body strikes were already at near-optimal sizing — bigger
    # targets didn't translate to better fills, just more spread cost.
    VOUCHER_BASE_TARGET_PER_STRIKE = {}   # empty = all strikes use VOUCHER_BASE_TARGET (90)

    # === v31 EXP 9: Wing dampener on deep entries ===
    # After wider thresholds (25/35) shipped, day-2 VEV_4500 losses jumped
    # 15 → 48 (220% increase). Wings have higher std-ratio (1.1+) but lower
    # correlation (0.60-0.62 vs body 0.72-0.76). On deep VELVET signals,
    # wings overshoot AND eat bigger MTM hits when price keeps extending.
    # Trim wing targets to 70% on signals where dist >= DEEP_DIST_THRESHOLD.
    USE_WING_DEEP_DAMPENER     = True
    WING_DEEP_DAMPENER_STRIKES = (4500, 5300)
    DEEP_DIST_THRESHOLD        = 30.0
    WING_DEEP_DAMPENER_FACTOR  = 0.70

    # v31 EXPERIMENT 4 REVERTED: Z-scaling combined with body sizing didn't
    # help on body strikes (slight negative). Body signals were already
    # well-sized; magnitude scaling pushed past the optimal point.
    USE_Z_SCALING        = False
    Z_SCALE_FACTOR       = 0.30
    Z_SCALE_CAP          = 1.60
    Z_SCALE_STRIKES      = (5000, 5100, 5200)

    # Peak-only extra caps by strike. 4000/4500 are wide but high-delta;
    # 5300/5400 are lower-dollar convex shorts. Keep all conservative.
    PEAK_EXTRA_TARGET_CAP = {
        4000: 70,
        4500: 95,
        5300: 120,
        5400: 90,
    }
    PEAK_EXTRA_MIN_DIST = 22.0  # v17: was 30
    PEAK_EXTRA_MIN_Z    = 2.80  # v17: was 3.20

    # Voucher execution
    VOUCHER_MAX_TAKE_PER_TICK    = 25
    VOUCHER_MAX_PASSIVE_PER_TICK = 60
    VOUCHER_TAKE_FRACTION        = 1.0  # v25: was 0.50 — vouchers reverse fast on signal, passive misses fills (+$6k)

    # Trim & adverse stop: keyed off VELVET's S vs FAIR (same triggers as VELVET).
    VOUCHER_TRIM_NEAR_FAIR_DIST = 1.0
    VOUCHER_TRIM_HARD_FAIR_DIST = 10.0
    VOUCHER_TRIM_KEEP_FRACTION  = 0.65

    # ============================================================
    #   CHOP-FOLLOW MICRO-LAYER (v11)
    # ============================================================
    # 43% of all ticks live in the chop zone (|S-FAIR| < 10) where the dV
    # signal is too noisy to trigger our main detector. But the Marks DON'T
    # change behavior in chop — they keep doing what they do. So we can use
    # *them* directly as the trigger here.
    #
    # When the price is in chop AND main target is 0:
    #   informed buyer (Mark 01/14/67) buys voucher → small same-direction long
    #   uninformed seller (Mark 22) sells voucher    → small fade-long
    #   informed seller (Mark 14/01) sells voucher   → small short
    # All chop trades go to a separate `chop_target_pos` per voucher; when we
    # leave the chop zone, those targets reset to 0.
    CHOP_FOLLOW_ZONE       = 8.0      # |S-FAIR| < this triggers chop layer
    CHOP_FOLLOW_VOUCHERS   = (5000, 5100, 5200, 5400, 5500)  # body strikes, liquid
    CHOP_FOLLOW_BASE_SIZE  = 4        # per-fire size
    CHOP_FOLLOW_CAP        = 25       # |chop_target_pos| cap per voucher
    CHOP_FOLLOW_INFORMED_BUYERS  = ("Mark 01", "Mark 14", "Mark 67")
    CHOP_FOLLOW_INFORMED_SELLERS = ("Mark 14", "Mark 01")
    CHOP_FOLLOW_FADE_SELLERS     = ("Mark 22",)

    # ============================================================
    #   HYDROGEL_PACK MARK-PAIR TRADER (v12)
    # ============================================================
    # Empirical (5 days, 1696 HG Mark trades):
    #   Mark 14: t=+34.6, +8.15 bps mean fwd return on HG mid     -> FOLLOW
    #   Mark 38: t=-34.0, -8.01 bps                                -> FADE
    #   They face each other 98% of the time (Mark 38 vs Mark 14).
    #   HG/VELVET correlation = 0.005 -> independent edge, zero interference.
    # Sizing:
    #   $9991 mid * 8 bps = $80 fwd edge per share
    #   Spread $15.7 -> ~$64 net round-trip after spread cost
    #   At size 6 per Mark 14 fire, expected $384/round-trip
    # === v35 MERGE: HG follower is signal-only, smaller size, slower decay ===
    # Mark 14's edge is ~+2 bps over 5000+ ticks (small).  Smaller size and slower
    # decay let the offset express through MM's passive quoting without paying
    # spread.  Mark 38 fade disabled (just Mark 14's mirror counterparty).
    HG_PRODUCT             = "HYDROGEL_PACK"
    HG_FOLLOW_BUYERS       = ("Mark 14",)
    HG_FOLLOW_SELLERS      = ("Mark 14",)
    HG_FADE_BUYERS         = ()                  # disabled
    HG_FADE_SELLERS        = ()                  # disabled
    HG_FOLLOW_SIZE         = 3                   # was 6 — small per-event nudge
    HG_FADE_SIZE           = 0                   # disabled
    HG_FOLLOW_MIN_QTY      = 5                   # NEW: skip noise trades (qty<5)
    HG_POS_LIMIT           = 200
    HG_TARGET_CAP          = 30                  # was 60 — offset cap, low conviction
    HG_DECAY_TS            = 80_000              # was 50k — hold longer, signal is long-horizon
    HG_DECAY_FRAC          = 0.92                # was 0.85 — slower decay
    HG_MAX_TAKE_PER_TICK   = 25                  # legacy, unused (no executor)
    HG_MAX_PASSIVE_PER_TICK= 50                  # legacy
    HG_TAKE_FRACTION       = 0.50                # legacy

    # === v13: HG passive MM ===
    HG_MM_QUOTE_SIZE       = 12       # v28: was 8 — pushes more flow capture (+$14k)
    HG_MM_POS_SOFT         = 100      # v28: was 50 — wider inventory band (+$12k)
    HG_MM_POS_HARD         = 180      # v28: was 120 — looser hard cap
    HG_MM_MIN_SPREAD       = 4        # require at least this spread
    HG_MM_IMPROVE_TICKS    = 1        # post bid+N / ask-N

    # === v14: DIRECTIONAL HG MM ===
    # Empirical HG fair: $9994. AR(1) κ=0.002118, half-life 327 ticks.
    # Skew inventory toward predicted reversion: long when HG below fair,
    # short when above. Target inventory = -K × (mid - fair), capped.
    HG_FAIR                = 9994.0
    HG_TILT_K              = 1.5      # shares per $ deviation (target inv = -K*Δ)
    HG_TILT_CAP            = 80       # |target inventory| cap from tilt
    HG_TILT_BID_BOOST      = 6        # extra bid size when seeking more long
    HG_TILT_ASK_BOOST      = 6        # extra ask size when seeking more short

    # === v15: aggressive take at extreme deviations ===
    # Realized-trade data (271 HG trades on day 3):
    #   Shorts at +50 dev: avg +$119/trade, n=31  (huge edge)
    #   Shorts at +30..+50: avg +$49/trade, n=74
    #   Longs at -30..-50: avg +$63/trade, n=8
    # When passive MM doesn't catch a fill, cross the spread to guarantee one.
    HG_TAKE_THRESHOLD      = 30       # v28: was 25 — slightly higher = more reliable signal (+$8k)
    HG_TAKE_SIZE           = 8
    HG_TAKE_CAP            = 200      # v28: was 100 — biggest single win (+$50k); ride mean reversion harder

    # === v16: VEV_4000 passive MM ===
    # Spread $20.78 avg, Mark flow 1,529 units/day, parity gap ~0 (deep ITM,
    # mid ≈ S - 4000). Independent of VELVET signal logic — pure spread capture.
    # Currently v15 only does directional peak-only on VEV_4000 (31 trades/day).
    # Adding passive MM gives 200-300 more trades/day at avg $15/RT.
    VEV4000_MM_QUOTE_SIZE     = 6       # base quote size each side
    VEV4000_MM_MIN_SPREAD     = 5       # require this much spread before quoting
    VEV4000_MM_IMPROVE_TICKS  = 1       # post bid+1 / ask-1
    VEV4000_MM_POS_SOFT       = 40      # soft cap; skew quotes when over
    VEV4000_MM_POS_HARD       = 100     # hard cap; one-sided beyond this

    # ============================================================
    T_OPT  = 7.0 / 365.0
    R      = 0.0
    MIN_IV = 1e-4
    MAX_IV = 5.0

    # ------------------------------------------------------------
    def _fresh(self):
        return {
            'last_ts': None, 'welford': {},
            'last_peak_signal_ts': -10**9, 'last_valley_signal_ts': -10**9,
            'peak_anchor_z': self.Z_THRESHOLD, 'valley_anchor_z': self.Z_THRESHOLD,
            'last_peak_S': None, 'last_valley_S': None,
            'last_chop_ts': -10**9,
            'target_pos': 0,
            'voucher_target_pos': {},  # str(K) -> int
            'chop_target_pos': {},     # v11: separate chop-layer target per voucher str(K)
            'hg_target_pos': 0,        # v12: HYDROGEL_PACK target position
            'hg_last_signal_ts': -10**9,
            'last_ob_imb_ts': -10**9,    # v27 OB imbalance cooldown
            'last_mark_combo_ts': -10**9,
            'last_micro_entry_ts': -10**9,  # v31 EXP 7B Mark micro-entry cooldown
            'ob_imb_log': [], 'mark_combo_log': [],
            'last_signal_side': None, 'last_signal_src': None,
            'last_target_update_ts': -10**9,
            'last_registered_signal_ts': -10**9,
            'last_risk_action_ts': -10**9,
            'signals_log': [], 'chop_log': [], 'target_log': [],
        }

    def _load(self, td):
        if not td: return self._fresh()
        try:
            st = json.loads(td)
            for k, v in self._fresh().items(): st.setdefault(k, v)
            return st
        except Exception: return self._fresh()

    def _save(self, st):
        for key in ('signals_log', 'chop_log', 'target_log'):
            if len(st.get(key, [])) > self.SIGNAL_LOG_CAP:
                st[key] = st[key][-self.SIGNAL_LOG_CAP:]
        return json.dumps(st, separators=(",", ":"))

    def _reset_day(self, st, ts):
        last = st.get('last_ts')
        if last is not None and ts < last:
            fresh = self._fresh(); st.clear(); st.update(fresh)
        st['last_ts'] = ts

    def _welford_update(self, st, K, x):
        w = st['welford'].setdefault(str(K), {'n': 0, 'mean': 0.0, 'M2': 0.0})
        w['n'] += 1
        d = x - w['mean']; w['mean'] += d / w['n']
        d2 = x - w['mean']; w['M2'] += d * d2

    def _z(self, st, K, x):
        w = st['welford'].get(str(K))
        if w is None or w['n'] < self.WARMUP_SAMPLES: return 0.0
        var = w['M2'] / max(1, w['n'] - 1)
        if var <= 0: return 0.0
        return (x - w['mean']) / math.sqrt(var)

    def _ncdf(self, x): return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    def _bs(self, S, K, sigma):
        if sigma <= 0 or self.T_OPT <= 0: return max(S - K, 0)
        sq = math.sqrt(self.T_OPT)
        d1 = (math.log(S/K) + 0.5*sigma*sigma*self.T_OPT) / (sigma*sq)
        d2 = d1 - sigma*sq
        return S*self._ncdf(d1) - K*math.exp(-self.R*self.T_OPT)*self._ncdf(d2)
    def _iv(self, S, K, price):
        intrinsic = max(S - K*math.exp(-self.R*self.T_OPT), 0)
        if price <= intrinsic + 1e-12: return self.MIN_IV
        lo, hi = self.MIN_IV, self.MAX_IV
        if self._bs(S, K, hi) < price: return hi
        for _ in range(28):
            m = 0.5*(lo+hi)
            if self._bs(S, K, m) < price: lo = m
            else: hi = m
        return max(self.MIN_IV, min(self.MAX_IV, 0.5*(lo+hi)))
    def _solve3(self, A, b):
        M = [A[0][:]+[b[0]], A[1][:]+[b[1]], A[2][:]+[b[2]]]
        for c in range(3):
            p = c
            for r in range(c+1, 3):
                if abs(M[r][c]) > abs(M[p][c]): p = r
            if abs(M[p][c]) < 1e-12: return None
            if p != c: M[c], M[p] = M[p], M[c]
            d = M[c][c]
            for j in range(c, 4): M[c][j] /= d
            for r in range(3):
                if r == c: continue
                f = M[r][c]
                for j in range(c, 4): M[r][j] -= f * M[c][j]
        return [M[0][3], M[1][3], M[2][3]]
    def _fit(self, pts, exclude_K=None):
        use = [p for p in pts if exclude_K is None or p['K'] != exclude_K]
        if len(use) < 3: use = pts[:]
        if len(use) < 3:
            avg = sum(p['iv'] for p in use)/max(1, len(use)); return 0.0, 0.0, avg
        s0=s1=s2=s3=s4=y0=y1=y2=0.0
        for p in use:
            x=p['x']; y=p['iv']; w=1.0/max(p['spread'], 1.0); x2=x*x
            s0+=w; s1+=w*x; s2+=w*x2; s3+=w*x2*x; s4+=w*x2*x2
            y0+=w*y; y1+=w*x*y; y2+=w*x2*y
        sol = self._solve3([[s4,s3,s2],[s3,s2,s1],[s2,s1,s0]], [y2,y1,y0])
        if sol is None: return 0.0, 0.0, y0/max(s0, 1e-12)
        return sol[0], sol[1], sol[2]

    def _bbb(self, od):
        if od is None or not od.buy_orders or not od.sell_orders: return None, None
        return max(od.buy_orders), min(od.sell_orders)
    def _add(self, r, sym, px, qty):
        if qty == 0: return
        r.setdefault(sym, []).append(Order(sym, int(round(px)), int(qty)))
    def _clip(self, x, lo, hi): return max(lo, min(hi, x))

    def _gate_peak(self, st, ts, S, cur_pos):
        if S < self.FAIR + self.MIN_DIST_PEAK: return False
        if ts - st['last_peak_signal_ts'] < self.COOLDOWN_TS: return False
        if cur_pos <= -self.POS_LIMIT: return False
        return True
    def _gate_valley(self, st, ts, S, cur_pos):
        if S > self.FAIR - self.MIN_DIST_VALLEY: return False
        if ts - st['last_valley_signal_ts'] < self.COOLDOWN_TS: return False
        if cur_pos >= self.POS_LIMIT: return False
        return True

    def _is_chop(self, st, ts, S):
        if not self.USE_CHOP_FILTER: return False
        if abs(S - self.FAIR) >= self.CHOP_OVERRIDE_DIST: return False
        return ts - st.get('last_chop_ts', -10**9) <= self.CHOP_LOOKBACK_TS

    def _maybe_drift_reset_peak(self, st, ts, S):
        last = st.get('last_peak_S')
        if last is not None and S < last - self.REVERSAL_DIST:
            st['peak_anchor_z'] = self.Z_THRESHOLD; st['last_peak_S'] = None; return
        if ts - st['last_peak_signal_ts'] > self.REGIME_TIMEOUT_TS:
            if st['peak_anchor_z'] > self.Z_THRESHOLD:
                st['peak_anchor_z'] = self.Z_THRESHOLD
    def _maybe_drift_reset_valley(self, st, ts, S):
        last = st.get('last_valley_S')
        if last is not None and S > last + self.REVERSAL_DIST:
            st['valley_anchor_z'] = self.Z_THRESHOLD; st['last_valley_S'] = None; return
        if ts - st['last_valley_signal_ts'] > self.REGIME_TIMEOUT_TS:
            if st['valley_anchor_z'] > self.Z_THRESHOLD:
                st['valley_anchor_z'] = self.Z_THRESHOLD

    # ------------------------------------------------------------
    # VELVET target sizing (unchanged from v3)
    # ------------------------------------------------------------
    def _base_target_abs(self, side, src, S, z):
        if src == 'mark_deep':   base = self.TARGET_DEEP
        elif src == 'mark':      base = self.TARGET_MARK
        else:                    base = self.TARGET_DV
        min_dist = self.MIN_DIST_PEAK if side == 'peak' else self.MIN_DIST_VALLEY
        dist = abs(S - self.FAIR)
        dist_extra = max(0.0, dist - min_dist) * self.DIST_SIZE_PER_TICK
        z_extra = max(0.0, float(z) - self.Z_THRESHOLD) * self.Z_SIZE_PER_POINT
        base = base + dist_extra + z_extra
        if dist >= self.FAR_SIZE_DIST:
            base += self.TARGET_EXTRA_FAR
        if side == 'peak' and dist < 22.0:
            base *= 0.78
        return int(min(self.TRADE_CAP, max(0, round(base))))

    # ------------------------------------------------------------
    # NEW: voucher target sizing
    # ------------------------------------------------------------
    def _voucher_target_abs(self, K, side, src, S, z):
        """Target voucher position size for a given signal.

        Core strikes trade both peak shorts and valley longs.
        Extra strikes are peak-only: they expand capacity on strong tops,
        then get covered on valley signals instead of flipping long.
        """
        dist = abs(S - self.FAIR)

        # Extra strikes only participate on strong PEAK shorts.
        if K in self.VOUCHER_PEAK_EXTRA_STRIKES:
            if side != 'peak':
                return 0
            if dist < self.PEAK_EXTRA_MIN_DIST and z < self.PEAK_EXTRA_MIN_Z:
                return 0

            cap = int(self.PEAK_EXTRA_TARGET_CAP.get(K, 80))
            # Scale gently with distance/z, but respect conservative strike cap.
            strength = 0.65
            strength += 0.015 * max(0.0, dist - self.PEAK_EXTRA_MIN_DIST)
            strength += 0.08 * max(0.0, float(z) - self.PEAK_EXTRA_MIN_Z)
            strength = max(0.0, min(1.0, strength))
            if src in ('mark', 'mark_deep'):
                strength *= self.VOUCHER_MARK_FRACTION
            return int(min(cap, max(0, round(cap * strength))))

        # Core strikes: original behavior with v31 per-strike base override.
        base = self.VOUCHER_BASE_TARGET_PER_STRIKE.get(K, self.VOUCHER_BASE_TARGET)
        min_dist = self.MIN_DIST_PEAK if side == 'peak' else self.MIN_DIST_VALLEY
        dist_extra = max(0.0, dist - min_dist) * self.VOUCHER_DIST_PER_TICK
        base = base + dist_extra
        if dist >= self.FAR_SIZE_DIST:
            base += self.VOUCHER_TARGET_FAR_BONUS
        if src in ('mark', 'mark_deep'):
            base *= self.VOUCHER_MARK_FRACTION

        # v31 EXPERIMENT 4: Z-magnitude scaling on highest-correlation strikes.
        # High-conviction signals (Z well past threshold) get larger position.
        if self.USE_Z_SCALING and K in self.Z_SCALE_STRIKES and z is not None:
            try:
                zf = float(z)
                z_excess = max(0.0, zf - self.Z_THRESHOLD)
                z_mult = min(self.Z_SCALE_CAP, 1.0 + self.Z_SCALE_FACTOR * z_excess)
                base *= z_mult
            except (TypeError, ValueError):
                pass

        # v31 EXPERIMENT 9: Wing dampener on deep entries.
        # Wings overshoot on extreme VELVET signals — trim their target by 30%.
        if (self.USE_WING_DEEP_DAMPENER
                and K in self.WING_DEEP_DAMPENER_STRIKES
                and dist >= self.DEEP_DIST_THRESHOLD):
            base *= self.WING_DEEP_DAMPENER_FACTOR

        return int(min(self.VOUCHER_TRADE_CAP, max(0, round(base))))

    def _register_signal(self, st, ts, side, src, S, z, z5100, price, extra=None):
        if st.get('last_registered_signal_ts') == ts: return False
        sign = -1 if side == 'peak' else 1
        old_target = int(st.get('target_pos', 0))
        base_abs = self._base_target_abs(side, src, S, z)
        day_ts = ts % 1_000_000
        if day_ts >= self.VERY_LATE_CAP_AFTER_TS:
            base_abs = min(base_abs, self.VERY_LATE_TRADE_CAP)
        elif day_ts >= self.LATE_CAP_AFTER_TS:
            base_abs = min(base_abs, self.LATE_TRADE_CAP)
        if old_target * sign > 0:
            new_abs = min(self.TRADE_CAP, abs(old_target) + self.TARGET_ADD)
            new_abs = max(new_abs, base_abs)
        else:
            new_abs = base_abs
        new_target = int(self._clip(sign * int(new_abs), -self.TRADE_CAP, self.TRADE_CAP))
        st['target_pos'] = new_target
        st['last_signal_side'] = side
        st['last_signal_src'] = src
        st['last_target_update_ts'] = ts
        st['last_registered_signal_ts'] = ts

        # Set voucher targets in same direction
        if self.USE_VOUCHER_OVERLAY:
            for K in self.VOUCHER_OVERLAY_STRIKES:
                v_abs = self._voucher_target_abs(K, side, src, S, z)
                old_v = int(st['voucher_target_pos'].get(str(K), 0))
                if old_v * sign > 0:  # extending same side
                    new_v_abs = min(self.VOUCHER_TRADE_CAP, abs(old_v) + max(20, v_abs // 3))
                    new_v_abs = max(new_v_abs, v_abs)
                else:
                    new_v_abs = v_abs
                new_v = int(self._clip(sign * int(new_v_abs), -self.VOUCHER_TRADE_CAP, self.VOUCHER_TRADE_CAP))
                st['voucher_target_pos'][str(K)] = new_v

        rec = {'ts': ts, 'kind': side, 'src': src,
               'z': round(z, 2), 'z5100': round(z5100, 2),
               'S': round(S, 1), 'price': price,
               'old_target': old_target, 'new_target': new_target}
        if extra: rec.update(extra)
        st['signals_log'].append(rec)
        st['target_log'].append({'ts': ts, 'S': round(S, 1), 'target': new_target, 'src': src, 'kind': side})
        return True

    def _apply_fair_trim(self, st, ts, S):
        if not self.USE_TRIM: return
        # VELVET trim
        target = int(st.get('target_pos', 0))
        if target != 0:
            new_target = target
            if target < 0:
                if S <= self.FAIR - self.TRIM_HARD_FAIR_DIST:    new_target = 0
                elif S <= self.FAIR - self.TRIM_NEAR_FAIR_DIST:  new_target = int(round(target * self.TRIM_KEEP_FRACTION))
            else:
                if S >= self.FAIR + self.TRIM_HARD_FAIR_DIST:    new_target = 0
                elif S >= self.FAIR + self.TRIM_NEAR_FAIR_DIST:  new_target = int(round(target * self.TRIM_KEEP_FRACTION))
            if new_target != target:
                st['target_pos'] = int(self._clip(new_target, -self.TRADE_CAP, self.TRADE_CAP))
                st['target_log'].append({'ts': ts, 'S': round(S, 1), 'target': st['target_pos'], 'src': 'trim', 'kind': 'fair_trim'})
        # Voucher trim — same triggers, keyed off VELVET S
        if self.USE_VOUCHER_OVERLAY:
            for K in self.VOUCHER_OVERLAY_STRIKES:
                v_target = int(st['voucher_target_pos'].get(str(K), 0))
                if v_target == 0: continue
                new_v = v_target
                if v_target < 0:
                    if S <= self.FAIR - self.VOUCHER_TRIM_HARD_FAIR_DIST:    new_v = 0
                    elif S <= self.FAIR - self.VOUCHER_TRIM_NEAR_FAIR_DIST:  new_v = int(round(v_target * self.VOUCHER_TRIM_KEEP_FRACTION))
                else:
                    if S >= self.FAIR + self.VOUCHER_TRIM_HARD_FAIR_DIST:    new_v = 0
                    elif S >= self.FAIR + self.VOUCHER_TRIM_NEAR_FAIR_DIST:  new_v = int(round(v_target * self.VOUCHER_TRIM_KEEP_FRACTION))
                if new_v != v_target:
                    st['voucher_target_pos'][str(K)] = int(self._clip(new_v, -self.VOUCHER_TRADE_CAP, self.VOUCHER_TRADE_CAP))

    def _apply_late_cap(self, st, ts, S):
        target = int(st.get('target_pos', 0))
        if target == 0: return
        day_ts = ts % 1_000_000
        cap = None
        if day_ts >= self.VERY_LATE_CAP_AFTER_TS: cap = self.VERY_LATE_TRADE_CAP
        elif day_ts >= self.LATE_CAP_AFTER_TS:    cap = self.LATE_TRADE_CAP
        if cap is None or abs(target) <= cap: return
        new_target = cap if target > 0 else -cap
        st['target_pos'] = int(new_target)
        st['target_log'].append({'ts': ts, 'S': round(S, 1), 'target': int(new_target), 'src': 'risk', 'kind': 'late_cap'})
        # Also cap vouchers proportionally
        if self.USE_VOUCHER_OVERLAY:
            for K in self.VOUCHER_OVERLAY_STRIKES:
                vt = int(st['voucher_target_pos'].get(str(K), 0))
                if vt > cap*2:    st['voucher_target_pos'][str(K)] = cap*2
                elif vt < -cap*2: st['voucher_target_pos'][str(K)] = -cap*2

    def _apply_adverse_stop(self, st, ts, S):
        target = int(st.get('target_pos', 0))
        if target == 0: return
        if target < 0:
            ref = st.get('last_peak_S')
            if ref is not None and S >= float(ref) + self.ADVERSE_STOP_TICKS:
                st['target_pos'] = 0
                if self.USE_VOUCHER_OVERLAY:
                    for K in self.VOUCHER_OVERLAY_STRIKES:
                        st['voucher_target_pos'][str(K)] = 0
                st['peak_anchor_z'] = self.Z_THRESHOLD
                st['last_peak_S'] = None
                st['target_log'].append({'ts': ts, 'S': round(S, 1), 'target': 0, 'src': 'risk', 'kind': 'adverse_stop_short'})
        elif target > 0:
            ref = st.get('last_valley_S')
            if ref is not None and S <= float(ref) - self.ADVERSE_STOP_TICKS:
                st['target_pos'] = 0
                if self.USE_VOUCHER_OVERLAY:
                    for K in self.VOUCHER_OVERLAY_STRIKES:
                        st['voucher_target_pos'][str(K)] = 0
                st['valley_anchor_z'] = self.Z_THRESHOLD
                st['last_valley_S'] = None
                st['target_log'].append({'ts': ts, 'S': round(S, 1), 'target': 0, 'src': 'risk', 'kind': 'adverse_stop_long'})

    # ------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------
    def _exec_one_product(self, result, state, st, product, target,
                           pos_limit, max_take, max_passive, take_frac, S):
        """Generic executor: take some at touch, post passive for rest.
        For VELVET, applies NEVER_SELL_BELOW_FAIR.
        For vouchers, applies the same gate keyed off VELVET S (passed in).
        """
        od = state.order_depths.get(product)
        if od is None: return
        u_bb, u_ba = self._bbb(od)
        if u_bb is None or u_ba is None: return
        pos = int(state.position.get(product, 0))
        target = int(self._clip(target, -pos_limit, pos_limit))

        need = target - pos
        if need == 0: return
        if need > 0: need = min(need, pos_limit - pos)
        else:        need = max(need, -pos_limit - pos)
        if need == 0: return

        abs_need = abs(need)
        take_qty = 0
        if abs_need >= self.MIN_TAKER_NEED:
            take_qty = min(max_take, max(1, int(abs_need * take_frac)))

        if need > 0:
            visible_ask = -int(od.sell_orders.get(u_ba, 0))
            if take_qty > 0 and visible_ask > 0:
                q = min(take_qty, visible_ask, need)
                self._add(result, product, u_ba, q); need -= q
            if need > 0:
                q = min(max_passive, need)
                px = u_bb + 1 if u_bb + 1 < u_ba else u_bb
                self._add(result, product, px, q)
        else:
            sell_need = -need
            # Price-protected sell: never sell when VELVET S is below fair
            # (signals that voucher should be cheap, not expensive — bad timing)
            allow_sell = True
            if self.NEVER_SELL_BELOW_FAIR and S < self.FAIR + self.MIN_SELL_EDGE:
                # For VELVET: don't sell at all below fair
                # For vouchers: don't sell either when VELVET is below fair
                # (the voucher tracks VELVET, so vouchers are 'cheap' here too)
                allow_sell = False

            visible_bid = int(od.buy_orders.get(u_bb, 0))
            if allow_sell and take_qty > 0 and visible_bid > 0:
                q = min(take_qty, visible_bid, sell_need)
                self._add(result, product, u_bb, -q); sell_need -= q

            if allow_sell and sell_need > 0:
                q = min(max_passive, sell_need)
                px = u_ba - 1 if u_ba - 1 > u_bb else u_ba
                if px > u_bb:
                    self._add(result, product, px, -q)

    def _velvet_passive_mm(self, result, state, st, S, in_chop):
        """Two-sided MM on VELVETFRUIT_EXTRACT.

        Quotes bid+1/ask-1 when:
          - main strategy is converged (|target-pos| <= tolerance)
          - spread is wide enough to be profitable
          - not in chop
        Skews quote sizes by inventory.
        """
        if in_chop: return
        prod = self.UNDERLYING
        od = state.order_depths.get(prod)
        if od is None: return
        u_bb, u_ba = self._bbb(od)
        if u_bb is None or u_ba is None: return
        spread = u_ba - u_bb
        if spread < self.VELVET_MM_MIN_SPREAD: return

        pos = int(state.position.get(prod, 0))
        target = int(st.get('target_pos', 0))
        # Only quote when main strategy is converged — don't fight active signal.
        if abs(target - pos) > self.VELVET_MM_CONVERGED_TOL: return

        # === LAYER A: Dead-zone size amplification ===
        # When no peak/valley signal has fired recently, the main strategy is
        # dormant — scale up MM size to capture more Mark spread.
        ts = int(state.timestamp)
        last_peak  = int(st.get('last_peak_signal_ts', -10**9))
        last_valley= int(st.get('last_valley_signal_ts', -10**9))
        last_signal= max(last_peak, last_valley)
        is_dead_zone = (
            self.USE_DEAD_ZONE_MM
            and abs(target) < 50
            and (ts - last_signal) >= self.DEAD_ZONE_LOOKBACK
        )
        size_mult = self.DEAD_ZONE_MM_SIZE_MULT if is_dead_zone else 1
        bid_size = self.VELVET_MM_QUOTE_SIZE * size_mult
        ask_size = self.VELVET_MM_QUOTE_SIZE * size_mult

        # Inventory skew
        if pos > self.VELVET_MM_POS_HARD:
            bid_size = 0; ask_size *= 2
        elif pos < -self.VELVET_MM_POS_HARD:
            bid_size *= 2; ask_size = 0
        elif pos > self.VELVET_MM_POS_SOFT:
            bid_size = max(1, bid_size // 2); ask_size = bid_size + 4
        elif pos < -self.VELVET_MM_POS_SOFT:
            ask_size = max(1, ask_size // 2); bid_size = ask_size + 4

        # Position-limit guards
        room_long = self.POS_LIMIT - pos
        room_short = self.POS_LIMIT + pos
        bid_size = max(0, min(bid_size, room_long))
        ask_size = max(0, min(ask_size, room_short))

        # Don't post asks below FAIR (price-protection: same as main exec logic)
        # Sell-protect: if S < FAIR + MIN_SELL_EDGE, don't post an ask at all.
        if self.NEVER_SELL_BELOW_FAIR and S < self.FAIR + self.MIN_SELL_EDGE:
            ask_size = 0
        # Buy-protect (symmetric): if S > FAIR by big amount, don't keep buying
        # Use a conservative gate
        if S > self.FAIR + 30:
            bid_size = 0

        bid_px = u_bb + self.VELVET_MM_IMPROVE_TICKS
        ask_px = u_ba - self.VELVET_MM_IMPROVE_TICKS
        if bid_px >= ask_px: return  # crossed; bail
        if bid_size > 0:
            self._add(result, prod, bid_px, +bid_size)
        if ask_size > 0:
            self._add(result, prod, ask_px, -ask_size)

    def _mark_micro_entry(self, st, ts, cur_pos):
        """v31 EXP 7 LAYER B: Mark-driven small directional entry.

        Only fires in dead zone (no main signal active for >= DEAD_ZONE_LOOKBACK)
        and respects cooldown. Adjusts target_pos additively — never overrides
        a stronger main signal because we only set target if abs(target) < MARK_MICRO_TARGET.

        Logic: mark_bull/mark_bear are accumulators of Mark 14 + Mark 38-fade votes.
          mark_bull > threshold → market expects rising → small LONG
          mark_bear > threshold → market expects falling → small SHORT
        """
        # Skip if we just had a peak/valley signal (let main strategy work)
        last_peak  = int(st.get('last_peak_signal_ts', -10**9))
        last_valley= int(st.get('last_valley_signal_ts', -10**9))
        if (ts - max(last_peak, last_valley)) < self.DEAD_ZONE_LOOKBACK:
            return
        # Cooldown
        last_micro = int(st.get('last_micro_entry_ts', -10**9))
        if (ts - last_micro) < self.MARK_MICRO_COOLDOWN_TS:
            return
        # Skip if main has a meaningful target already
        cur_target = int(st.get('target_pos', 0))
        if abs(cur_target) >= self.MARK_MICRO_TARGET:
            return
        # Read mark sentiment
        bull = float(st.get('mark_bull', 0.0))
        bear = float(st.get('mark_bear', 0.0))
        if bull > self.MARK_MICRO_BULL_THRESH and bull > bear + 0.5:
            new_t = self.MARK_MICRO_TARGET
            # Only nudge if same direction or zero
            if cur_target >= 0:
                st['target_pos'] = max(cur_target, new_t)
                st['last_micro_entry_ts'] = ts
        elif bear > self.MARK_MICRO_BEAR_THRESH and bear > bull + 0.5:
            new_t = -self.MARK_MICRO_TARGET
            if cur_target <= 0:
                st['target_pos'] = min(cur_target, new_t)
                st['last_micro_entry_ts'] = ts

    def _voucher_passive_mm(self, result, state, st, S, in_chop, K):
        """Two-sided MM on a voucher body strike.

        Posts bid+1/ask-1 when:
          - main voucher target is converged (|target - pos| <= tol)
          - spread wide enough
          - not in chop
        Inventory-skewed.
        """
        if in_chop: return
        prod = f"VEV_{K}"
        od = state.order_depths.get(prod)
        if od is None: return
        u_bb, u_ba = self._bbb(od)
        if u_bb is None or u_ba is None: return
        spread = u_ba - u_bb
        if spread < self.VOUCHER_MM_MIN_SPREAD: return

        pos = int(state.position.get(prod, 0))
        target = int(st.get('voucher_target_pos', {}).get(str(K), 0))
        if abs(target - pos) > self.VOUCHER_MM_CONVERGED_TOL: return

        bid_size = self.VOUCHER_MM_QUOTE_SIZE
        ask_size = self.VOUCHER_MM_QUOTE_SIZE
        # Inventory skew
        if pos > self.VOUCHER_MM_POS_HARD:
            bid_size = 0; ask_size *= 2
        elif pos < -self.VOUCHER_MM_POS_HARD:
            bid_size *= 2; ask_size = 0
        elif pos > self.VOUCHER_MM_POS_SOFT:
            bid_size = max(1, bid_size // 2); ask_size = bid_size + 3
        elif pos < -self.VOUCHER_MM_POS_SOFT:
            ask_size = max(1, ask_size // 2); bid_size = ask_size + 3

        # Position-limit guards
        room_long = self.VOUCHER_POS_LIMIT - pos
        room_short = self.VOUCHER_POS_LIMIT + pos
        bid_size = max(0, min(bid_size, room_long))
        ask_size = max(0, min(ask_size, room_short))

        # Same NEVER_SELL_BELOW_FAIR gate keyed off VELVET S
        if self.NEVER_SELL_BELOW_FAIR and S < self.FAIR + self.MIN_SELL_EDGE:
            ask_size = 0
        if S > self.FAIR + 30:
            bid_size = 0

        bid_px = u_bb + self.VOUCHER_MM_IMPROVE_TICKS
        ask_px = u_ba - self.VOUCHER_MM_IMPROVE_TICKS
        if bid_px >= ask_px: return
        if bid_size > 0:
            self._add(result, prod, bid_px, +bid_size)
        if ask_size > 0:
            self._add(result, prod, ask_px, -ask_size)

    # ============================================================
    #   MAIN
    # ============================================================
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        st = self._load(state.traderData)
        ts = int(state.timestamp)
        self._reset_day(st, ts)

        u_od = state.order_depths.get(self.UNDERLYING)
        u_bb, u_ba = self._bbb(u_od)
        if u_bb is None or u_ba is None:
            return result, 0, self._save(st)
        S = 0.5 * (u_bb + u_ba)
        cur_pos = int(state.position.get(self.UNDERLYING, 0))

        # ============================================================
        # v31 FIX: dynamic VELVET fair (clip ±VELV_CLIP_RANGE around 5250)
        # ============================================================
        if self.USE_DYNAMIC_VELV_FAIR:
            vf = st.setdefault('velv_fair', {'sum': 0.0, 'n': 0})
            vf['sum'] = float(vf.get('sum', 0.0)) + S
            vf['n']   = int(vf.get('n', 0)) + 1
            cum = vf['sum'] / max(1, vf['n'])
            adj = max(-self.VELV_CLIP_RANGE, min(self.VELV_CLIP_RANGE, cum - self.FAIR))
            eff_fair = self.FAIR + adj
        else:
            eff_fair = self.FAIR
        st['eff_fair'] = eff_fair   # available for any helper that wants it

        # ============================================================
        # v31 FIX: VELVET fair-touch unload (mirrors HG winner)
        # Cross spread to flatten when bid >= eff_fair (long) or
        # ask <= eff_fair (short). Locks free profit before the rest
        # of the strategy reasons about position.
        # ============================================================
        if self.USE_VELV_FAIR_UNLOAD and abs(cur_pos) >= self.VELV_FAIR_UNLOAD_MIN_POS:
            if cur_pos > 0:
                qty_left = min(cur_pos, self.VELV_FAIR_UNLOAD_SIZE, self.POS_LIMIT + cur_pos)
                for bp in sorted(u_od.buy_orders.keys(), reverse=True):
                    if qty_left <= 0: break
                    if bp < eff_fair: break
                    avail = int(u_od.buy_orders[bp])
                    qty = min(qty_left, avail)
                    if qty > 0:
                        self._add(result, self.UNDERLYING, bp, -qty)
                        qty_left -= qty; cur_pos -= qty
            elif cur_pos < 0:
                qty_left = min(-cur_pos, self.VELV_FAIR_UNLOAD_SIZE, self.POS_LIMIT - cur_pos)
                for ap in sorted(u_od.sell_orders.keys()):
                    if qty_left <= 0: break
                    if ap > eff_fair: break
                    avail = -int(u_od.sell_orders[ap])
                    qty = min(qty_left, avail)
                    if qty > 0:
                        self._add(result, self.UNDERLYING, ap, qty)
                        qty_left -= qty; cur_pos += qty

        self._maybe_drift_reset_peak(st, ts, S)
        self._maybe_drift_reset_valley(st, ts, S)

        z_per = {5000: 0.0, 6000: 0.0, 5100: 0.0}
        z_signed_per = {}      # v31 EXP 6: signed Z for IV_SMILE_ARB direction
        pts = []
        for K in self.ALL_STRIKES:
            od = state.order_depths.get(f"VEV_{K}")
            bb, ba = self._bbb(od)
            if bb is None or ba is None: continue
            mid = 0.5*(bb+ba); spr = ba - bb
            if mid <= 0 or spr <= 0: continue
            iv = self._iv(S, K, mid)
            x = math.log(K / max(S, 1e-9))
            pts.append({'K': K, 'iv': iv, 'x': x, 'spread': spr})

        if len(pts) >= 5:
            # v31: also compute signed Z for IV smile arb on body strikes
            arb_strikes = set(self.SIGNAL_VOUCHERS) | set(getattr(self, 'IV_SMILE_ARB_STRIKES', ()))
            for tK in arb_strikes:
                tp = next((p for p in pts if p['K'] == tK), None)
                if tp is None: continue
                a, b, c = self._fit(pts, exclude_K=tK)
                fair_iv = a*tp['x']*tp['x'] + b*tp['x'] + c
                fair_iv = max(self.MIN_IV, min(self.MAX_IV, fair_iv))
                dV = tp['iv'] - fair_iv
                self._welford_update(st, tK, dV)
                signed_z = self._z(st, tK, dV)
                z_signed_per[tK] = signed_z
                if tK in z_per:
                    z_per[tK] = abs(signed_z)

        if (self.USE_CHOP_FILTER
                and z_per[self.CHOP_VOUCHER] >= self.CHOP_Z_THRESH
                and abs(S - self.FAIR) <= self.CHOP_DETECT_ZONE):
            st['last_chop_ts'] = ts
            if not st['chop_log'] or st['chop_log'][-1].get('ts') != ts:
                st['chop_log'].append({'ts': ts, 'z5100': round(z_per[5100], 2), 'S': round(S, 1)})

        in_chop = self._is_chop(st, ts, S)
        z5000 = z_per[5000]; z6000 = z_per[6000]

        # ---------- (A) dV escalation ----------
        if self.USE_DV and not in_chop:
            # v31 EXP 8: dV signals require min-dev to fire (asymmetric)
            _dev = S - self.FAIR
            _peak_min   = self.DV_MIN_DIST_PEAK   if self.USE_ASYM_PRICE_OVERRIDE else 0.0
            _valley_min = self.DV_MIN_DIST_VALLEY if self.USE_ASYM_PRICE_OVERRIDE else 0.0
            if (self._gate_peak(st, ts, S, cur_pos)
                    and z5000 > st['peak_anchor_z'] + self.ESCALATION_EPS
                    and _dev >= _peak_min):
                st['last_peak_signal_ts'] = ts
                st['peak_anchor_z']       = z5000
                st['last_peak_S']         = S
                st['valley_anchor_z']     = self.Z_THRESHOLD
                self._register_signal(st, ts, 'peak', 'dV', S, z5000, z_per[5100], u_bb)
            if (self._gate_valley(st, ts, S, cur_pos)
                    and z6000 > st['valley_anchor_z'] + self.ESCALATION_EPS
                    and -_dev >= _valley_min):
                st['last_valley_signal_ts'] = ts
                st['valley_anchor_z']       = z6000
                st['last_valley_S']         = S
                st['peak_anchor_z']         = self.Z_THRESHOLD
                self._register_signal(st, ts, 'valley', 'dV', S, z6000, z_per[5100], u_ba)

        # ---------- (A2) v21: PRICE-DEVIATION OVERRIDE (v31 EXP 8: asymmetric) ----------
        # Capture peaks/valleys that price reaches but dV signal misses.
        # Asymmetric thresholds reflect price-distribution asymmetry:
        #   peak:   +$18 (cheaper to participate, less upside room)
        #   valley: -$28 (deeper required, fatter downside tail)
        if self.USE_PRICE_OVERRIDE and not in_chop:
            _po_dev = S - eff_fair
            _peak_th   = self.PRICE_OVERRIDE_DIST_PEAK   if self.USE_ASYM_PRICE_OVERRIDE else self.PRICE_OVERRIDE_DIST
            _valley_th = self.PRICE_OVERRIDE_DIST_VALLEY if self.USE_ASYM_PRICE_OVERRIDE else self.PRICE_OVERRIDE_DIST
            if (_po_dev >= _peak_th
                    and self._gate_peak(st, ts, S, cur_pos)):
                _po_z = self.Z_THRESHOLD + 0.05
                st['last_peak_signal_ts'] = ts
                st['last_peak_S']         = S
                st['valley_anchor_z']     = self.Z_THRESHOLD
                self._register_signal(st, ts, 'peak', 'dV', S, _po_z, z_per[5100], u_bb)
            if (-_po_dev >= _valley_th
                    and self._gate_valley(st, ts, S, cur_pos)):
                _po_z = self.Z_THRESHOLD + 0.05
                st['last_valley_signal_ts'] = ts
                st['last_valley_S']         = S
                st['peak_anchor_z']         = self.Z_THRESHOLD
                self._register_signal(st, ts, 'valley', 'dV', S, _po_z, z_per[5100], u_ba)

        # ---------- (A3) v31 IV SMILE ARB ----------
        # Trade voucher directly when its IV deviates strongly from smile.
        # Additive only: nudges voucher target if existing target is in same
        # direction or zero. Never overrides opposite-direction conviction.
        # Conservative: body strikes only (5000, 5100) where signal is cleanest.
        if self.USE_IV_SMILE_ARB and not in_chop:
            for arb_K in self.IV_SMILE_ARB_STRIKES:
                signed_z = z_signed_per.get(arb_K, 0.0)
                if abs(signed_z) < self.IV_SMILE_ARB_Z:
                    continue
                # signed_z > 0: voucher IV ABOVE smile → overpriced → SHORT it
                # signed_z < 0: voucher IV BELOW smile → underpriced → LONG it
                arb_target = -int(self.IV_SMILE_ARB_TARGET) if signed_z > 0 else int(self.IV_SMILE_ARB_TARGET)
                cur_v = int(st['voucher_target_pos'].get(str(arb_K), 0))
                # Only push in same direction; don't fight existing target
                if arb_target > 0 and cur_v >= 0:
                    if arb_target > cur_v:
                        st['voucher_target_pos'][str(arb_K)] = arb_target
                elif arb_target < 0 and cur_v <= 0:
                    if arb_target < cur_v:
                        st['voucher_target_pos'][str(arb_K)] = arb_target

        # ---------- (A4) v27: ORDER BOOK IMBALANCE OVERLAY ----------
        # When bid depth >> ask depth on VELVET, sellers will exhaust → small long.
        # Only fires when main target is small (don't fight main signal).
        if self.USE_OB_IMBALANCE and not in_chop and u_od is not None:
            cur_target = int(st.get('target_pos', 0))
            if abs(cur_target) <= self.OB_IMBALANCE_MAX_MAIN:
                bid_vol_total = sum(int(v) for v in u_od.buy_orders.values())
                ask_vol_total = sum(abs(int(v)) for v in u_od.sell_orders.values())
                tot = bid_vol_total + ask_vol_total
                if tot > 0:
                    imb = (bid_vol_total - ask_vol_total) / tot
                    cd_clear = ts - st['last_ob_imb_ts'] >= self.OB_IMBALANCE_COOLDOWN_TS
                    if cd_clear and abs(imb) >= self.OB_IMBALANCE_THRESH:
                        sign = +1 if imb > 0 else -1
                        st['last_ob_imb_ts'] = ts
                        # Don't override stronger main target
                        new_t = sign * self.OB_IMBALANCE_TARGET
                        if sign > 0:
                            st['target_pos'] = max(cur_target, new_t)
                        else:
                            st['target_pos'] = min(cur_target, new_t)
                        if self.USE_VOUCHER_OVERLAY:
                            for K in self.VOUCHER_CORE_STRIKES:
                                cur_v = int(st['voucher_target_pos'].get(str(K), 0))
                                new_v = sign * self.OB_IMBALANCE_VOUCHER
                                if sign > 0:
                                    st['voucher_target_pos'][str(K)] = max(cur_v, new_v)
                                else:
                                    st['voucher_target_pos'][str(K)] = min(cur_v, new_v)
                        st['ob_imb_log'].append({'ts': ts, 'imb': round(imb, 2), 'sign': sign})

        # ---------- (A5) v27: MARK COUNTERPARTY COMBO ----------
        # Joint signal: informed Mark buys VELVET + uninformed Mark sells in same tick
        if self.USE_MARK_COMBO and not in_chop:
            mt = getattr(state, 'market_trades', None) or {}
            velvet_marks = mt.get('VELVETFRUIT_EXTRACT', []) or []
            informed_buys = any(getattr(t,'buyer','') in self.MARK_COMBO_INFORMED for t in velvet_marks)
            uninformed_sells = any(getattr(t,'seller','') in self.MARK_COMBO_UNINFORMED for t in velvet_marks)
            informed_sells = any(getattr(t,'seller','') in self.MARK_COMBO_INFORMED for t in velvet_marks)
            uninformed_buys = any(getattr(t,'buyer','') in self.MARK_COMBO_UNINFORMED for t in velvet_marks)

            cd_clear = ts - st['last_mark_combo_ts'] >= self.MARK_COMBO_COOLDOWN_TS
            if cd_clear:
                # Valley combo: informed buying + uninformed selling = strong long
                if informed_buys and uninformed_sells:
                    st['last_mark_combo_ts'] = ts
                    cur_t = int(st.get('target_pos', 0))
                    new_t = self.MARK_COMBO_VELVET_TARGET
                    st['target_pos'] = max(cur_t, new_t)
                    if self.USE_VOUCHER_OVERLAY:
                        for K in self.VOUCHER_CORE_STRIKES:
                            cur_v = int(st['voucher_target_pos'].get(str(K), 0))
                            st['voucher_target_pos'][str(K)] = max(cur_v, self.MARK_COMBO_VOUCHER_TARGET)
                    st['mark_combo_log'].append({'ts': ts, 'kind': 'valley_combo'})
                # Peak combo: informed selling + uninformed buying = strong short
                elif informed_sells and uninformed_buys:
                    st['last_mark_combo_ts'] = ts
                    cur_t = int(st.get('target_pos', 0))
                    new_t = -self.MARK_COMBO_VELVET_TARGET
                    st['target_pos'] = min(cur_t, new_t)
                    if self.USE_VOUCHER_OVERLAY:
                        for K in self.VOUCHER_CORE_STRIKES:
                            cur_v = int(st['voucher_target_pos'].get(str(K), 0))
                            st['voucher_target_pos'][str(K)] = min(cur_v, -self.MARK_COMBO_VOUCHER_TARGET)
                    st['mark_combo_log'].append({'ts': ts, 'kind': 'peak_combo'})

        # ---------- (B) Mark exception + deep override ----------
        if self.USE_MARK and not in_chop:
            mt = getattr(state, 'market_trades', None) or {}
            for prod in self.MARK_PRODUCTS:
                trades = mt.get(prod, []) or []
                for tr in trades:
                    buyer  = getattr(tr, 'buyer', '') or ''
                    seller = getattr(tr, 'seller', '') or ''
                    qty    = int(getattr(tr, 'quantity', 0) or 0)
                    if seller in self.INFORMED_SELLERS:
                        deep_far     = (S - self.FAIR) >= self.DEEP_DIST and z5000 >= self.DEEP_MIN_Z
                        normal_excep = z5000 >= self.MARK_EXCEPTION_Z
                        if (deep_far or normal_excep) and self._gate_peak(st, ts, S, cur_pos):
                            st['last_peak_signal_ts'] = ts
                            st['peak_anchor_z']       = max(st['peak_anchor_z'], z5000)
                            st['last_peak_S']         = S
                            st['valley_anchor_z']     = self.Z_THRESHOLD
                            tag = 'mark_deep' if deep_far and not normal_excep else 'mark'
                            self._register_signal(st, ts, 'peak', tag, S, z5000, z_per[5100], u_bb,
                                                  {'who': seller, 'sym': prod, 'qty': qty})
                            continue
                    valley_aligned = (buyer in self.INFORMED_BUYERS) or (seller in self.FADE_SELLERS)
                    if valley_aligned:
                        deep_far     = (self.FAIR - S) >= self.DEEP_DIST and z6000 >= self.DEEP_MIN_Z
                        normal_excep = z6000 >= self.MARK_EXCEPTION_Z
                        if (deep_far or normal_excep) and self._gate_valley(st, ts, S, cur_pos):
                            st['last_valley_signal_ts'] = ts
                            st['valley_anchor_z']       = max(st['valley_anchor_z'], z6000)
                            st['last_valley_S']         = S
                            st['peak_anchor_z']         = self.Z_THRESHOLD
                            who = buyer if buyer in self.INFORMED_BUYERS else "~" + seller
                            tag = 'mark_deep' if deep_far and not normal_excep else 'mark'
                            self._register_signal(st, ts, 'valley', tag, S, z6000, z_per[5100], u_ba,
                                                  {'who': who, 'sym': prod, 'qty': qty})

        # Risk management & trim (also adjusts voucher targets)
        day_ts = ts % 1_000_000
        if self.HARD_FLATTEN_AFTER_TS is not None and day_ts >= self.HARD_FLATTEN_AFTER_TS:
            st['target_pos'] = 0
            if self.USE_VOUCHER_OVERLAY:
                for K in self.VOUCHER_OVERLAY_STRIKES:
                    st['voucher_target_pos'][str(K)] = 0
        else:
            self._apply_fair_trim(st, ts, S)
            self._apply_adverse_stop(st, ts, S)
            self._apply_late_cap(st, ts, S)

        # Execute VELVET to target
        self._exec_one_product(result, state, st, self.UNDERLYING,
                                int(st.get('target_pos', 0)),
                                self.POS_LIMIT, self.MAX_TAKE_PER_TICK,
                                self.MAX_PASSIVE_PER_TICK, self.TAKE_FRACTION, S)

        # v24: VELVET passive MM — add bid+1/ask-1 two-sided quotes when main
        # strategy is converged, harvesting spread from incoming Mark flow.
        if self.USE_VELVET_MM:
            self._velvet_passive_mm(result, state, st, S, in_chop)

        # v31 EXP 7 LAYER B: Mark-driven micro-entries in dead zone.
        # Fires AFTER main signal handlers — only nudges target if main is silent.
        if self.USE_MARK_MICRO_ENTRY and not in_chop:
            self._mark_micro_entry(st, ts, cur_pos)

        # v26: voucher passive MM on body strikes — same idea as VELVET MM but
        # for vouchers, with main-target convergence gate to avoid fighting entry.
        if self.USE_VOUCHER_MM:
            for K in self.VOUCHER_MM_STRIKES:
                self._voucher_passive_mm(result, state, st, S, in_chop, K)

        # Execute vouchers to their targets (gated by VELVET's S vs FAIR)
        if self.USE_VOUCHER_OVERLAY:
            for K in self.VOUCHER_OVERLAY_STRIKES:
                v_target = int(st['voucher_target_pos'].get(str(K), 0))
                self._exec_one_product(result, state, st, f"VEV_{K}",
                                        v_target,
                                        self.VOUCHER_POS_LIMIT,
                                        self.VOUCHER_MAX_TAKE_PER_TICK,
                                        self.VOUCHER_MAX_PASSIVE_PER_TICK,
                                        self.VOUCHER_TAKE_FRACTION, S)

        # ============================================================
        # v11: Chop-follow micro-layer
        # ============================================================
        if self.USE_CHOP_FOLLOW:
            self._run_chop_follow(result, state, st, S, ts)

        # ============================================================
        # v12: HYDROGEL_PACK Mark-pair trader (independent of VELVET)
        # ============================================================
        if self.USE_HG_TRADER:
            self._run_hg_trader(result, state, st, ts)

        # v13: passive MM on HG (replaced with 500873+FAIR_UNLOAD winner)
        if self.USE_HG_MM:
            self._run_hg_mm(result, state, st)

        # v16: passive MM on VEV_4000 (deep ITM, $20 spread, huge Mark flow)
        if self.USE_VEV4000_MM:
            self._run_vev4000_mm(result, state)

        return result, 0, self._save(st)

    def _run_vev4000_mm(self, result, state):
        """Passive MM on VEV_4000 — deep ITM, mid ≈ S - 4000, spread ~$20.78.
        Independent of any other logic. Posts bid+1 / ask-1 each tick with
        inventory-aware skew. Captures spread on Mark crosses (Mark 14/38)."""
        product = "VEV_4000"
        od = state.order_depths.get(product)
        if od is None: return
        bb, ba = self._bbb(od)
        if bb is None or ba is None: return
        spread = ba - bb
        if spread < self.VEV4000_MM_MIN_SPREAD: return

        pos = int(state.position.get(product, 0))
        cap = 300  # Prosperity hard limit for vouchers

        bid_px = bb + self.VEV4000_MM_IMPROVE_TICKS
        ask_px = ba - self.VEV4000_MM_IMPROVE_TICKS
        if bid_px >= ask_px: return

        # Inventory-aware sizing
        bid_size = self.VEV4000_MM_QUOTE_SIZE
        ask_size = self.VEV4000_MM_QUOTE_SIZE
        if pos > self.VEV4000_MM_POS_SOFT:
            scale = max(0.0, 1.0 - (pos - self.VEV4000_MM_POS_SOFT) /
                        max(1, self.VEV4000_MM_POS_HARD - self.VEV4000_MM_POS_SOFT))
            bid_size = int(round(self.VEV4000_MM_QUOTE_SIZE * scale * 0.5))
        elif pos < -self.VEV4000_MM_POS_SOFT:
            scale = max(0.0, 1.0 - (-pos - self.VEV4000_MM_POS_SOFT) /
                        max(1, self.VEV4000_MM_POS_HARD - self.VEV4000_MM_POS_SOFT))
            ask_size = int(round(self.VEV4000_MM_QUOTE_SIZE * scale * 0.5))

        # Hard cap
        if pos >= self.VEV4000_MM_POS_HARD: bid_size = 0
        if pos <= -self.VEV4000_MM_POS_HARD: ask_size = 0

        bid_size = min(bid_size, max(0, cap - pos))
        ask_size = min(ask_size, max(0, cap + pos))

        if bid_size > 0:
            self._add(result, product, bid_px, bid_size)
        if ask_size > 0:
            self._add(result, product, ask_px, -ask_size)

    def _run_hg_mm(self, result, state, st):
        """REPLACED with hg_500873_plus_fair_unload winner.
        Tested: $126,096 over 3 days (+$3.4k vs prior version), zero DD added.

        Layers:
          L1. Aggressive fair-touch unload — locks profit when bid/ask crosses fair
          L2. Distance-scaled directional take + passive (500873.py logic)
          L3. Near-fair MM
        """
        SYM = 'HYDROGEL_PACK'
        LIMIT = 200
        # Tunable params (matched to hg_500873_plus_fair_unload.py)
        HG_PRIOR = 9994.0
        HG_CLIP_RANGE = 5.0
        ENTRY_BAND = 8.0; OFFLOAD_BAND = 10.0; FLAT_BAND = 5.0
        SOFT_DIST = 20.0; MID_DIST = 35.0; HARD_DIST = 50.0
        SOFT_T = 60; MID_T = 140; HARD_T = 185; MAX_T = 200
        TAKE_THRESHOLD = 30.0; OPEN_TAKE_EDGE = 7.0; OFFLOAD_TAKE_EDGE = 2.0
        MAX_TAKE_PER_TICK = 30; MAX_PASSIVE_PER_TICK = 50
        TAKE_FRACTION = 0.45; MIN_TAKER_NEED = 8
        MM_BAND = 9.0; MM_QUOTE_SIZE = 12; MM_MAX_INV = 80; MM_MIN_SPREAD = 10
        FAIR_TOUCH_UNLOAD_SIZE = 100; FAIR_TOUCH_MIN_POS = 8
        END_REDUCE_TS = 990_000; END_FLAT_TS = 997_000

        od = state.order_depths.get(SYM)
        if od is None: return
        bb, ba = self._bbb(od)
        if bb is None or ba is None: return
        spread = ba - bb
        if not od.buy_orders or not od.sell_orders: return
        mid = 0.5 * (bb + ba)
        ts = state.timestamp
        pos = int(state.position.get(SYM, 0))

        # Update HG-specific cum-mid fair (per-tick clip ±5)
        # Stored in main 'st' dict so it survives across run() calls via traderData.
        hg_state = st.setdefault('hg_unload_state', {'mid_sum': 0.0, 'mid_n': 0})
        hg_state['mid_sum'] = float(hg_state.get('mid_sum', 0.0)) + mid
        hg_state['mid_n']   = int(hg_state.get('mid_n', 0)) + 1
        cum = hg_state['mid_sum'] / max(1, hg_state['mid_n'])
        adj = max(-HG_CLIP_RANGE, min(HG_CLIP_RANGE, cum - HG_PRIOR))
        fair = HG_PRIOR + adj

        orders: List[Order] = []

        # ====================================================================
        # L1: Aggressive fair-touch unload — locks free profit when
        #     bid >= fair (long) or ask <= fair (short)
        # ====================================================================
        if abs(pos) >= FAIR_TOUCH_MIN_POS:
            if pos > 0:
                qty_left = min(pos, FAIR_TOUCH_UNLOAD_SIZE, LIMIT + pos)
                for bid_p in sorted(od.buy_orders.keys(), reverse=True):
                    if qty_left <= 0: break
                    if bid_p < fair: break
                    avail = int(od.buy_orders[bid_p])
                    qty = min(qty_left, avail)
                    if qty > 0:
                        orders.append(Order(SYM, int(bid_p), -int(qty)))
                        qty_left -= qty; pos -= qty
            elif pos < 0:
                qty_left = min(-pos, FAIR_TOUCH_UNLOAD_SIZE, LIMIT - pos)
                for ask_p in sorted(od.sell_orders.keys()):
                    if qty_left <= 0: break
                    if ask_p > fair: break
                    avail = -int(od.sell_orders[ask_p])
                    qty = min(qty_left, avail)
                    if qty > 0:
                        orders.append(Order(SYM, int(ask_p), int(qty)))
                        qty_left -= qty; pos += qty

        # ====================================================================
        # L2: Distance-scaled directional target
        # ====================================================================
        dev = mid - fair; abs_dev = abs(dev); day_ts = ts % 1_000_000
        if day_ts >= END_FLAT_TS:
            target = 0
        else:
            if abs_dev < ENTRY_BAND:
                target = 0
            else:
                if abs_dev < SOFT_DIST:
                    x = (abs_dev - ENTRY_BAND) / (SOFT_DIST - ENTRY_BAND)
                    mag = int(round(x * SOFT_T))
                elif abs_dev < MID_DIST:
                    x = (abs_dev - SOFT_DIST) / (MID_DIST - SOFT_DIST)
                    mag = int(round(SOFT_T + x * (MID_T - SOFT_T)))
                elif abs_dev < HARD_DIST:
                    x = (abs_dev - MID_DIST) / (HARD_DIST - MID_DIST)
                    mag = int(round(MID_T + x * (HARD_T - MID_T)))
                else:
                    mag = MAX_T
                target = -mag if dev > 0 else mag

            if abs_dev <= FLAT_BAND:
                target = 0
            elif abs_dev < OFFLOAD_BAND:
                scale = (abs_dev - FLAT_BAND) / max(1e-9, OFFLOAD_BAND - FLAT_BAND)
                target = int(round(target * scale))

            if day_ts >= END_REDUCE_TS:
                target = max(-80, min(80, target))

            target = max(-LIMIT, min(LIMIT, target))

            # v35 MERGE: ADD Mark 14 follow signal as a target offset.
            # The offset (max ±HG_TARGET_CAP=30) combines additively with the
            # mean-reversion target.  When they align (price below fair AND
            # Mark 14 buying), they reinforce; when they oppose, partial cancel.
            # KEY: only the TARGET shifts — execution gates (reducing/far_entry/
            # very_late) still control whether to cross spread, so the Mark
            # signal expresses through PASSIVE quoting, not spread crosses.
            if day_ts < END_FLAT_TS:
                mark_offset = int(st.get('hg_mark_target', 0))
                if mark_offset != 0:
                    target = max(-LIMIT, min(LIMIT, target + mark_offset))

        # L2 execution: aggressor + passive (500873.py logic)
        sim_pos = pos
        need = target - sim_pos
        reducing = (pos > target and pos > 0) or (pos < target and pos < 0) or target == 0
        far_entry = abs_dev >= TAKE_THRESHOLD
        very_late = day_ts >= END_FLAT_TS

        if need > 0:
            buy_cap = LIMIT - sim_pos; max_buy = min(buy_cap, need)
            if max_buy > 0:
                allow = (very_late
                         or (reducing and ba <= fair + OFFLOAD_TAKE_EDGE)
                         or (far_entry and ba <= fair - OPEN_TAKE_EDGE))
                if allow and max_buy >= MIN_TAKER_NEED:
                    take_q = min(MAX_TAKE_PER_TICK, int(math.ceil(max_buy * TAKE_FRACTION)))
                    bought = 0
                    for px in sorted(od.sell_orders.keys()):
                        avail = -int(od.sell_orders[px])
                        if avail <= 0: continue
                        q = min(avail, take_q - bought, LIMIT - (sim_pos + bought))
                        if q <= 0: break
                        orders.append(Order(SYM, int(px), int(q))); bought += q
                        if bought >= take_q: break
                    sim_pos += bought
        elif need < 0:
            sell_cap = LIMIT + sim_pos; max_sell = min(sell_cap, -need)
            if max_sell > 0:
                allow = (very_late
                         or (reducing and bb >= fair - OFFLOAD_TAKE_EDGE)
                         or (far_entry and bb >= fair + OPEN_TAKE_EDGE))
                if allow and max_sell >= MIN_TAKER_NEED:
                    take_q = min(MAX_TAKE_PER_TICK, int(math.ceil(max_sell * TAKE_FRACTION)))
                    sold = 0
                    for px in sorted(od.buy_orders.keys(), reverse=True):
                        avail = int(od.buy_orders[px])
                        if avail <= 0: continue
                        q = min(avail, take_q - sold, LIMIT + (sim_pos - sold))
                        if q <= 0: break
                        orders.append(Order(SYM, int(px), -int(q))); sold += q
                        if sold >= take_q: break
                    sim_pos -= sold

        # L2 passive component
        need = target - sim_pos
        if need > 0:
            qty = min(MAX_PASSIVE_PER_TICK, need, LIMIT - sim_pos)
            if qty > 0:
                px = bb + 1
                if target > 0:
                    px = min(px, int(math.floor(fair - 1)))
                else:
                    px = min(px, int(math.floor(fair + OFFLOAD_TAKE_EDGE)))
                if px < ba:
                    orders.append(Order(SYM, int(px), int(qty)))
        elif need < 0:
            qty = min(MAX_PASSIVE_PER_TICK, -need, LIMIT + sim_pos)
            if qty > 0:
                px = ba - 1
                if target < 0:
                    px = max(px, int(math.ceil(fair + 1)))
                else:
                    px = max(px, int(math.ceil(fair - OFFLOAD_TAKE_EDGE)))
                if px > bb:
                    orders.append(Order(SYM, int(px), -int(qty)))

        # ====================================================================
        # L3: Near-fair MM (only when target ~ 0 and inventory contained)
        # ====================================================================
        sim_pos = pos + sum(o.quantity for o in orders)
        if (spread >= MM_MIN_SPREAD and abs(mid - fair) <= MM_BAND
                and abs(target) <= 10 and abs(sim_pos) <= MM_MAX_INV):
            bid_size = MM_QUOTE_SIZE; ask_size = MM_QUOTE_SIZE
            if mid > fair + 2: ask_size += 6; bid_size = max(2, bid_size - 4)
            elif mid < fair - 2: bid_size += 6; ask_size = max(2, ask_size - 4)
            if sim_pos > 0:
                ask_size += min(16, sim_pos // 6); bid_size = max(1, bid_size - sim_pos // 10)
            elif sim_pos < 0:
                bid_size += min(16, -sim_pos // 6); ask_size = max(1, ask_size - (-sim_pos) // 10)
            bid_size = max(0, min(bid_size, LIMIT - sim_pos))
            ask_size = max(0, min(ask_size, LIMIT + sim_pos))
            bid_px = min(bb + 1, int(math.floor(fair - 1)))
            ask_px = max(ba - 1, int(math.ceil(fair + 1)))
            if bid_size > 0 and bid_px < ba:
                orders.append(Order(SYM, int(bid_px), int(bid_size)))
            if ask_size > 0 and ask_px > bb:
                orders.append(Order(SYM, int(ask_px), -int(ask_size)))

        # Append all generated orders to result
        for o in orders:
            self._add(result, SYM, o.price, o.quantity)

    def _run_hg_trader(self, result, state, st, ts):
        """v35 MERGE: SIGNAL-ONLY scanner.  Maintains st['hg_mark_target']
        as an offset that _run_hg_mm reads and adds to its L2 target.  No
        executor — single execution point in HG_MM avoids the dual-fight
        problem that lost -$159k in v34.

        Filter: only big Mark 14 trades (qty >= HG_FOLLOW_MIN_QTY) update the
        offset; small trades are noise.  Mark 38 ignored (just Mark 14's mirror
        counterparty).  Slow decay because Mark 14's signal is long-horizon.

        `result` and the underlying order book are unused — kept in signature
        so the existing run() call site doesn't need to change.
        """
        offset = int(st.get('hg_mark_target', 0))
        last_sig = int(st.get('hg_mark_last_ts', -10**9))

        # Time-decay offset toward zero if no recent qualifying Mark activity
        if ts - last_sig > self.HG_DECAY_TS and abs(offset) > 0:
            offset = int(round(offset * self.HG_DECAY_FRAC))

        # Scan market_trades; only big Mark 14 trades update the offset
        mt = getattr(state, 'market_trades', None) or {}
        hg_trades = mt.get(self.HG_PRODUCT, []) or []
        for tr in hg_trades:
            buyer  = getattr(tr, 'buyer', '') or ''
            seller = getattr(tr, 'seller', '') or ''
            qty    = int(getattr(tr, 'quantity', 0) or 0)
            if qty < self.HG_FOLLOW_MIN_QTY:
                continue
            if buyer in self.HG_FOLLOW_BUYERS:
                offset = min(self.HG_TARGET_CAP, offset + self.HG_FOLLOW_SIZE)
                st['hg_mark_last_ts'] = ts
            elif seller in self.HG_FOLLOW_SELLERS:
                offset = max(-self.HG_TARGET_CAP, offset - self.HG_FOLLOW_SIZE)
                st['hg_mark_last_ts'] = ts
            # Mark 38 (fade) intentionally not handled — pure mirror of Mark 14

        st['hg_mark_target'] = offset
        # Legacy state key kept in sync for backwards-compat (unused now)
        st['hg_target_pos'] = offset

    def _run_chop_follow(self, result, state, st, S, ts):
        """Counterparty-follow micro-layer for the chop zone.

        Active only when |S - FAIR| < CHOP_FOLLOW_ZONE AND main target_pos == 0
        (we don't have a directional position yet). Watches market_trades for
        Mark events on body-strike vouchers and accumulates a small target.

        When the price exits chop, the chop targets are reset to 0 (we hand
        execution back to the main strategy and let trim/main close out).
        """
        chop_targets = st.setdefault('chop_target_pos', {})

        # If we exit the chop zone: clear chop bookkeeping. Do NOT issue any
        # orders here — the main strategy now owns those products. Issuing
        # exec orders to bring chop_target=0 would fight the main voucher
        # overlay (which is just turning ON its own non-zero targets).
        if abs(S - self.FAIR) >= self.CHOP_FOLLOW_ZONE:
            for K_str in list(chop_targets.keys()):
                chop_targets[K_str] = 0
            return

        # Don't run if main strategy holds ANY target (VELVET or voucher).
        # The chop layer must NOT overlap with main targets on the same product.
        main_target = int(st.get('target_pos', 0))
        main_voucher_max = max((abs(int(v)) for v in st.get('voucher_target_pos', {}).values()),
                                default=0)
        if abs(main_target) > 5 or main_voucher_max > 5:
            return

        # Scan market_trades for Mark events on our chop-watch vouchers
        mt = getattr(state, 'market_trades', None) or {}
        for K in self.CHOP_FOLLOW_VOUCHERS:
            sym = f"VEV_{K}"
            trades = mt.get(sym, []) or []
            if not trades: continue
            for tr in trades:
                buyer = getattr(tr, 'buyer', '') or ''
                seller = getattr(tr, 'seller', '') or ''
                cur = int(chop_targets.get(str(K), 0))

                # BUY signals: informed buyer, OR fade Mark 22 sells
                if (buyer in self.CHOP_FOLLOW_INFORMED_BUYERS
                        or seller in self.CHOP_FOLLOW_FADE_SELLERS):
                    new_target = min(self.CHOP_FOLLOW_CAP,
                                     cur + self.CHOP_FOLLOW_BASE_SIZE)
                    chop_targets[str(K)] = new_target
                    continue
                # SELL signals: informed seller (Mark 14/01)
                if seller in self.CHOP_FOLLOW_INFORMED_SELLERS:
                    new_target = max(-self.CHOP_FOLLOW_CAP,
                                     cur - self.CHOP_FOLLOW_BASE_SIZE)
                    chop_targets[str(K)] = new_target

        # Execute toward chop targets — but ONLY on products where main
        # voucher target is zero (avoid two layers fighting on same product).
        main_v = st.get('voucher_target_pos', {})
        for K in self.CHOP_FOLLOW_VOUCHERS:
            if abs(int(main_v.get(str(K), 0))) > 5:
                continue
            target = int(chop_targets.get(str(K), 0))
            self._exec_one_product(result, state, st, f"VEV_{K}",
                                    target, self.VOUCHER_POS_LIMIT,
                                    self.VOUCHER_MAX_TAKE_PER_TICK,
                                    self.VOUCHER_MAX_PASSIVE_PER_TICK,
                                    self.VOUCHER_TAKE_FRACTION, S)