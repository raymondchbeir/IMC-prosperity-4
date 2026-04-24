import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

PRODUCT = "ASH_COATED_OSMIUM"
ANCHOR = 10000.0
LAMBDA = 0.45
WINDOW = 60


def load_prices():
    files = [
        "prices_round_1_day_-2.csv",
        "prices_round_1_day_-1.csv",
        "prices_round_1_day_0.csv",
    ]

    return pd.concat(
        [pd.read_csv(f, sep=";") for f in files],
        ignore_index=True,
    )


def compute_features(df):
    df = df[df["product"] == PRODUCT].copy()
    df = df.sort_values(["day", "timestamp"]).reset_index(drop=True)

    df["bid_price"] = df["bid_price_1"]
    df["ask_price"] = df["ask_price_1"]
    df["bid_volume"] = df["bid_volume_1"].fillna(0).abs()
    df["ask_volume"] = df["ask_volume_1"].fillna(0).abs()

    df["mid"] = df["mid_price"]

    # Remove broken book rows, especially mid_price = 0 rows
    df = df[
        (df["mid"] > 0)
        & (df["bid_price"] > 0)
        & (df["ask_price"] > 0)
        & (df["ask_price"] >= df["bid_price"])
    ].copy()

    df["spread"] = df["ask_price"] - df["bid_price"]

    # Compute rolling values separately by day to avoid fake day-boundary jumps
    g = df.groupby("day", group_keys=False)

    df["median_mid"] = g["mid"].transform(
        lambda x: x.rolling(WINDOW, min_periods=10).median()
    )

    df["fair"] = ANCHOR + LAMBDA * (df["median_mid"] - ANCHOR)

    df["mad"] = g.apply(
        lambda x: (x["mid"] - x["median_mid"])
        .abs()
        .rolling(WINDOW, min_periods=10)
        .median()
    ).reset_index(level=0, drop=True)

    df["scale"] = df["mad"].clip(lower=1.0)
    df["z"] = (df["mid"] - df["fair"]) / df["scale"]

    df["delta_mid"] = g["mid"].diff()

    df["rolling_vol"] = g["delta_mid"].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    )

    df["rolling_return_5"] = g["mid"].diff(5)
    df["rolling_return_20"] = g["mid"].diff(20)
    df["abs_z_change"] = g["z"].diff().abs()

    denom = df["bid_volume"] + df["ask_volume"]
    df["imbalance"] = np.where(
        denom > 0,
        (df["bid_volume"] - df["ask_volume"]) / denom,
        0,
    )

    # Evaluation-only columns. Do not train on these.
    df["future_return_20"] = g["mid"].shift(-20) - df["mid"]
    df["future_abs_return_20"] = df["future_return_20"].abs()

    return df


def fit_hmm(df, n_states=3):
    feature_cols = [
        "z",
        "delta_mid",
        "rolling_vol",
        "spread",
        "imbalance",
        "rolling_return_5",
        "rolling_return_20",
        "abs_z_change",
    ]

    clean = df.dropna(subset=feature_cols).copy()

    X = clean[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=1000,
        random_state=42,
        min_covar=1e-3,
    )

    model.fit(X_scaled)

    probs = model.predict_proba(X_scaled)

    clean["hmm_state"] = model.predict(X_scaled)
    clean["hmm_confidence"] = probs.max(axis=1)

    return clean, model, scaler, feature_cols


def profile_regimes(df):
    summary = df.groupby("hmm_state").agg(
        count=("timestamp", "count"),
        avg_z=("z", "mean"),
        avg_abs_z=("z", lambda x: x.abs().mean()),
        avg_delta_mid=("delta_mid", "mean"),
        avg_rolling_vol=("rolling_vol", "mean"),
        avg_spread=("spread", "mean"),
        avg_imbalance=("imbalance", "mean"),
        avg_future_return_20=("future_return_20", "mean"),
        avg_future_abs_return_20=("future_abs_return_20", "mean"),
        avg_confidence=("hmm_confidence", "mean"),
    )

    print("\n=== HMM Regime Profile ===")
    print(summary)


def print_sanity_checks(features):
    print("\n=== Feature sanity check ===")
    print(features[["mid", "spread", "delta_mid", "rolling_vol", "z"]].describe())

    print("\n=== Largest delta_mid rows ===")
    largest = features.reindex(
        features["delta_mid"].abs().sort_values(ascending=False).head(10).index
    )

    print(
        largest[
            ["day", "timestamp", "mid", "bid_price", "ask_price", "delta_mid"]
        ]
    )


def save_state_map(df):
    state_summary = df.groupby("hmm_state").agg(
        count=("timestamp", "count"),
        avg_abs_future_return=("future_abs_return_20", "mean"),
        avg_spread=("spread", "mean"),
        avg_vol=("rolling_vol", "mean"),
        avg_abs_z=("z", lambda x: x.abs().mean()),
    )

    print("\n=== State interpretation helper ===")
    print(state_summary)

    print("\nSuggested interpretation:")
    print("1. Highest avg_vol and/or avg_spread = volatile/noisy regime.")
    print("2. Highest avg_abs_future_return = runner / directional regime.")
    print("3. Lowest avg_vol and low future movement = mean-reverting/choppy regime.")


if __name__ == "__main__":
    raw = load_prices()
    features = compute_features(raw)

    print_sanity_checks(features)

    hmm_df, model, scaler, feature_cols = fit_hmm(features, n_states=3)

    profile_regimes(hmm_df)
    save_state_map(hmm_df)

    hmm_df.to_csv("osmium_hmm_regimes.csv", index=False)

    print("\nSaved: osmium_hmm_regimes.csv")