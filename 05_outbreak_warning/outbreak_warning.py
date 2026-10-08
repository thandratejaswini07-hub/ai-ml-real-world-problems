"""Community disease-outbreak early warning (synthetic data demo).

Signals: clinic visits, pharmacy fever-medicine sales, symptom search trends.
Method : learn each signal's normal seasonal/weekly pattern -> z-score the residuals
         -> combine signals -> alert on sustained anomalies. Isolation Forest is
         run alongside as a second opinion.
Run:  python outbreak_warning.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

OUT = Path("../outputs/outbreak")
SIGNALS = ["clinic_visits", "pharmacy_sales", "search_trend"]


def make_data(days=1095, outbreak_start=1000, seed=3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    season = 1 + 0.3 * np.sin(2 * np.pi * t / 365) + 0.1 * np.cos(4 * np.pi * t / 365)
    weekly = 1 + 0.15 * (pd.Series(t % 7).isin([5, 6]).values * -1)  # quieter weekends
    base = season * weekly
    since = np.clip(t - outbreak_start, 0, None)
    # Indirect signals react earlier than clinic visits
    lift = {"clinic_visits": np.where(since > 6, 1.0 * (1.12 ** (since - 6)) - 1, 0),
            "pharmacy_sales": np.where(since > 3, 0.9 * (1.12 ** (since - 3)) - 1, 0),
            "search_trend": np.where(since > 0, 0.7 * (1.12 ** since) - 1, 0)}
    scale = {"clinic_visits": 120, "pharmacy_sales": 300, "search_trend": 60}
    df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=days)})
    for s in SIGNALS:
        mean = scale[s] * base * (1 + np.minimum(lift[s], 2.5))
        noise = 1.0 if s != "search_trend" else 1.8  # search data is noisier
        df[s] = rng.poisson(mean) + rng.normal(0, noise, days)
    df["true_outbreak"] = (t >= outbreak_start).astype(int)
    return df


def design(n_days, dates):
    t = np.arange(n_days)
    cols = [np.ones(n_days), t / n_days,
            np.sin(2 * np.pi * t / 365), np.cos(2 * np.pi * t / 365),
            np.sin(4 * np.pi * t / 365), np.cos(4 * np.pi * t / 365)]
    dow = dates.dt.dayofweek.values
    cols += [(dow == d).astype(float) for d in range(6)]
    return np.column_stack(cols)


def zscores(df, train_days):
    X = design(len(df), df["date"])
    z = pd.DataFrame(index=df.index)
    for s in SIGNALS:
        y = df[s].values
        beta, *_ = np.linalg.lstsq(X[:train_days], y[:train_days], rcond=None)
        resid = y - X @ beta
        z[s] = resid / resid[:train_days].std()
    return z


def detect(df, train_days=730, z_thresh=3.0, consecutive=3):
    z = zscores(df, train_days)
    df["composite_z"] = z[SIGNALS].clip(lower=0).mean(axis=1).rolling(2, min_periods=1).mean() * 1.8
    hot = (df["composite_z"] > z_thresh).astype(int)
    df["alert"] = (hot.rolling(consecutive).sum() >= consecutive).astype(int)
    iso = IsolationForest(contamination=0.01, random_state=0).fit(z.iloc[:train_days])
    df["iso_anomaly"] = (iso.predict(z) == -1).astype(int)
    return df


def main():
    df = detect(make_data())
    OUT.mkdir(parents=True, exist_ok=True)
    start = df.loc[df.true_outbreak == 1, "date"].iloc[0]
    first = df.loc[df.alert == 1, "date"]
    false_alarms = ((df.alert == 1) & (df.true_outbreak == 0)).sum()
    print("Outbreak truly began :", start.date())
    if len(first):
        print("First early warning  :", first.iloc[0].date(),
              f"({(first.iloc[0] - start).days} days after onset)")
    else:
        print("No alert raised")
    print("False-alarm days     :", int(false_alarms))
    # Hospital-admission-based detection would only see the clinic signal rise strongly later:
    clinic_late = df.loc[(df.clinic_visits > df.clinic_visits.iloc[:730].mean() * 2) & (df.date >= start), "date"]
    if len(clinic_late):
        print("Clinic-only 2x spike :", clinic_late.iloc[0].date(), "(what a hospital-surge approach would see)")

    alerts = df.loc[df.alert == 1, ["date", "composite_z"] + SIGNALS]
    alerts.to_csv(OUT / "outbreak_alerts.csv", index=False)

    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for s in SIGNALS:
        ax[0].plot(df.date, df[s] / df[s].iloc[:730].mean(), label=s, lw=.8)
    ax[0].set_ylabel("Signal (relative to normal)"); ax[0].legend()
    ax[1].plot(df.date, df.composite_z, color="k", lw=.8)
    ax[1].axhline(3, color="orange", ls="--", label="alert threshold")
    ax[1].fill_between(df.date, 0, df.composite_z.max(), where=df.alert == 1, color="red", alpha=.25, label="ALERT")
    ax[1].axvline(start, color="grey", ls=":", label="true outbreak start")
    ax[1].set_ylabel("Composite anomaly score"); ax[1].legend()
    plt.tight_layout(); plt.savefig(OUT / "outbreak_plot.png", dpi=120)
    print("Saved outputs to", OUT)


if __name__ == "__main__":
    main()
