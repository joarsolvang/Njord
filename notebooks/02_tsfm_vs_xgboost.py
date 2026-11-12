import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import logging
    from datetime import date

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    from njord import elexon

    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("njord").setLevel(logging.INFO)  # set to DEBUG to also see cache hits

    return date, elexon, go, mo, np, pd, px


@app.cell
def _(mo):
    mo.md(r"""
    # 02 · Time series foundation model vs XGBoost

    **Task:** each day at 00:00 UTC, forecast the next 48 half-hourly market index prices (MID).

    - **Seasonal naive:** tomorrow's price equals today's. Any model has to beat this.
    - **XGBoost:** gradient-boosted trees on hand-made features (time of day, weekday, price lags, NESO
      forecasts), trained once on 2025 to July 2026. Quantile loss gives P10/P50/P90.
    - **Chronos-2** (Amazon, 120M params): a transformer pretrained on a large set of time series. **Zero-shot:**
      it's never trained on GB prices. At each origin it just reads the recent price history (and optionally the
      NESO forecasts as covariates) and outputs quantiles.

    **NESO forecasts used** (all taken as they stood at the origin):

    - **Wind** (`WINDFOR`): transmission-connected wind.
    - **Demand** (`NDF`): transmission demand. Embedded solar and wind are already subtracted, which is why
      there's no separate solar feature.
    - **Residual demand** = demand - wind: what gas and other flexible plant must cover. Gas usually sets the
      price, so this is the classic price driver.
    - **Indicated margin** (`MELNGC`): spare generation capacity over expected demand. A thin margin means
      scarcity and price spikes, usually in the winter evening peak.
    - **Indicated imbalance** (`IMBALNGC`): generators' planned output minus demand. Negative means more plant
      still has to be bought, usually at a higher price.

    The question: how much does each approach get you for how little effort, and do the uncertainty bands mean
    anything? Interval quality matters later for the battery, where bidding depends on the spread between
    peak and off-peak prices.

    **Why wind?** In GB, wind sets much of the price. Windy hours push gas plants off the margin and prices
    drop, sometimes below zero. NESO publishes its wind forecast many times a day. We use the latest version
    published *before* each forecast origin, so no model sees later information.

    The first run downloads about two years of data (around 200 API calls, then cached), and Chronos-2
    inference takes a few minutes on CPU.
    """)
    return


@app.cell
def _(date, elexon):
    START, TRAIN_END, END = date(2025, 1, 1), date(2026, 7, 31), date(2026, 11, 10)

    client = elexon.ElexonClient()
    _mid = client.market_index(START, END)
    price = (
        _mid[_mid.dataProvider == "APXMIDP"]
        .drop_duplicates("startTime")
        .set_index("startTime")
        .price.sort_index()
        .asfreq("30min")
        .interpolate(limit=4)  # shortcut: fills short gaps; long gaps stay NaN and those origins are skipped
    )
    # NESO forecasts, each republished every 30-60 min up to ~40 h ahead. Columns: publishTime, startTime, value.
    _cols = ["publishTime", "startTime", "value"]
    forecasts = {
        "wind_fc": client.wind_forecast(START, END).rename(columns={"generation": "value"})[_cols],
        # National demand forecast: transmission demand, so embedded (mostly solar) generation is already netted off
        "demand_fc": client.demand_forecast(START, END).rename(columns={"demand": "value"})[_cols],
        # Indicated margin: available generation minus demand. Small margin -> scarcity -> price spikes.
        "margin_fc": client.margin_forecast(START, END).rename(columns={"margin": "value"})[_cols],
        # Indicated imbalance: generators' planned output minus demand. Negative -> more plant still needs to be bought.
        "imbal_fc": client.imbalance_forecast(START, END).rename(columns={"imbalance": "value"})[_cols],
    }
    return END, START, TRAIN_END, forecasts, price


@app.cell
def _(forecasts, np, pd):
    def _to_halfhours(s: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
        s = s.sort_index()
        return s.reindex(s.index.union(index)).interpolate(method="time").ffill().reindex(index).to_numpy()

    def asof(name: str, origin: pd.Timestamp, index: pd.DatetimeIndex, df: pd.DataFrame | None = None) -> np.ndarray:
        """Forecast `name` for `index` as it stood at `origin`: latest run with publishTime <= origin."""
        df = forecasts[name] if df is None else df
        _w = df[(df.publishTime <= origin) & (df.startTime >= index[0] - pd.Timedelta("1h"))]
        _w = _w.sort_values("publishTime").drop_duplicates("startTime", keep="last")
        return _to_halfhours(_w.set_index("startTime").value, index)

    def _nowcast(df: pd.DataFrame) -> pd.Series:
        """For each time, the latest forecast published before it. Used as Chronos's past covariates."""
        n = df[df.publishTime <= df.startTime].sort_values("publishTime").drop_duplicates("startTime", keep="last")
        n = n.set_index("startTime").value
        idx = pd.date_range(n.index.min(), n.index.max(), freq="30min")
        return pd.Series(_to_halfhours(n, idx), idx)

    nowcasts = pd.DataFrame({k: _nowcast(v) for k, v in forecasts.items()})
    nowcasts["residual_fc"] = nowcasts.demand_fc - nowcasts.wind_fc

    # Self-check (no lookahead): dropping every run published after the origin must not change the answer.
    _o = pd.Timestamp("2026-08-01", tz="UTC")
    _i = pd.date_range(_o, periods=48, freq="30min")
    for _k, _df in forecasts.items():
        assert (_df.publishTime > _o).any(), _k
        assert np.allclose(asof(_k, _o, _i), asof(_k, _o, _i, _df[_df.publishTime <= _o]), equal_nan=True), _k
    return asof, nowcasts


@app.cell
def _(END, START, asof, forecasts, mo, pd, price):
    H = 48

    def make_rows(origin: pd.Timestamp) -> pd.DataFrame | None:
        """One row per target half-hour. Every input is known at `origin`."""
        idx = pd.date_range(origin, periods=H, freq="30min")
        past = price[: origin - pd.Timedelta("30min")]
        if len(past) < 7 * H or past.iloc[-7 * H :].isna().any():
            return None
        local = idx.tz_convert("Europe/London")
        d = pd.DataFrame(
            {
                "origin": origin,
                "hod": local.hour + local.minute / 60,
                "dow": local.day_of_week,
                "lag_1d": past.iloc[-H:].to_numpy(),  # same half-hour yesterday
                "lag_7d": past.iloc[-7 * H : -6 * H].to_numpy(),  # same half-hour a week ago
                "mean_1d": past.iloc[-H:].mean(),
                **{k: asof(k, origin, idx) for k in forecasts},
                "y": price.reindex(idx).to_numpy(),
            },
            index=idx,
        )
        # Residual demand: what's left for gas and other dispatchable plant after wind. Gas usually sets the price.
        return d.assign(residual_fc=d.demand_fc - d.wind_fc)

    _origins = pd.date_range(pd.Timestamp(START, tz="UTC") + pd.Timedelta("8D"), pd.Timestamp(END, tz="UTC"))
    rows = pd.concat([r for o in _origins if (r := make_rows(o)) is not None])
    rows = rows[rows.y.notna()]
    COVARIATES = [*forecasts, "residual_fc"]
    FEATURES = ["hod", "dow", "lag_1d", "lag_7d", "mean_1d", *COVARIATES]
    mo.md(f"{rows.origin.nunique()} daily origins, {len(rows)} target half-hours.")
    return COVARIATES, FEATURES, H, rows


@app.cell
def _(FEATURES, TRAIN_END, pd, rows):
    import xgboost as xgb

    QUANTILES = [0.1, 0.5, 0.9]
    _is_train = rows.origin < pd.Timestamp(TRAIN_END, tz="UTC")
    train, test = rows[_is_train], rows[~_is_train].copy()

    # shortcut: trained once; a fair backtest would refit as the test period advances (e.g. monthly).
    _model = xgb.XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=QUANTILES, n_estimators=400, learning_rate=0.05, max_depth=6
    )

    # Trees can't extrapolate: test prices (autumn 2026) sit above the whole training range, so a model on raw
    # prices is stuck near the old level. Predict the deviation from yesterday's mean, with lags rebased and the
    # level itself dropped (in 2025 a high level always meant "falls back", which misleads in 2026).
    REL = [f for f in FEATURES if f != "mean_1d"]

    def _rel(d):
        return d[REL].assign(lag_1d=d.lag_1d - d.mean_1d, lag_7d=d.lag_7d - d.mean_1d)

    _model.fit(_rel(train), train.y - train.mean_1d)
    test[["xgb_p10", "xgb_p50", "xgb_p90"]] = _model.predict(_rel(test)) + test.mean_1d.to_numpy()[:, None]
    test["naive"] = test.lag_1d
    xgb_importance = pd.Series(_model.feature_importances_, REL).sort_values()
    return QUANTILES, test, xgb_importance


@app.cell
def _(COVARIATES, H, QUANTILES, nowcasts, pd, price, test):
    import torch
    from chronos import Chronos2Pipeline

    _pipe = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2", device_map="cuda" if torch.cuda.is_available() else "cpu"
    )
    CONTEXT = 2048  # ~6 weeks of half-hours

    def run_chronos(covariates: list[str]) -> pd.DataFrame:
        ctx, fut = [], []
        for origin, rows_o in test.groupby("origin"):
            hist = price[: origin - pd.Timedelta("30min")].iloc[-CONTEXT:].interpolate(limit_direction="both")
            c = pd.DataFrame({"item_id": str(origin), "timestamp": hist.index.tz_localize(None), "target": hist.values})
            idx = pd.date_range(origin, periods=H, freq="30min")  # full horizon even if some actuals are missing
            f = pd.DataFrame({"item_id": str(origin), "timestamp": idx.tz_localize(None)})
            for k in covariates:
                # Past: the short-lead forecast for each time; future: the forecast as it stood at the origin.
                c[k] = nowcasts[k].reindex(hist.index).ffill().bfill().to_numpy()
                f[k] = rows_o[k].reindex(idx).ffill().bfill().to_numpy()
            ctx.append(c)
            fut.append(f)
        out = _pipe.predict_df(
            pd.concat(ctx),
            future_df=pd.concat(fut) if covariates else None,
            prediction_length=H,
            quantile_levels=QUANTILES,
            freq="30min",
        )
        return out.assign(timestamp=out.timestamp.dt.tz_localize("UTC")).set_index("timestamp")[["0.1", "0.5", "0.9"]]

    MODELS = {
        "xgb": "XGBoost (all features)",
        "chr": "Chronos-2 (price only)",
        "chrw": "Chronos-2 + wind fc",
        "chra": "Chronos-2 + all fc",
    }
    results = test.copy()
    for _key, _covs in [("chr", []), ("chrw", ["wind_fc"]), ("chra", COVARIATES)]:
        _fc = run_chronos(_covs)
        results[[f"{_key}_p10", f"{_key}_p50", f"{_key}_p90"]] = _fc.reindex(results.index).to_numpy()
    return MODELS, results


@app.cell
def _(mo):
    mo.md(r"""
    ## Scores

    - **MAE** (£/MWh) of the median forecast. Lower is better.
    - **Pinball loss**, averaged over P10/P50/P90: the standard score for quantile forecasts. It rewards bands
      that are both narrow and right.
    - **Coverage of P10-P90**: should be about 80%. Lower means overconfident, higher means bands too wide.
    - **Spread MAE**: error in each day's (max - min) price. For a battery this matters more than plain MAE,
      because the day's shape, not its level, is what pays.
    """)
    return


@app.cell
def _(MODELS, np, pd, results):
    def pinball(y: pd.Series, q: pd.Series, a: float) -> float:
        d = y - q
        return float(np.mean(np.maximum(a * d, (a - 1) * d)))

    def spread_mae(col: str) -> float:
        g = results.groupby("origin")
        return float(((g[col].max() - g[col].min()) - (g.y.max() - g.y.min())).abs().mean())

    _y = results.y
    _rows = {"Seasonal naive": {"MAE": (results.naive - _y).abs().mean(), "Spread MAE": spread_mae("naive")}}
    for _key, _label in MODELS.items():
        _rows[_label] = {
            "MAE": (results[f"{_key}_p50"] - _y).abs().mean(),
            "Pinball": np.mean([pinball(_y, results[f"{_key}_p{int(a * 100)}"], a) for a in (0.1, 0.5, 0.9)]),
            "P10-P90 coverage": results[f"{_key}_p10"].le(_y).mul(results[f"{_key}_p90"].ge(_y)).mean(),
            "Spread MAE": spread_mae(f"{_key}_p50"),
        }
    scores = pd.DataFrame(_rows).T.round(3)
    scores
    return (scores,)


@app.cell
def _(MODELS, mo, results):
    day = mo.ui.dropdown(
        {str(o.date()): o for o in results.origin.unique()}, value=str(results.origin.iloc[-1].date()), label="Day"
    )
    model = mo.ui.dropdown({v: k for k, v in MODELS.items()}, value=MODELS["chra"], label="Bands for")
    mo.hstack([day, model], justify="start")
    return day, model


@app.cell
def _(MODELS, day, go, model, results):
    _d = results[results.origin == day.value]
    _k = model.value
    _fig = go.Figure()
    _fig.add_scatter(x=_d.index, y=_d[f"{_k}_p90"], line_width=0, showlegend=False, hoverinfo="skip")
    _fig.add_scatter(
        x=_d.index, y=_d[f"{_k}_p10"], fill="tonexty", line_width=0, name="P10-P90", fillcolor="rgba(99,110,250,0.2)"
    )
    _fig.add_scatter(x=_d.index, y=_d.y, name="Actual MID", line={"color": "black", "width": 2})
    _fig.add_scatter(x=_d.index, y=_d.naive, name="Seasonal naive", line={"dash": "dot", "width": 1})
    for _k2, _name in MODELS.items():
        _fig.add_scatter(x=_d.index, y=_d[f"{_k2}_p50"], name=f"{_name} P50", line_width=1.5)
    _fig.update_layout(yaxis_title="£/MWh", hovermode="x unified", title="Day-ahead forecast vs actual")
    return


@app.cell
def _(MODELS, pd, px, results):
    _abs = {"Naive": results.naive, **{v: results[f"{k}_p50"] for k, v in MODELS.items()}}
    _err = pd.DataFrame({k: (v - results.y).abs() for k, v in _abs.items()}).groupby(results.origin).mean()
    px.line(
        _err.rolling(7).mean(), labels={"value": "MAE £/MWh (7-day rolling)", "origin": ""}, title="Error over time"
    )
    return


@app.cell
def _(px, xgb_importance):
    px.bar(xgb_importance, orientation="h", title="XGBoost feature importance (gain)").update_layout(showlegend=False)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Things to try next

    - **Lead time:** the real day-ahead auction closes at 11:00 UK time on the day before delivery, so a true
      day-ahead forecast runs 13 to 37 hours ahead, not 0 to 24.
    - **Refit XGBoost monthly** in the test period. Chronos sees new data at every origin, so XGBoost is at a
      disadvantage right now.
    - **Fine-tune Chronos-2** on GB prices, or try TimesFM 2.5 / Moirai 2 through the same harness.
    - **Imbalance price:** spikier and harder to forecast. Pretrained models may cope better there, or
      worse.
    """)
    return


if __name__ == "__main__":
    app.run()
