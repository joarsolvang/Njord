import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import logging
    from datetime import date

    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    from njord import elexon

    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("njord").setLevel(logging.INFO)  # set to DEBUG to also see cache hits

    return date, elexon, go, mo, pd, px


@app.cell
def _(mo):
    mo.md(r"""
    # 01 · First look at GB market data (Elexon BMRS)

    Open with `uv run marimo edit notebooks/01_explore_elexon.py`. The first run downloads about a month of
    data into `~/.njord/`; later runs read it from there.

    **Glossary**

    - **Settlement period (SP):** the half-hour that GB power is traded and settled in.
    - **Market Index Price (MID):** volume-weighted price of short-term trades on the exchanges for each SP.
      It's the closest public proxy for the intraday "market price".
    - **System price / imbalance price:** what you pay or receive for being out of balance (producing more
      or less than you sold). GB uses a single cash-out price: one price for both directions in each SP.
    - **Net Imbalance Volume (NIV):** + means the system was short and NESO had to buy energy; - means it
      was long. A short system usually means a high system price.
    """)
    return


@app.cell
def _(date, elexon, mo):
    START, END = date(2026, 10, 1), date(2026, 10, 31)

    client = elexon.ElexonClient()
    system = client.system_prices(START, END).set_index("startTime")

    _mid = client.market_index(START, END)
    mid = _mid[_mid.dataProvider == "APXMIDP"].set_index("startTime").sort_index()  # N2EX index is all zeros

    fuel = client.generation_by_fuel(START, END).pivot_table(index="startTime", columns="fuelType", values="generation")

    _windfor = client.wind_forecast(START, END)
    windfor = _windfor.assign(lead_h=(_windfor.startTime - _windfor.publishTime).dt.total_seconds() / 3600)

    mo.md(
        f"Loaded {START} to {END}: **{len(system)}** system prices, **{len(mid)}** MID, "
        f"**{len(fuel)}** fuel-mix periods, **{len(windfor)}** wind forecast rows."
    )
    return fuel, mid, system, windfor


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Prices: market index vs imbalance

    The gap between the two prices is where a battery earns money. The imbalance price is much spikier
    because it reflects the last, most expensive actions NESO took to balance the system in each period.
    """)
    return


@app.cell
def _(go, mid, system):
    _fig = go.Figure()
    _fig.add_scatter(x=system.index, y=system.systemSellPrice, name="System (imbalance) price", line_width=1)
    _fig.add_scatter(x=mid.index, y=mid.price, name="Market index price (APX)", line_width=1.5)
    _fig.update_layout(yaxis_title="£/MWh", hovermode="x unified", title="GB prices per settlement period")
    return


@app.cell
def _(px, system):
    # Short system -> high price
    px.scatter(
        system,
        x="netImbalanceVolume",
        y="systemSellPrice",
        opacity=0.4,
        labels={"netImbalanceVolume": "NIV (MWh, + = system short)", "systemSellPrice": "System price (£/MWh)"},
        title="Imbalance price vs net imbalance volume",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Generation mix and what it does to price

    FUELHH is outturn by fuel type from transmission-metered units, in MW averaged over each SP.
    Embedded (distribution-connected) wind and solar aren't in here; they show up as lower demand instead.
    """)
    return


@app.cell
def _(fuel, px):
    _order = fuel.mean().sort_values(ascending=False).index
    px.area(fuel[_order], labels={"value": "MW", "startTime": ""}, title="Generation by fuel type")
    return


@app.cell
def _(fuel, mid, px):
    px.scatter(
        mid[["price"]].join(fuel["WIND"]).dropna(),
        x="WIND",
        y="price",
        opacity=0.4,
        labels={"WIND": "Transmission wind (MW)", "price": "MID (£/MWh)"},
        title="More wind -> lower price (merit-order effect)",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Daily price shape: the battery's bread and butter

    A 2-hour battery (50 MW / 100 MWh) can roughly buy the cheapest 2 hours and sell the most expensive 2
    hours each day. That spread, minus 12% round-trip losses, is a crude upper bound on daily arbitrage value.
    """)
    return


@app.cell
def _(mid):
    local_price = mid.price.tz_convert("Europe/London")
    return (local_price,)


@app.cell
def _(local_price, px):
    _p = local_price
    _hh = _p.groupby([_p.index.date, _p.index.hour + _p.index.minute / 60]).mean().unstack(0)
    px.line(
        _hh, labels={"index": "UK local hour", "value": "£/MWh"}, title="Daily price profiles (one line per day)"
    ).update_traces(line_width=1, opacity=0.4).update_layout(showlegend=False)
    return


@app.cell
def _(local_price, px):
    _p = local_price
    _daily = _p.groupby(_p.index.date).agg(lambda s: s.nlargest(4).mean() - s.nsmallest(4).mean() / 0.88)
    px.bar(
        _daily, labels={"index": "", "value": "£/MWh"}, title="Top-4 SP minus bottom-4 SP price, after losses"
    ).update_layout(showlegend=False)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Wind forecasts are point-in-time data

    NESO republishes its national wind forecast many times a day. Each row has `publishTime` (when it was
    available) and `startTime` (the hour it's for). In a backtest you may only use forecasts published
    before the decision time. Here is how the forecast for each hour changes as it gets closer.
    """)
    return


@app.cell
def _(fuel):
    outturn = fuel["WIND"].resample("1h").mean()
    return (outturn,)


@app.cell
def _(go, outturn, windfor):
    _fig = go.Figure()
    for _lo, _hi in [(0, 6), (12, 24), (24, 48)]:
        _f = windfor[windfor.lead_h.between(_lo, _hi)].groupby("startTime").generation.mean()
        _fig.add_scatter(x=_f.index, y=_f, name=f"forecast, {_lo}-{_hi} h ahead", line_width=1)
    _fig.add_scatter(x=outturn.index, y=outturn, name="outturn (FUELHH)", line={"color": "black", "width": 2})
    _fig.update_layout(yaxis_title="MW", hovermode="x unified", title="National wind: forecast vs outturn by lead time")
    return


@app.cell
def _(outturn, px, windfor):
    # Forecast error grows with lead time
    _err = windfor.join(outturn.rename("actual"), on="startTime").dropna()
    _err = _err[_err.lead_h >= 0].assign(
        abs_err=(_err.generation - _err.actual).abs(), lead_bin=(_err.lead_h // 6 * 6).astype(int)
    )
    px.box(
        _err,
        x="lead_bin",
        y="abs_err",
        points=False,
        labels={"lead_bin": "Lead time (h, 6 h bins)", "abs_err": "|forecast - outturn| (MW)"},
        title="Wind forecast error vs lead time",
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
