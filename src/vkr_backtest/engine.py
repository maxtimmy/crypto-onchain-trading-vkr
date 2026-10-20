"""Causal, minute-open spot simulation with Decimal account bookkeeping.

Signals are consumed once at their scheduled execution minute. Missing future
prices never determine whether an earlier entry is accepted.
"""

import math
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation, ROUND_FLOOR, localcontext
from numbers import Integral, Real


MINUTE = 60_000
HOUR = 60 * MINUTE
DAY = 24 * HOUR
ZERO = Decimal(0)
ONE = Decimal(1)
BPS = Decimal(10_000)
MODES = ("fixed_horizon", "buy_hold", "cash")


def _decimal(value, name, positive=False):
    if isinstance(value, bool):
        raise ValueError(name + " must be a finite decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(name + " must be a finite decimal") from error
    if not result.is_finite() or (result <= 0 if positive else result < 0):
        raise ValueError(name + " must be " + ("positive" if positive else "nonnegative") + " and finite")
    return result


def _time(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value % MINUTE:
        raise ValueError(name + " must be integer UTC milliseconds on a minute boundary")
    return int(value)


def _inputs(signals, quotes, config, costs, mode):
    if mode not in MODES:
        raise ValueError("Unsupported simulation mode")
    if not isinstance(config, Mapping) or not isinstance(costs, Mapping) or not isinstance(quotes, Mapping):
        raise ValueError("config, costs and quotes must be mappings")
    start, end, terminal = [_time(config[name], name) for name in ("start_time", "end_time", "terminal_time")]
    if end <= start or terminal != end - MINUTE:
        raise ValueError("terminal_time must equal end_time minus one minute in a nonempty period")
    parsed = {"start_time": start, "end_time": end, "terminal_time": terminal}
    for name in ("initial_cash", "allocation", "qty_step"):
        parsed[name] = _decimal(config[name], name, positive=True)
    parsed["min_notional"] = _decimal(config["min_notional"], "min_notional")
    if parsed["allocation"] > 1:
        raise ValueError("allocation cannot exceed available cash")
    holding = config["holding_minutes"]
    if isinstance(holding, bool) or not isinstance(holding, Integral) or holding <= 0:
        raise ValueError("holding_minutes must be a positive integer")
    parsed["holding_minutes"] = int(holding)
    rates = {name: _decimal(costs[name], name) / BPS
             for name in ("fee_bps", "half_spread_bps", "slippage_bps")}
    if rates["fee_bps"] >= 1 or rates["half_spread_bps"] + rates["slippage_bps"] >= 1:
        raise ValueError("Execution costs must leave positive sell proceeds")
    clean_quotes = {_time(time, "quote time"): _decimal(price, "quote open", positive=True)
                    for time, price in quotes.items()}
    if not isinstance(signals, list):
        raise ValueError("signals must be a list of scheduled signal mappings")
    scheduled = {}
    for signal in signals:
        if not isinstance(signal, Mapping):
            raise ValueError("Each signal must be a mapping")
        clean = {name: _time(signal[name], name)
                 for name in ("decision_time", "signal_time", "execution_time")}
        if not clean["decision_time"] <= clean["signal_time"] <= clean["execution_time"]:
            raise ValueError("Signal times must be causal")
        time = clean["execution_time"]
        if not start <= time < end or time in scheduled:
            raise ValueError("Execution times must be unique and inside the simulation period")
        if not isinstance(signal["enter"], bool):
            raise ValueError("Signal enter must be boolean")
        score = signal.get("score")
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score):
                raise ValueError("Signal score must be finite or None")
            score = float(score)
        clean.update({"enter": signal["enter"], "score": score})
        if mode == "buy_hold" and clean["enter"] and time != start + 2 * MINUTE:
            raise ValueError("buy_hold enters only at start_time plus two minutes")
        scheduled[time] = clean
    return scheduled, clean_quotes, parsed, rates


def _json_native(value):
    if isinstance(value, Decimal):
        text = format(value, "f")
        return "0" if value == 0 else text.rstrip("0").rstrip(".") if "." in text else text
    if isinstance(value, dict):
        return {key: _json_native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_native(item) for item in value]
    return value


def _curve_append(curve, row):
    if curve and curve[-1]["time"] == row["time"]:
        if curve[-1]["kind"] != "initial":
            curve[-1] = row
    else:
        curve.append(row)


def _sharpe(daily):
    observations = [row for row in daily if row["kind"] in ("initial", "daily_close")]
    returns = [float(right["equity"] / left["equity"] - ONE)
               for left, right in zip(observations, observations[1:]) if left["equity"] > 0]
    if len(returns) < 2 or min(returns) == max(returns):
        return None
    mean = math.fsum(returns) / len(returns)
    variance = math.fsum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    return math.sqrt(365) * mean / math.sqrt(variance) if variance > 0 else None


def simulate(signals, quotes, config, costs, mode="fixed_horizon"):
    """Return JSON-native trades, curves and complete/provisional results.

    Quotes contain only independently validated minute OPEN prices. The caller
    supplies entry decisions; this engine never chooses a model or threshold.
    Quantities are floored to ``qty_step`` and all account amounts use Decimal.
    """
    with localcontext() as context:
        context.prec = 80
        scheduled, quotes, config, rates = _inputs(signals, quotes, config, costs, mode)
        return _json_native(_simulate(scheduled, quotes, config, rates, mode))


def _simulate(scheduled, quotes, config, rates, mode):
    start, end, terminal = [config[name] for name in ("start_time", "end_time", "terminal_time")]
    initial = cash = config["initial_cash"]
    quantity, last_price, position = ZERO, None, None
    fee_rate, spread, slip = [rates[name] for name in ("fee_bps", "half_spread_bps", "slippage_bps")]
    fee_total = spread_total = slip_total = turnover_notional = ZERO
    trades, hourly, daily = [], [], []
    counters = {name: 0 for name in (
        "signals_total", "enter_signals", "ignored_while_holding", "entry_missing_quote",
        "entry_calendar_rejected", "entry_min_notional_rejected", "entry_insufficient_cash",
        "exit_missing_quote_minutes", "exit_min_notional_rejected", "delayed_exits",
        "entries", "closed_trades", "held_minutes", "total_minutes", "stale_minutes")}
    counters["signals_total"] = len(scheduled)
    counters["enter_signals"] = sum(signal["enter"] for signal in scheduled.values())
    counters["total_minutes"] = (end - start) // MINUTE
    peak = initial
    max_drawdown = ZERO
    initial_row = {"time": start, "equity": initial, "cash": cash, "quantity": ZERO,
                   "reference_price": None, "stale": False, "kind": "initial"}
    hourly.append(dict(initial_row))
    daily.append(dict(initial_row))
    last_row = initial_row
    for time in range(start, end, MINUTE):
        reference = quotes.get(time)
        if reference is not None:
            last_price = reference
        if position is not None and (time >= position["due_time"] or time == terminal):
            if reference is None:
                counters["exit_missing_quote_minutes"] += 1
            else:
                fill = reference * (ONE - spread - slip)
                notional = quantity * fill
                if notional < config["min_notional"]:
                    counters["exit_min_notional_rejected"] += 1
                else:
                    fee = notional * fee_rate
                    cash += notional - fee
                    sell_spread, sell_slip = quantity * reference * spread, quantity * reference * slip
                    fee_total += fee
                    spread_total += sell_spread
                    slip_total += sell_slip
                    turnover_notional += notional
                    pnl = notional - fee - position["entry_notional"] - position["entry_fee"]
                    delayed = time > position["due_time"]
                    trade = dict(position, exit_time=time, exit_reference_price=reference,
                                 exit_fill_price=fill, exit_notional=notional, exit_fee=fee,
                                 spread_cost=position["entry_spread_cost"] + sell_spread,
                                 slippage_cost=position["entry_slippage_cost"] + sell_slip,
                                 net_pnl=pnl, exit_delayed=delayed,
                                 exit_reason="terminal" if time == terminal else "pending" if delayed else "scheduled")
                    trades.append(trade)
                    counters["closed_trades"] += 1
                    counters["delayed_exits"] += int(delayed)
                    quantity, position = ZERO, None
        signal = scheduled.get(time)
        if mode != "cash" and signal is not None and signal["enter"]:
            due = time + config["holding_minutes"] * MINUTE if mode == "fixed_horizon" else terminal
            if due > terminal or time >= terminal:
                counters["entry_calendar_rejected"] += 1
            elif position is not None:
                counters["ignored_while_holding"] += 1
            elif reference is None:
                counters["entry_missing_quote"] += 1
            else:
                fill = reference * (ONE + spread + slip)
                available = cash * config["allocation"]
                step = config["qty_step"]
                bought = ((available / (fill * (ONE + fee_rate))) / step).to_integral_value(rounding=ROUND_FLOOR) * step
                notional = bought * fill
                fee = notional * fee_rate
                if bought <= 0 or notional + fee > cash:
                    counters["entry_insufficient_cash"] += 1
                elif notional < config["min_notional"]:
                    counters["entry_min_notional_rejected"] += 1
                else:
                    cash -= notional + fee
                    quantity = bought
                    buy_spread, buy_slip = quantity * reference * spread, quantity * reference * slip
                    fee_total += fee
                    spread_total += buy_spread
                    slip_total += buy_slip
                    turnover_notional += notional
                    counters["entries"] += 1
                    position = {
                        "trade_id": counters["entries"], "score": signal["score"],
                        "entry_decision_time": signal["decision_time"], "entry_signal_time": signal["signal_time"],
                        "entry_execution_time": time, "entry_time": time, "due_time": due,
                        "entry_reference_price": reference, "entry_fill_price": fill,
                        "quantity": quantity, "entry_notional": notional, "entry_fee": fee,
                        "entry_spread_cost": buy_spread, "entry_slippage_cost": buy_slip,
                    }
        if cash < 0 or quantity < 0:
            raise RuntimeError("Spot account cannot borrow cash or sell short")
        held = position is not None
        stale = held and reference is None
        if held and last_price is None:
            raise RuntimeError("An open position must have an observed entry price")
        equity = cash + quantity * last_price if held else cash
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak)
        counters["held_minutes"] += int(held)
        counters["stale_minutes"] += int(stale)
        last_row = {"time": time, "equity": equity, "cash": cash, "quantity": quantity,
                    "reference_price": last_price if held else reference,
                    "stale": stale, "kind": "hourly"}
        if (time - start) % HOUR == 0:
            _curve_append(hourly, dict(last_row))
        if time % DAY == DAY - MINUTE:
            _curve_append(daily, dict(last_row, kind="daily_close"))
    _curve_append(hourly, dict(last_row, kind="terminal"))
    # Preserve a 23:59 close as such, so it remains a daily Sharpe observation.
    if not daily or daily[-1]["time"] != terminal:
        _curve_append(daily, dict(last_row, kind="terminal"))
    flat = position is None
    realized = sum((trade["net_pnl"] for trade in trades), ZERO)
    identity_residual = initial + realized - cash if flat else None
    if flat and identity_residual != ZERO:
        raise RuntimeError("Closed spot trades do not reconcile with the cash account")
    result = {
        "status": "COMPLETE" if flat else "INCOMPLETE", "metrics": None,
        "trades": trades, "equity_hourly": hourly, "equity_daily": daily, "counters": counters,
        "final_state": {"time": terminal, "cash": cash, "quantity": quantity, "last_price": last_price,
                        "equity": last_row["equity"], "stale": last_row["stale"], "position": position},
        "accounting": {"initial_cash": initial, "realized_net_pnl": realized,
                       "final_cash": cash, "identity_residual": identity_residual, "flat": flat},
    }
    if flat:
        days = (end - start) / DAY
        net_return = float(cash / initial - ONE)
        annual_log = math.log(float(cash / initial)) * 365 / days if cash > 0 else None
        try:
            annualized = math.expm1(annual_log) if annual_log is not None else -1.0
        except OverflowError:
            annualized = None
        profits = sum((trade["net_pnl"] for trade in trades if trade["net_pnl"] > 0), ZERO)
        losses = -sum((trade["net_pnl"] for trade in trades if trade["net_pnl"] < 0), ZERO)
        result["metrics"] = {
            "initial_equity": initial, "final_equity": cash, "net_return": net_return,
            "annualized_return": annualized, "daily_sharpe": _sharpe(daily),
            "max_drawdown": float(max_drawdown), "turnover": float(turnover_notional / initial),
            "exposure": counters["held_minutes"] / counters["total_minutes"],
            "fees_paid": fee_total, "spread_cost": spread_total, "slippage_cost": slip_total,
            "closed_trades": len(trades),
            "win_rate": sum(trade["net_pnl"] > 0 for trade in trades) / len(trades) if trades else None,
            "profit_factor": float(profits / losses) if losses else None, "days": days,
        }
    return result
