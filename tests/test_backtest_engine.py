"""Causal executions and exact, hand-calculated Decimal account identities."""

import copy
import json
import math
import unittest
from datetime import datetime
from decimal import Decimal

from src.vkr_backtest.engine import DAY, HOUR, MINUTE, simulate


def utc_ms(value):
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


class BacktestEngineTests(unittest.TestCase):
    def setUp(self):
        self.start = utc_ms("2024-01-01T00:00:00Z")
        self.config = {"start_time": self.start, "end_time": self.start + 10 * MINUTE,
                       "terminal_time": self.start + 9 * MINUTE, "initial_cash": "10000",
                       "holding_minutes": 2, "allocation": "1", "qty_step": "0.000001", "min_notional": "10"}
        self.costs = {"fee_bps": 10, "half_spread_bps": 1, "slippage_bps": 1}
        self.free = {"fee_bps": 0, "half_spread_bps": 0, "slippage_bps": 0}
        self.quotes = {self.start + index * MINUTE: "100" for index in range(10)}

    def signal(self, minute, enter=True, score=.01):
        time = self.start + minute * MINUTE
        return {"decision_time": time - 2 * MINUTE, "signal_time": time - MINUTE,
                "execution_time": time, "enter": enter, "score": score}

    def run_sim(self, signals=None, quotes=None, config=None, costs=None, mode="fixed_horizon"):
        return simulate([self.signal(2)] if signals is None else signals,
                        self.quotes if quotes is None else quotes,
                        self.config if config is None else config,
                        self.costs if costs is None else costs, mode)

    def test_manual_fees_fill_quantity_and_exact_cash_identity(self):
        result = self.run_sim()
        trade = result["trades"][0]
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(trade["entry_fill_price"], "100.02")
        self.assertEqual(trade["exit_fill_price"], "99.98")
        self.assertEqual(trade["quantity"], "99.880123")
        self.assertEqual(trade["entry_notional"], "9990.00990246")
        self.assertEqual(trade["entry_fee"], "9.99000990246")
        self.assertEqual(trade["exit_notional"], "9986.01469754")
        self.assertEqual(trade["exit_fee"], "9.98601469754")
        self.assertEqual(trade["net_pnl"], "-23.97122952")
        self.assertEqual(result["accounting"]["identity_residual"], "0")
        self.assertEqual(result["final_state"]["cash"], "9976.02877048")
        self.assertEqual(Decimal(result["final_state"]["cash"]),
                         Decimal("10000") + Decimal(trade["net_pnl"]))
        self.assertEqual(result["metrics"]["fees_paid"], "19.9760246")
        self.assertEqual(result["metrics"]["spread_cost"], "1.99760246")
        self.assertEqual(result["metrics"]["slippage_cost"], "1.99760246")
        self.assertEqual(result["metrics"]["turnover"], float(Decimal("19976.0246") / 10000))
        self.assertEqual(result["metrics"]["exposure"], .2)
        self.assertEqual(result["metrics"]["win_rate"], 0)
        self.assertEqual(result["metrics"]["profit_factor"], 0)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_insufficient_cash_and_min_notional_do_not_borrow(self):
        for overrides, counter in [({"initial_cash": "0.000001"}, "entry_insufficient_cash"),
                                   ({"initial_cash": "5"}, "entry_min_notional_rejected")]:
            with self.subTest(overrides=overrides):
                result = self.run_sim(config=dict(self.config, **overrides))
                self.assertEqual(result["trades"], [])
                self.assertEqual(result["counters"][counter], 1)
                self.assertEqual(result["final_state"]["cash"], overrides["initial_cash"])
                self.assertEqual(result["final_state"]["quantity"], "0")

    def test_exit_before_entry_at_same_minute_and_held_signals_not_replayed(self):
        signals = [self.signal(2), self.signal(3), self.signal(4)]
        result = self.run_sim(signals=signals, costs=self.free)
        self.assertEqual([trade["entry_time"] for trade in result["trades"]],
                         [self.start + 2 * MINUTE, self.start + 4 * MINUTE])
        self.assertEqual(result["trades"][0]["exit_time"], result["trades"][1]["entry_time"])
        self.assertEqual(result["counters"]["ignored_while_holding"], 1)
        self.assertEqual(result["counters"]["entries"], 2)

    def test_missing_exact_entry_is_cancelled_without_future_replay(self):
        del self.quotes[self.start + 2 * MINUTE]
        result = self.run_sim()
        self.assertEqual(result["counters"]["entry_missing_quote"], 1)
        self.assertEqual(result["trades"], [])
        self.assertEqual(result["final_state"]["cash"], "10000")

    def test_missing_future_exit_cannot_cancel_past_entry_and_pending_exit_uses_first_valid_open(self):
        del self.quotes[self.start + 4 * MINUTE]
        del self.quotes[self.start + 5 * MINUTE]
        self.quotes[self.start + 6 * MINUTE] = "110"
        result = self.run_sim(costs=self.free)
        self.assertEqual(result["counters"]["entries"], 1)
        self.assertEqual(result["counters"]["exit_missing_quote_minutes"], 2)
        self.assertEqual(result["counters"]["delayed_exits"], 1)
        self.assertEqual(result["trades"][0]["entry_time"], self.start + 2 * MINUTE)
        self.assertEqual(result["trades"][0]["due_time"], self.start + 4 * MINUTE)
        self.assertEqual(result["trades"][0]["exit_time"], self.start + 6 * MINUTE)
        self.assertEqual(result["trades"][0]["exit_reason"], "pending")
        self.assertEqual(result["trades"][0]["exit_reference_price"], "110")

    def test_calendar_rejects_due_after_terminal_but_accepts_due_exactly_terminal(self):
        result = self.run_sim(signals=[self.signal(7), self.signal(8)], costs=self.free)
        self.assertEqual(result["counters"]["entry_calendar_rejected"], 1)
        self.assertEqual(result["trades"][0]["entry_time"], self.start + 7 * MINUTE)
        self.assertEqual(result["trades"][0]["exit_time"], self.config["terminal_time"])

    def test_terminal_missing_is_incomplete_and_never_uses_a_future_year_quote(self):
        start = utc_ms("2024-12-31T23:50:00Z")
        self.start = start
        config = dict(self.config, start_time=start, end_time=start + 10 * MINUTE,
                      terminal_time=start + 9 * MINUTE)
        quotes = {start + index * MINUTE: "100" for index in range(9)}
        quotes[start + 10 * MINUTE] = "1000000"  # 2025 must never fill a 2024 exit.
        result = self.run_sim(quotes=quotes, config=config, mode="buy_hold")
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertIsNone(result["metrics"])
        self.assertEqual(result["trades"], [])
        self.assertIsNotNone(result["final_state"]["position"])
        self.assertTrue(result["final_state"]["stale"])
        self.assertEqual(result["final_state"]["last_price"], "100")
        self.assertEqual(result["final_state"]["time"], config["terminal_time"])
        self.assertIsNone(result["accounting"]["identity_residual"])
        self.assertLess(max(row["time"] for row in result["equity_hourly"]), config["end_time"])

    def test_sell_min_notional_stays_pending_instead_of_forcing_liquidation(self):
        for minute in range(4, 10):
            self.quotes[self.start + minute * MINUTE] = "0.01"
        result = self.run_sim(costs=self.free)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["counters"]["exit_min_notional_rejected"], 6)
        self.assertEqual(result["trades"], [])
        self.assertEqual(result["final_state"]["quantity"], "100")
        self.quotes[self.start + 6 * MINUTE] = "100"
        recovered = self.run_sim(costs=self.free)
        self.assertEqual(recovered["status"], "COMPLETE")
        self.assertEqual(recovered["trades"][0]["exit_time"], self.start + 6 * MINUTE)

    def test_all_minute_drawdown_and_stale_mark_use_last_observed_open(self):
        config = dict(self.config, end_time=self.start + 130 * MINUTE,
                      terminal_time=self.start + 129 * MINUTE, holding_minutes=120)
        quotes = {self.start + index * MINUTE: "100" for index in range(130)}
        quotes[self.start + 30 * MINUTE] = "50"  # Loss between hourly samples.
        del quotes[self.start + 60 * MINUTE]
        result = self.run_sim(quotes=quotes, config=config, costs=self.free)
        self.assertEqual(result["metrics"]["max_drawdown"], .5)
        self.assertEqual(result["counters"]["stale_minutes"], 1)
        stale = next(row for row in result["equity_hourly"] if row["time"] == self.start + HOUR)
        self.assertTrue(stale["stale"])
        self.assertEqual(stale["reference_price"], "100")
        self.assertEqual(stale["equity"], "10000")

    def test_cash_and_constant_returns_have_no_sharpe(self):
        config = dict(self.config, end_time=self.start + 3 * DAY, terminal_time=self.start + 3 * DAY - MINUTE)
        cash = self.run_sim(quotes={}, config=config, mode="cash")
        self.assertEqual(cash["status"], "COMPLETE")
        self.assertEqual(cash["trades"], [])
        self.assertEqual(cash["metrics"]["net_return"], 0)
        self.assertEqual(cash["metrics"]["annualized_return"], 0)
        self.assertIsNone(cash["metrics"]["daily_sharpe"])
        self.assertEqual(cash["metrics"]["turnover"], 0)
        self.assertEqual(cash["metrics"]["exposure"], 0)
        self.assertEqual(len(cash["equity_daily"]), 4)
        self.assertTrue(all(not row["stale"] for row in cash["equity_hourly"]))

    def test_buy_hold_closes_only_at_predeclared_terminal(self):
        result = self.run_sim(mode="buy_hold", costs=self.free)
        self.assertEqual(result["trades"][0]["entry_time"], self.start + 2 * MINUTE)
        self.assertEqual(result["trades"][0]["exit_time"], self.config["terminal_time"])
        self.assertEqual(result["trades"][0]["due_time"], self.config["terminal_time"])
        self.assertEqual(result["metrics"]["exposure"], .7)
        self.assertIsNone(result["metrics"]["profit_factor"])

    def test_daily_sharpe_uses_daily_equity_returns_and_sample_std(self):
        config = dict(self.config, end_time=self.start + 3 * DAY, terminal_time=self.start + 3 * DAY - MINUTE)
        quotes = {self.start + 2 * MINUTE: "100", self.start + DAY - MINUTE: "110",
                  self.start + 2 * DAY - MINUTE: "99", self.start + 3 * DAY - MINUTE: "108.9"}
        result = self.run_sim(quotes=quotes, config=config, costs=self.free, mode="buy_hold")
        returns = [.1, -.1, .1]
        mean = sum(returns) / 3
        std = math.sqrt(sum((value - mean) ** 2 for value in returns) / 2)
        self.assertAlmostEqual(result["metrics"]["daily_sharpe"], math.sqrt(365) * mean / std)
        self.assertAlmostEqual(result["metrics"]["annualized_return"], 1.089 ** (365 / 3) - 1)
        self.assertEqual(result["metrics"]["max_drawdown"], .1)

    def test_constant_nonzero_daily_returns_also_have_no_sharpe(self):
        config = dict(self.config, end_time=self.start + 3 * DAY, terminal_time=self.start + 3 * DAY - MINUTE)
        quotes = {self.start + 2 * MINUTE: "100", self.start + DAY - MINUTE: "110",
                  self.start + 2 * DAY - MINUTE: "121", self.start + 3 * DAY - MINUTE: "133.1"}
        result = self.run_sim(quotes=quotes, config=config, costs=self.free, mode="buy_hold")
        self.assertIsNone(result["metrics"]["daily_sharpe"])
        self.assertAlmostEqual(result["metrics"]["net_return"], .331)

    def test_inputs_are_immutable_and_repeated_simulation_is_identical(self):
        signals = [self.signal(2), self.signal(4)]
        before = copy.deepcopy((signals, self.quotes, self.config, self.costs))
        first = self.run_sim(signals=signals)
        second = self.run_sim(signals=signals)
        self.assertEqual(first, second)
        self.assertEqual((signals, self.quotes, self.config, self.costs), before)

    def test_allocation_floors_quantity_and_keeps_unused_cash(self):
        result = self.run_sim(config=dict(self.config, allocation="0.5", qty_step="3"), costs=self.free)
        self.assertEqual(result["trades"][0]["quantity"], "48")
        self.assertEqual(result["trades"][0]["entry_notional"], "4800")
        self.assertEqual(result["final_state"]["cash"], "10000")

    def test_invalid_config_quotes_and_signals_fail_before_simulation(self):
        for config in (dict(self.config, allocation="1.1"), dict(self.config, qty_step="0"),
                       dict(self.config, terminal_time=self.config["end_time"]),
                       dict(self.config, initial_cash="NaN")):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.run_sim(config=config)
        for quotes in ({self.start: "0"}, {self.start: "NaN"}, {self.start + 1: "100"}):
            with self.subTest(quotes=quotes), self.assertRaises(ValueError):
                self.run_sim(quotes=quotes)
        for signal in (dict(self.signal(2), enter=1), dict(self.signal(2), score=float("nan")),
                       dict(self.signal(2), signal_time=self.start + 3 * MINUTE)):
            with self.subTest(signal=signal), self.assertRaises(ValueError):
                self.run_sim(signals=[signal])


if __name__ == "__main__":
    unittest.main()
