"""Static local research charts, built exclusively from simulation outputs."""

from datetime import datetime, timezone
from pathlib import Path


LABELS = {"cash": "Деньги / ноль / среднее", "buy_hold": "Купить и держать",
          "sma24_168": "SMA 24 / 168", "depth4": "CatBoost, глубина 4",
          "depth6": "CatBoost, глубина 6"}
COLORS = {"cash": "#777777", "buy_hold": "#23916b", "sma24_168": "#9267aa",
          "depth4": "#2872ad", "depth6": "#cf8425"}


def plot_backtest(summary, details, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, sharey=True)
        for axis, scenario in zip(axes, ("base", "stress")):
            for method in LABELS:
                rows = details[scenario][method]["equity_hourly"]
                x = [datetime.fromtimestamp(row["time"] / 1000, timezone.utc) for row in rows]
                initial = float(summary["scenarios"][scenario][method]["accounting"]["initial_cash"])
                axis.plot(x, [100 * (float(row["equity"]) / initial - 1) for row in rows],
                          label=LABELS[method], color=COLORS[method], linewidth=1.3,
                          linestyle="--" if method == "cash" else "-")
            axis.axhline(0, color="#cccccc", linewidth=0.7, zorder=0)
            axis.grid(axis="y", color="#e5e7eb", linewidth=0.7)
            axis.set_axisbelow(True)
            axis.set_ylabel("Изменение капитала, %")
            axis.set_title("Базовые расходы" if scenario == "base" else "Повышенные расходы", loc="left")
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2, tz=timezone.utc))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m.%Y", tz=timezone.utc))
        axes[-1].set_xlabel("Время UTC; часовые отметки и фиксированный выход 31.12.2024 23:59")
        figure.suptitle("Первый технический бэктест BTCUSDT · 2024", y=0.98, fontsize=16)
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94),
                      frameon=False, ncol=3, fontsize=10)
        figure.text(0.5, 0.025, "Сценарные цены исполнения и расходы. 2024 уже использовался для диагностики; независимого теста нет.",
                    ha="center", fontsize=10)
        if summary["status"] != "COMPLETE":
            figure.text(0.5, 0.06, "НЕПОЛНЫЙ РАСЧЁТ: кривые содержат предварительную оценку открытых позиций", ha="center", color="#a72d2d")
        figure.tight_layout(rect=(0.02, 0.06, 0.98, 0.85), h_pad=2)
        path = directory / "equity_2024.png"
        figure.savefig(path, dpi=150, metadata={"Software": "Matplotlib"})
        plt.close(figure)
    return [path.name]
