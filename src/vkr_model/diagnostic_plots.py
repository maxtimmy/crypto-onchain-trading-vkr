"""Static research figures from the deterministic diagnostic summary only."""

from pathlib import Path


METHOD_LABELS = {
    "zero": "Ноль", "train_mean": "Среднее обучения",
    "depth4": "CatBoost, глубина 4", "depth6": "CatBoost, глубина 6",
}
COLORS = {
    "target": "#252525", "zero": "#7a7a7a", "train_mean": "#748239",
    "depth4": "#2872ad", "depth6": "#cf8425",
}


def _axes_style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#e3e6e8", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=10)


def plot_diagnostics(summary, output_dir):
    """Return relative PNG names; never query inputs or invoke model methods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelcolor": "#30343b", "text.color": "#30343b",
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white"}):
        monthly = summary["validation"]["monthly"]
        months = sorted(monthly)
        x = list(range(len(months)))
        month_names = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн",
                       "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]
        labels = [month_names[int(month[5:7]) - 1] for month in months]
        figure, axes = plt.subplots(2, 1, figsize=(11.5, 8), sharex=True)
        for model in ("depth4", "depth6"):
            for baseline in ("zero", "train_mean"):
                values = [monthly[month]["comparisons"][model][baseline] for month in months]
                style = {"color": COLORS[model], "linestyle": "--" if baseline == "zero" else "-",
                         "marker": "o" if model == "depth4" else "s",
                         "markersize": 4, "linewidth": 1.6,
                         "label": "Глубина %s − %s" % (model[-1], "ноль" if baseline == "zero" else "среднее")}
                axes[0].plot(x, [value["excess_mse"] * 1e8 for value in values], **style)
                axes[1].plot(x, [value["excess_mae"] * 1e4 for value in values], **style)
        for axis in axes:
            _axes_style(axis)
            axis.axhline(0, color="#40444b", linewidth=1)
        axes[0].set_ylabel("Превышение MSE, (б.п.)²")
        axes[1].set_ylabel("Превышение MAE, б.п.")
        axes[1].set_xticks(x, labels)
        axes[1].set_xlabel("Месяц исполнения, 2024 год, UTC")
        axes[0].set_xlim(-0.2, len(months) - 0.8)
        figure.suptitle("Ошибки CatBoost относительно постоянных прогнозов", y=0.97, fontsize=16)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.93),
                      frameon=False, ncol=2, fontsize=10)
        figure.text(0.5, 0.035, "Выше нуля — хуже базового прогноза. В каждом месяце строки одинаковы у всех методов.",
                    ha="center", fontsize=10)
        figure.tight_layout(rect=(0.02, 0.06, 0.98, 0.85), h_pad=2)
        figure.savefig(directory / "monthly_errors.png", dpi=160,
                       metadata={"Software": "Matplotlib"})
        plt.close(figure)

        figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharey="row")
        distributions = summary["distributions"]
        for column, split in enumerate(("train", "validation")):
            values = distributions[split]
            probabilities = values["probabilities"]
            series = {"target": values["target_quantiles"], **values["prediction_quantiles"]}
            for row, indices in enumerate((list(range(101)), list(range(1, 100)))):
                axis = axes[row, column]
                for method in ("target", "zero", "train_mean", "depth4", "depth6"):
                    axis.plot([probabilities[index] * 100 for index in indices],
                              [series[method][index] * 1e4 for index in indices],
                              color=COLORS[method], linewidth=1.6,
                              linestyle={"target": "-", "zero": ":", "train_mean": "--",
                                         "depth4": "-", "depth6": "--"}[method],
                              label="Фактический ответ" if method == "target" else METHOD_LABELS[method])
                _axes_style(axis)
                axis.axhline(0, color="#a0a4aa", linewidth=0.7)
                axis.set_xlim(0 if row == 0 else 1, 100 if row == 0 else 99)
                axis.set_xticks([0, 25, 50, 75, 100] if row == 0 else [1, 25, 50, 75, 99])
                axis.set_xlabel("Процентиль")
                if column == 0:
                    axis.set_ylabel("Доходность / прогноз, б.п.")
                period = "Обучение, 2022–2023" if split == "train" else "Валидация, 2024"
                scope = "0–100%" if row == 0 else "1–99%, увеличение центра"
                axis.set_title("%s · %s" % (period, scope), fontsize=11, pad=12)
        figure.suptitle("Распределения четырёхчасовых ответов и прогнозов", y=0.98, fontsize=16)
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 0.94),
                      ncol=3, frameon=False, fontsize=10)
        figure.text(0.5, 0.025,
                    "Эмпирические квантили, линейный метод, 101 точка. 1 б.п. = 0.0001. Оси одинаковы внутри каждой строки.",
                    ha="center", fontsize=10)
        figure.tight_layout(rect=(0.015, 0.055, 0.985, 0.85), h_pad=2.5, w_pad=2)
        figure.savefig(directory / "forecast_quantiles.png", dpi=160,
                       metadata={"Software": "Matplotlib"})
        plt.close(figure)
    return ["monthly_errors.png", "forecast_quantiles.png"]
