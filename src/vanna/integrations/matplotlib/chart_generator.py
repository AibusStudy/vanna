"""Matplotlib-based chart generator with deterministic chart selection."""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


class MatplotlibChartGenerator:
    """Render a DataFrame as a PNG using data-shape heuristics."""

    COLOR_PALETTE = [
        "#35B9F1",  # dashboard blue
        "#54D6BD",  # mint
        "#8C8CEB",  # lavender
        "#F38CAF",  # soft pink
        "#77DCE8",  # cyan
        "#A8E6CF",  # pale mint
    ]
    TEXT_COLOR = "#475569"
    MUTED_TEXT_COLOR = "#7C8798"
    GRID_COLOR = "#DCE3EA"

    def generate_chart(self, df: pd.DataFrame, title: str = "Chart") -> dict[str, Any]:
        if df.empty:
            raise ValueError("Cannot visualize empty DataFrame")

        frame = self._infer_datetimes(df.copy())
        plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        figure, axes = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
        figure.patch.set_alpha(0)
        axes.set_facecolor("none")
        chart_type = self._draw(figure, axes, frame, title)
        self._apply_dashboard_style(figure, axes, title, chart_type)

        buffer = io.BytesIO()
        try:
            figure.savefig(
                buffer,
                format="png",
                dpi=144,
                bbox_inches="tight",
                transparent=True,
            )
        finally:
            plt.close(figure)

        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {
            "format": "png",
            "mime_type": "image/png",
            "data_url": f"data:image/png;base64,{encoded}",
            "chart_type": chart_type,
        }

    @staticmethod
    def _infer_datetimes(df: pd.DataFrame) -> pd.DataFrame:
        for column in df.select_dtypes(include=["object"]).columns:
            name = str(column).lower()
            if not any(token in name for token in ("date", "time", "month", "year")):
                continue
            converted = pd.to_datetime(df[column], errors="coerce")
            if converted.notna().mean() >= 0.8:
                df[column] = converted
        return df

    def _draw(
        self,
        figure: plt.Figure,
        axes: plt.Axes,
        df: pd.DataFrame,
        title: str,
    ) -> str:
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetimes = df.select_dtypes(include=["datetime64"]).columns.tolist()

        if len(df) == 1 and numeric:
            axes.axis("off")
            values = [(str(column), df.iloc[0][column]) for column in numeric[:4]]
            for index, (label, value) in enumerate(values):
                x = (index + 0.5) / len(values)
                axes.text(
                    x,
                    0.58,
                    self._format_value(value),
                    transform=axes.transAxes,
                    ha="center",
                    va="center",
                    fontsize=25,
                    color=self.TEXT_COLOR,
                    fontweight="normal",
                )
                axes.text(
                    x,
                    0.42,
                    label.replace("_", " ").title(),
                    transform=axes.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=self.MUTED_TEXT_COLOR,
                )
            chart_type = "kpi"
        elif datetimes and numeric:
            time_column = datetimes[0]
            ordered = df.sort_values(time_column)
            for index, column in enumerate(numeric[:5]):
                color = self.COLOR_PALETTE[index % len(self.COLOR_PALETTE)]
                axes.plot(
                    ordered[time_column],
                    ordered[column],
                    label=str(column),
                    color=color,
                    linewidth=2,
                )
                axes.fill_between(
                    ordered[time_column],
                    ordered[column],
                    color=color,
                    alpha=0.12,
                )
            axes.set_xlabel(str(time_column))
            axes.set_ylabel("Value")
            chart_type = "line"
        elif len(numeric) == 1 and not categorical:
            axes.hist(df[numeric[0]].dropna(), color=self.COLOR_PALETTE[0])
            axes.set_xlabel(str(numeric[0]))
            axes.set_ylabel("Count")
            chart_type = "histogram"
        elif len(numeric) == 1 and len(categorical) == 1:
            grouped = df.groupby(categorical[0], dropna=False)[numeric[0]].sum()
            bars = grouped.plot.bar(ax=axes, color=self.COLOR_PALETTE[0], width=0.68)
            axes.set_ylabel(str(numeric[0]))
            self._label_bars(bars)
            chart_type = "bar"
        elif categorical and len(numeric) >= 2:
            grouped = df.groupby(categorical[0], dropna=False)[numeric[:6]].sum()
            grouped.plot.bar(
                ax=axes,
                color=self.COLOR_PALETTE,
                width=0.72,
            )
            axes.set_ylabel("Value")
            chart_type = "grouped_bar"
        elif len(numeric) == 2:
            axes.scatter(
                df[numeric[0]],
                df[numeric[1]],
                color=self.COLOR_PALETTE[0],
                alpha=0.8,
                s=48,
                edgecolors="white",
                linewidths=0.6,
            )
            axes.set_xlabel(str(numeric[0]))
            axes.set_ylabel(str(numeric[1]))
            chart_type = "scatter"
        elif len(numeric) >= 3:
            correlation = df[numeric].corr()
            image = axes.imshow(correlation, cmap="PuBuGn", vmin=-1, vmax=1)
            axes.set_xticks(np.arange(len(numeric)), labels=numeric, rotation=45, ha="right")
            axes.set_yticks(np.arange(len(numeric)), labels=numeric)
            figure.colorbar(image, ax=axes, label="Correlation")
            chart_type = "heatmap"
        elif len(categorical) >= 2:
            grouped = df.groupby(categorical[:2], dropna=False).size().unstack(fill_value=0)
            grouped.plot.bar(ax=axes, color=self.COLOR_PALETTE)
            axes.set_ylabel("Count")
            chart_type = "grouped_bar"
        elif len(categorical) == 1:
            counts = df[categorical[0]].fillna("Unknown").astype(str).value_counts().head(10)
            counts.sort_values().plot.barh(ax=axes, color=self.COLOR_PALETTE[0])
            axes.set_xlabel("Count")
            axes.set_ylabel(str(categorical[0]))
            chart_type = "horizontal_bar"
        elif len(df.columns) >= 2:
            axes.bar(
                df.iloc[:, 0].astype(str),
                pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(0),
                color=self.COLOR_PALETTE[1],
            )
            axes.set_xlabel(str(df.columns[0]))
            axes.set_ylabel(str(df.columns[1]))
            chart_type = "bar"
        else:
            raise ValueError("Cannot determine appropriate visualization for this DataFrame")

        return chart_type

    def _apply_dashboard_style(
        self,
        figure: plt.Figure,
        axes: plt.Axes,
        title: str,
        chart_type: str,
    ) -> None:
        figure.patch.set_alpha(0)
        for current_axes in figure.get_axes():
            current_axes.set_facecolor("none")
            current_axes.tick_params(
                colors=self.MUTED_TEXT_COLOR,
                labelsize=8,
                length=0,
            )
            current_axes.xaxis.label.set_color(self.MUTED_TEXT_COLOR)
            current_axes.yaxis.label.set_color(self.MUTED_TEXT_COLOR)
            for spine in current_axes.spines.values():
                spine.set_color(self.GRID_COLOR)
                spine.set_linewidth(0.8)

        if chart_type != "kpi":
            axes.grid(axis="y", color=self.GRID_COLOR, linewidth=0.8, alpha=0.7)
            axes.set_axisbelow(True)
            axes.yaxis.set_major_formatter(FuncFormatter(self._compact_number))
            axes.set_title(
                title,
                loc="left",
                pad=16,
                fontsize=11,
                fontweight="normal",
                color=self.TEXT_COLOR,
            )
            if chart_type in {"bar", "grouped_bar"}:
                axes.tick_params(axis="x", labelrotation=35)
        else:
            axes.set_title(
                title,
                loc="left",
                pad=8,
                fontsize=11,
                fontweight="normal",
                color=self.TEXT_COLOR,
            )

        legend = axes.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
            for text in legend.get_texts():
                text.set_color(self.MUTED_TEXT_COLOR)
                text.set_fontsize(8)

    @staticmethod
    def _compact_number(value: float, _position: int) -> str:
        absolute = abs(value)
        if absolute >= 1_000_000_000:
            return f"{value / 1_000_000_000:.1f}B"
        if absolute >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if absolute >= 1_000:
            return f"{value / 1_000:.1f}K"
        return f"{value:g}"

    @staticmethod
    def _format_value(value: Any) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):,.2f}".rstrip("0").rstrip(".")
        return str(value)

    def _label_bars(self, axes: plt.Axes) -> None:
        for container in axes.containers:
            axes.bar_label(
                container,
                labels=[self._format_value(bar.get_height()) for bar in container],
                padding=3,
                fontsize=8,
                color=self.MUTED_TEXT_COLOR,
            )
