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


class MatplotlibChartGenerator:
    """Render a DataFrame as a PNG using data-shape heuristics."""

    COLOR_PALETTE = ["#15a8a8", "#fe5d26", "#bf1363", "#023d60"]

    def generate_chart(self, df: pd.DataFrame, title: str = "Chart") -> dict[str, Any]:
        if df.empty:
            raise ValueError("Cannot visualize empty DataFrame")

        frame = self._infer_datetimes(df.copy())
        figure, axes = plt.subplots(figsize=(10, 6), constrained_layout=True)
        chart_type = self._draw(figure, axes, frame, title)

        buffer = io.BytesIO()
        try:
            figure.savefig(buffer, format="png", dpi=144, bbox_inches="tight")
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
        if len(df.columns) >= 4:
            axes.axis("off")
            preview = df.head(50)
            table = axes.table(
                cellText=preview.astype(str).values,
                colLabels=[str(column) for column in preview.columns],
                loc="center",
                cellLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.25)
            axes.set_title(title)
            return "table"

        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetimes = df.select_dtypes(include=["datetime64"]).columns.tolist()

        if datetimes and numeric:
            time_column = datetimes[0]
            ordered = df.sort_values(time_column)
            for index, column in enumerate(numeric[:5]):
                axes.plot(
                    ordered[time_column],
                    ordered[column],
                    label=str(column),
                    color=self.COLOR_PALETTE[index % len(self.COLOR_PALETTE)],
                )
            axes.set_xlabel(str(time_column))
            axes.set_ylabel("Value")
            axes.legend()
            chart_type = "line"
        elif len(numeric) == 1 and not categorical:
            axes.hist(df[numeric[0]].dropna(), color=self.COLOR_PALETTE[0])
            axes.set_xlabel(str(numeric[0]))
            axes.set_ylabel("Count")
            chart_type = "histogram"
        elif len(numeric) == 1 and len(categorical) == 1:
            grouped = df.groupby(categorical[0], dropna=False)[numeric[0]].sum()
            grouped.plot.bar(ax=axes, color=self.COLOR_PALETTE[1])
            axes.set_ylabel(str(numeric[0]))
            chart_type = "bar"
        elif len(numeric) == 2:
            axes.scatter(
                df[numeric[0]], df[numeric[1]], color=self.COLOR_PALETTE[2]
            )
            axes.set_xlabel(str(numeric[0]))
            axes.set_ylabel(str(numeric[1]))
            chart_type = "scatter"
        elif len(numeric) >= 3:
            correlation = df[numeric].corr()
            image = axes.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
            axes.set_xticks(np.arange(len(numeric)), labels=numeric, rotation=45, ha="right")
            axes.set_yticks(np.arange(len(numeric)), labels=numeric)
            figure.colorbar(image, ax=axes, label="Correlation")
            chart_type = "heatmap"
        elif len(categorical) >= 2:
            grouped = df.groupby(categorical[:2], dropna=False).size().unstack(fill_value=0)
            grouped.plot.bar(ax=axes, color=self.COLOR_PALETTE)
            axes.set_ylabel("Count")
            chart_type = "grouped_bar"
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

        axes.set_title(title)
        axes.grid(axis="y", alpha=0.2)
        return chart_type
