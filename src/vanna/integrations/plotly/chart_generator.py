"""Responsive Plotly chart generator using the GenSQL dashboard style."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio


class PlotlyChartGenerator:
    """Build an interactive Plotly figure from a SQL-result DataFrame."""

    COLOR_PALETTE = ["#35B9F1", "#54D6BD", "#8C8CEB", "#F38CAF", "#77DCE8", "#A8E6CF"]
    TEXT_COLOR = "#475569"
    MUTED_TEXT_COLOR = "#7C8798"
    GRID_COLOR = "#DCE3EA"

    def generate_chart(self, df: pd.DataFrame, title: str = "Chart") -> dict[str, Any]:
        if df.empty:
            raise ValueError("Cannot visualize empty DataFrame")
        frame = self._infer_datetimes(df.copy())
        figure, chart_type = self._create_figure(frame)
        self._apply_dashboard_style(figure, title, chart_type)
        return {
            "chart_type": chart_type,
            "figure": json.loads(pio.to_json(figure)),
            "config": {
                "responsive": True,
                "displaylogo": False,
                "scrollZoom": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
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

    def _create_figure(self, df: pd.DataFrame) -> tuple[go.Figure, str]:
        numeric = df.select_dtypes(include=["number"]).columns.tolist()
        categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
        datetimes = df.select_dtypes(include=["datetime64"]).columns.tolist()

        if len(df) == 1 and numeric:
            figure = go.Figure()
            values = numeric[:4]
            for index, column in enumerate(values):
                figure.add_trace(go.Indicator(
                    mode="number", value=float(df.iloc[0][column]),
                    number={"valueformat": ",", "font": {"color": self.TEXT_COLOR}},
                    title={"text": str(column).replace("_", " ").title(), "font": {"color": self.MUTED_TEXT_COLOR, "size": 12}},
                    domain={"row": 0, "column": index},
                ))
            figure.update_layout(grid={"rows": 1, "columns": len(values), "pattern": "independent"})
            return figure, "kpi"

        if datetimes and numeric:
            time_column = datetimes[0]
            ordered = df.sort_values(time_column)
            figure = go.Figure()
            for index, column in enumerate(numeric[:5]):
                figure.add_trace(go.Scatter(
                    x=ordered[time_column], y=ordered[column], mode="lines", name=str(column),
                    line={"color": self.COLOR_PALETTE[index % len(self.COLOR_PALETTE)], "width": 2},
                    fill="tozeroy" if index == 0 else None,
                    fillcolor="rgba(53,185,241,0.12)" if index == 0 else None,
                ))
            figure.update_layout(xaxis_title=str(time_column), yaxis_title="Value", hovermode="x unified")
            return figure, "line"

        if len(numeric) == 1 and not categorical:
            column = numeric[0]
            return go.Figure(go.Histogram(x=df[column], marker_color=self.COLOR_PALETTE[0], name=str(column))), "histogram"

        if len(numeric) == 1 and len(categorical) == 1:
            category, value = categorical[0], numeric[0]
            grouped = df.groupby(category, dropna=False)[value].sum().reset_index()
            figure = go.Figure(go.Bar(x=grouped[category].astype(str), y=grouped[value], marker_color=self.COLOR_PALETTE[0], text=grouped[value], textposition="outside"))
            figure.update_layout(xaxis_title=str(category), yaxis_title=str(value))
            return figure, "bar"

        if categorical and len(numeric) >= 2:
            category = categorical[0]
            grouped = df.groupby(category, dropna=False)[numeric[:6]].sum().reset_index()
            figure = go.Figure()
            for index, column in enumerate(numeric[:6]):
                figure.add_trace(go.Bar(x=grouped[category].astype(str), y=grouped[column], name=str(column), marker_color=self.COLOR_PALETTE[index % len(self.COLOR_PALETTE)]))
            figure.update_layout(barmode="group", xaxis_title=str(category), yaxis_title="Value")
            return figure, "grouped_bar"

        if len(numeric) == 2:
            first, second = numeric
            figure = go.Figure(go.Scatter(x=df[first], y=df[second], mode="markers", marker={"color": self.COLOR_PALETTE[0], "size": 9, "opacity": 0.82, "line": {"color": "white", "width": 1}}))
            figure.update_layout(xaxis_title=str(first), yaxis_title=str(second))
            return figure, "scatter"

        if len(numeric) >= 3:
            correlation = df[numeric].corr()
            figure = go.Figure(go.Heatmap(z=correlation.values, x=numeric, y=numeric, zmin=-1, zmax=1, colorscale=[[0, "#8C8CEB"], [0.5, "#F8FAFC"], [1, "#54D6BD"]], colorbar={"title": "Correlation"}))
            return figure, "heatmap"

        if len(categorical) >= 2:
            first, second = categorical[:2]
            grouped = df.groupby([first, second], dropna=False).size().reset_index(name="count")
            figure = go.Figure()
            for index, (name, values) in enumerate(grouped.groupby(second, dropna=False)):
                figure.add_trace(go.Bar(x=values[first].astype(str), y=values["count"], name=str(name), marker_color=self.COLOR_PALETTE[index % len(self.COLOR_PALETTE)]))
            figure.update_layout(barmode="group", xaxis_title=str(first), yaxis_title="Count")
            return figure, "grouped_bar"

        if len(categorical) == 1:
            column = categorical[0]
            counts = df[column].fillna("Unknown").astype(str).value_counts().head(10).sort_values()
            figure = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", marker_color=self.COLOR_PALETTE[0]))
            figure.update_layout(xaxis_title="Count", yaxis_title=str(column))
            return figure, "horizontal_bar"

        if len(df.columns) >= 2:
            first, second = df.columns[:2]
            values = pd.to_numeric(df[second], errors="coerce").fillna(0)
            figure = go.Figure(go.Bar(x=df[first].astype(str), y=values, marker_color=self.COLOR_PALETTE[1]))
            figure.update_layout(xaxis_title=str(first), yaxis_title=str(second))
            return figure, "bar"

        raise ValueError("Cannot determine appropriate visualization for this DataFrame")

    def _apply_dashboard_style(self, figure: go.Figure, title: str, chart_type: str) -> None:
        figure.update_layout(
            title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 16, "color": self.TEXT_COLOR}},
            autosize=True, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Malgun Gothic, Inter, sans-serif", "color": self.TEXT_COLOR, "size": 12},
            colorway=self.COLOR_PALETTE, margin={"l": 56, "r": 24, "t": 64, "b": 56},
            hoverlabel={"bgcolor": "white", "font": {"color": self.TEXT_COLOR}},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            uirevision="gensql-chart",
        )
        if chart_type != "kpi":
            figure.update_xaxes(showgrid=False, zeroline=False, linecolor=self.GRID_COLOR, tickfont={"color": self.MUTED_TEXT_COLOR}, title_font={"color": self.MUTED_TEXT_COLOR}, automargin=True)
            figure.update_yaxes(showgrid=True, gridcolor=self.GRID_COLOR, gridwidth=1, zeroline=False, linecolor=self.GRID_COLOR, tickfont={"color": self.MUTED_TEXT_COLOR}, title_font={"color": self.MUTED_TEXT_COLOR}, tickformat="~s", automargin=True)
