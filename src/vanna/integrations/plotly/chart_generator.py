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

    def generate_chart(
        self,
        df: pd.DataFrame,
        title: str = "Chart",
        column_metadata: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        if df.empty:
            raise ValueError("Cannot visualize empty DataFrame")
        frame = self._infer_datetimes(df.copy())
        self.column_metadata = column_metadata or {}
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
                        x=ordered[time_column], y=ordered[column], mode="lines+markers+text", name=self._label(column),
                        text=[self._format_value(value) for value in ordered[column]], textposition="top center",
                        customdata=[[self._description(column)] for _ in range(len(ordered))],
                        hovertemplate=f"<b>{self._label(column)}</b><br>{self._label(time_column)}: %{{x}}<br>값: %{{y:,}}<br>%{{customdata[0]}}<extra></extra>",
                    line={"color": self.COLOR_PALETTE[index % len(self.COLOR_PALETTE)], "width": 2},
                    fill="tozeroy" if index == 0 else None,
                    fillcolor="rgba(53,185,241,0.12)" if index == 0 else None,
                ))
            figure.update_layout(xaxis_title=self._label(time_column), yaxis_title="값", hovermode="closest")
            return figure, "line"

        if len(numeric) == 1 and not categorical:
            column = numeric[0]
            return go.Figure(go.Histogram(x=df[column], marker_color=self.COLOR_PALETTE[0], name=self._label(column), hovertemplate=f"<b>{self._label(column)}</b><br>구간: %{{x}}<br>건수: %{{y:,}}<br>{self._description(column)}<extra></extra>")), "histogram"

        if len(numeric) == 1 and len(categorical) == 1:
            category, value = categorical[0], numeric[0]
            grouped = df.groupby(category, dropna=False)[value].sum().reset_index()
            descriptions = [self._description(value)] * len(grouped)
            if self._should_use_donut(grouped[value], value):
                figure = go.Figure(go.Pie(
                    labels=grouped[category].astype(str), values=grouped[value], hole=0.58,
                    marker={"colors": self.COLOR_PALETTE, "line": {"color": "white", "width": 2}},
                    textinfo="percent+label", textposition="outside", customdata=descriptions,
                    hovertemplate=f"<b>%{{label}}</b><br>{self._label(value)}: %{{value:,}}<br>비율: %{{percent}}<br>%{{customdata}}<extra></extra>",
                ))
                return figure, "donut"
            figure = go.Figure(go.Bar(
                x=grouped[category].astype(str), y=grouped[value], marker_color=self.COLOR_PALETTE[0],
                name=self._label(value), text=[self._format_value(item) for item in grouped[value]], textposition="outside",
                customdata=[[item] for item in descriptions],
                hovertemplate=f"<b>{self._label(value)}</b><br>{self._label(category)}: %{{x}}<br>값: %{{y:,}}<br>%{{customdata[0]}}<extra></extra>",
            ))
            figure.update_layout(xaxis_title=self._label(category), yaxis_title=self._label(value))
            return figure, "bar"

        if categorical and len(numeric) >= 2:
            category = categorical[0]
            grouped = df.groupby(category, dropna=False)[numeric[:6]].sum().reset_index()
            figure = go.Figure()
            for index, column in enumerate(numeric[:6]):
                figure.add_trace(go.Bar(
                    x=grouped[category].astype(str), y=grouped[column], name=self._label(column),
                    marker_color=self.COLOR_PALETTE[index % len(self.COLOR_PALETTE)],
                    text=[self._format_value(item) for item in grouped[column]], textposition="outside",
                    customdata=[[self._description(column)] for _ in range(len(grouped))],
                    hovertemplate=f"<b>{self._label(column)}</b><br>{self._label(category)}: %{{x}}<br>값: %{{y:,}}<br>%{{customdata[0]}}<extra></extra>",
                ))
            figure.update_layout(barmode="group", xaxis_title=self._label(category), yaxis_title="값")
            return figure, "grouped_bar"

        if len(numeric) == 2:
            first, second = numeric
            figure = go.Figure(go.Scatter(
                x=df[first], y=df[second], mode="markers+text", name=self._label(second),
                text=[self._format_value(item) for item in df[second]], textposition="top center",
                marker={"color": self.COLOR_PALETTE[0], "size": 9, "opacity": 0.82, "line": {"color": "white", "width": 1}},
                customdata=[[self._description(second)] for _ in range(len(df))],
                hovertemplate=f"<b>{self._label(second)}</b><br>{self._label(first)}: %{{x:,}}<br>{self._label(second)}: %{{y:,}}<br>%{{customdata[0]}}<extra></extra>",
            ))
            figure.update_layout(xaxis_title=self._label(first), yaxis_title=self._label(second))
            return figure, "scatter"

        if len(numeric) >= 3:
            correlation = df[numeric].corr()
            labels = [self._label(column) for column in numeric]
            figure = go.Figure(go.Heatmap(z=correlation.values, x=labels, y=labels, zmin=-1, zmax=1, texttemplate="%{z:.2f}", colorscale=[[0, "#8C8CEB"], [0.5, "#F8FAFC"], [1, "#54D6BD"]], colorbar={"title": "상관계수"}, hovertemplate="<b>%{x} × %{y}</b><br>상관계수: %{z:.3f}<extra></extra>"))
            return figure, "heatmap"

        if len(categorical) >= 2:
            first, second = categorical[:2]
            grouped = df.groupby([first, second], dropna=False).size().reset_index(name="count")
            figure = go.Figure()
            for index, (name, values) in enumerate(grouped.groupby(second, dropna=False)):
                figure.add_trace(go.Bar(x=values[first].astype(str), y=values["count"], name=str(name), marker_color=self.COLOR_PALETTE[index % len(self.COLOR_PALETTE)], text=values["count"], textposition="outside"))
            figure.update_layout(barmode="group", xaxis_title=self._label(first), yaxis_title="건수")
            return figure, "grouped_bar"

        if len(categorical) == 1:
            column = categorical[0]
            counts = df[column].fillna("Unknown").astype(str).value_counts().head(10).sort_values()
            figure = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", marker_color=self.COLOR_PALETTE[0]))
            figure.update_traces(text=counts.values, textposition="outside", name=self._label(column), hovertemplate=f"<b>%{{y}}</b><br>건수: %{{x:,}}<br>{self._description(column)}<extra></extra>")
            figure.update_layout(xaxis_title="건수", yaxis_title=self._label(column))
            return figure, "horizontal_bar"

        if len(df.columns) >= 2:
            first, second = df.columns[:2]
            values = pd.to_numeric(df[second], errors="coerce").fillna(0)
            figure = go.Figure(go.Bar(x=df[first].astype(str), y=values, marker_color=self.COLOR_PALETTE[1]))
            figure.update_traces(text=[self._format_value(item) for item in values], textposition="outside", name=self._label(second))
            figure.update_layout(xaxis_title=self._label(first), yaxis_title=self._label(second))
            return figure, "bar"

        raise ValueError("Cannot determine appropriate visualization for this DataFrame")

    def _apply_dashboard_style(self, figure: go.Figure, title: str, chart_type: str) -> None:
        figure.update_layout(
            title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 18, "color": self.TEXT_COLOR}},
            autosize=True, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Malgun Gothic, Inter, sans-serif", "color": self.TEXT_COLOR, "size": 12},
            colorway=self.COLOR_PALETTE, margin={"l": 56, "r": 24, "t": 64, "b": 56},
            hoverlabel={"bgcolor": "white", "font": {"color": self.TEXT_COLOR}},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "center", "x": 0.5, "font": {"color": self.TEXT_COLOR}},
            uirevision="gensql-chart",
        )
        if chart_type != "kpi":
            figure.update_xaxes(showgrid=False, zeroline=False, showline=False, tickfont={"color": self.MUTED_TEXT_COLOR}, title_font={"color": self.MUTED_TEXT_COLOR}, automargin=True)
            figure.update_yaxes(showgrid=False, zeroline=False, showline=False, tickfont={"color": self.MUTED_TEXT_COLOR}, title_font={"color": self.MUTED_TEXT_COLOR}, tickformat="~s", automargin=True)

    def _label(self, column: Any) -> str:
        name = str(column)
        metadata = self.column_metadata.get(name, {})
        return metadata.get("logical_name") or name.replace("_", " ").title()

    def _description(self, column: Any) -> str:
        metadata = self.column_metadata.get(str(column), {})
        return metadata.get("description") or ""

    @staticmethod
    def _format_value(value: Any) -> str:
        if pd.isna(value):
            return "-"
        if isinstance(value, int):
            return f"{value:,}"
        if isinstance(value, float):
            return f"{value:,.2f}".rstrip("0").rstrip(".")
        return str(value)

    @staticmethod
    def _should_use_donut(values: pd.Series, column: Any) -> bool:
        name = str(column).lower()
        total = float(values.fillna(0).sum())
        return len(values) <= 8 and values.ge(0).all() and (
            any(token in name for token in ("percent", "percentage", "ratio", "rate", "share", "pct"))
            or 99 <= total <= 101
        )
