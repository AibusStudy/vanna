"""Tests for responsive Plotly visualization output."""

import pandas as pd

from vanna.integrations.plotly import PlotlyChartGenerator


def test_generates_dashboard_styled_bar_chart() -> None:
    result = PlotlyChartGenerator().generate_chart(
        pd.DataFrame({"region": ["East", "West"], "sales": [10, 20]}),
        "Sales by region",
    )

    assert result["chart_type"] == "bar"
    assert result["figure"]["data"][0]["type"] == "bar"
    assert result["figure"]["layout"]["paper_bgcolor"] == "rgba(0,0,0,0)"
    assert result["figure"]["layout"]["plot_bgcolor"] == "rgba(0,0,0,0)"
    assert result["config"]["responsive"] is True


def test_infers_named_date_column_as_interactive_line_chart() -> None:
    result = PlotlyChartGenerator().generate_chart(
        pd.DataFrame({"order_date": ["2026-01-01", "2026-02-01"], "sales": [10, 20]})
    )

    assert result["chart_type"] == "line"
    assert result["figure"]["data"][0]["type"] == "scatter"
    assert result["figure"]["data"][0]["mode"] == "lines"


def test_single_row_numeric_result_uses_kpi_indicators() -> None:
    result = PlotlyChartGenerator().generate_chart(
        pd.DataFrame({"female": [47_235], "male": [45_965]}),
        "Exam population",
    )

    assert result["chart_type"] == "kpi"
    assert [trace["type"] for trace in result["figure"]["data"]] == [
        "indicator",
        "indicator",
    ]
