"""Tests for responsive Plotly visualization output."""

import pandas as pd

from vanna.integrations.plotly import PlotlyChartGenerator


def test_generates_dashboard_styled_bar_chart() -> None:
    result = PlotlyChartGenerator().generate_chart(
        pd.DataFrame({"region": ["East", "West"], "sales": [10, 20]}),
        "Sales by region",
        column_metadata={
            "region": {"logical_name": "지역", "description": "판매 지역"},
            "sales": {"logical_name": "매출액", "description": "지역별 매출 합계"},
        },
    )

    assert result["chart_type"] == "bar"
    assert result["figure"]["data"][0]["type"] == "bar"
    assert result["figure"]["layout"]["paper_bgcolor"] == "rgba(0,0,0,0)"
    assert result["figure"]["layout"]["plot_bgcolor"] == "rgba(0,0,0,0)"
    assert result["figure"]["layout"]["xaxis"]["title"]["text"] == "지역"
    assert result["figure"]["layout"]["yaxis"]["title"]["text"] == "매출액"
    assert result["figure"]["layout"]["xaxis"]["showgrid"] is False
    assert result["figure"]["layout"]["yaxis"]["showgrid"] is False
    assert result["figure"]["data"][0]["name"] == "매출액"
    assert result["figure"]["data"][0]["customdata"] == [
        ["지역별 매출 합계"],
        ["지역별 매출 합계"],
    ]
    assert "%{customdata[0]}" in result["figure"]["data"][0]["hovertemplate"]
    assert result["config"]["responsive"] is True


def test_infers_named_date_column_as_interactive_line_chart() -> None:
    result = PlotlyChartGenerator().generate_chart(
        pd.DataFrame({"order_date": ["2026-01-01", "2026-02-01"], "sales": [10, 20]})
    )

    assert result["chart_type"] == "line"
    assert result["figure"]["data"][0]["type"] == "scatter"
    assert result["figure"]["data"][0]["mode"] == "lines+markers+text"


def test_percentage_categories_use_labeled_donut_chart() -> None:
    result = PlotlyChartGenerator().generate_chart(
        pd.DataFrame({"category": ["제품", "실내", "의상"], "share_percent": [40, 35, 25]}),
        "디자이너 주요 취업 분류",
        column_metadata={
            "category": {"logical_name": "취업 분류", "description": "디자인 직무 분류"},
            "share_percent": {"logical_name": "구성 비율", "description": "전체 취업자 중 비율"},
        },
    )

    assert result["chart_type"] == "donut"
    assert result["figure"]["data"][0]["hole"] == 0.58
    assert result["figure"]["data"][0]["textinfo"] == "percent+label"


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
