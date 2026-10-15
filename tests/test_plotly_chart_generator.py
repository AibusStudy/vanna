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


def test_year_and_gender_are_dimensions_with_korean_values_and_red_series() -> None:
    result = PlotlyChartGenerator().generate_chart(
        pd.DataFrame({
            "SEX_TYPE": ["F", "M", "F", "M"],
            "HLCK_BIZ_YR": [2018, 2018, 2019, 2019],
            "general_exam_completed_count": [140_622, 136_925, 141_100, 137_200],
        }),
        "2018년부터 2019년까지 성별·연도별 건강검진 수검자 추이",
        column_metadata={
            "SEX_TYPE": {
                "logical_name": "성별",
                "description": "남성은 M, 여성은 F로 표기함",
                "sql_role": "code",
                "semantic_type": "gender",
                "code_name": "SEX_TYPE",
            },
            "HLCK_BIZ_YR": {
                "logical_name": "검진 연도",
                "description": "건강검진 사업연도",
                "sql_role": "time",
                "semantic_type": "year",
            },
            "general_exam_completed_count": {
                "logical_name": "건강검진 수검자 수",
                "description": "건강검진을 받은 고유 인원 수",
            },
        },
    )

    assert result["chart_type"] == "grouped_bar"
    assert [trace["name"] for trace in result["figure"]["data"]] == ["2018년", "2019년"]
    assert result["figure"]["data"][0]["x"] == ["여성", "남성"]
    assert result["figure"]["data"][0]["y"] == [140_622, 136_925]
    assert result["figure"]["data"][0]["marker"]["color"] == "#B5121B"
    assert result["figure"]["layout"]["xaxis"]["title"]["text"] == "성별"
    assert result["figure"]["layout"]["yaxis"]["title"]["text"] == "건강검진 수검자 수"
    assert "검진 연도: 2018년" in result["figure"]["data"][0]["hovertemplate"]


def test_large_close_values_use_a_rounded_focused_axis_range() -> None:
    result = PlotlyChartGenerator().generate_chart(
        pd.DataFrame({
            "exam_year": [2018, 2019, 2020],
            "participant_count": [92_096, 92_251, 93_200],
        }),
        "연도별 수검자 추이",
        column_metadata={
            "exam_year": {
                "logical_name": "검진 연도",
                "sql_role": "time",
                "semantic_type": "year",
            },
            "participant_count": {"logical_name": "수검자 수"},
        },
    )

    yaxis = result["figure"]["layout"]["yaxis"]
    assert yaxis["range"][0] < 92_096
    assert yaxis["range"][0] > 0
    assert yaxis["range"][1] > 93_200
    assert yaxis["dtick"] in {100, 200, 500, 1_000}
    assert yaxis["tickformat"] == ",d"
