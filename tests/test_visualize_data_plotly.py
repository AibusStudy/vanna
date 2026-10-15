"""Tests for CSV-to-Plotly visualization tool output."""

from types import SimpleNamespace

import pytest

from vanna.components import ChartComponent, SimpleTextComponent
from vanna.tools.visualize_data import VisualizeDataArgs, VisualizeDataTool


class CsvFileSystem:
    async def read_file(self, filename, context):
        assert filename == "query_results.csv"
        return "region,sales\nEast,10\nWest,20\n"


@pytest.mark.asyncio
async def test_visualize_data_returns_plotly_chart_from_csv() -> None:
    context = SimpleNamespace(metadata={
        "main_workflow": {
            "metadata": {"candidates": [{
                "column_name": "sales",
                "column_logical_name": "매출액",
                "column_desc": "지역별 매출 합계",
            }]},
            "structured_question": {},
        }
    })
    result = await VisualizeDataTool(file_system=CsvFileSystem()).execute(
        context,
        VisualizeDataArgs(filename="query_results.csv", title="Sales by region"),
    )

    assert result.success is True
    assert isinstance(result.ui_component.rich_component, ChartComponent)
    assert result.ui_component.rich_component.chart_type == "plotly"
    assert result.ui_component.rich_component.data["data"][0]["type"] == "bar"
    assert result.ui_component.rich_component.data["layout"]["yaxis"]["title"]["text"] == "매출액"
    assert result.ui_component.rich_component.config["source_file"] == "query_results.csv"
    assert result.ui_component.rich_component.config["responsive"] is True
    assert isinstance(result.ui_component.simple_component, SimpleTextComponent)
    assert result.metadata["source_format"] == "csv"
    assert result.metadata["visualization_format"] == "plotly_json"


def test_chart_title_is_summarized_from_structured_question() -> None:
    context = SimpleNamespace(metadata={
        "main_workflow": {
            "structured_question": {
                "target_entity": {"entity_meaning": "건강검진 받은 사람"},
                "group_by": [
                    {"original_text": "연도별"},
                    {"original_text": "성별"},
                ],
                "time_conditions": [{
                    "start_value": "20180101",
                    "end_value": "20201231",
                }],
            }
        }
    })

    title = VisualizeDataTool._chart_title(
        context,
        "2018년부터 2020년까지 성별·연도별 건강검진 받은 사람의 추이를 보여줘.",
    )

    assert title == "2018년부터 2020년까지 연도별·성별 건강검진 수검자 추이"
