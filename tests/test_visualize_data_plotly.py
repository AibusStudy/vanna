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
    result = await VisualizeDataTool(file_system=CsvFileSystem()).execute(
        SimpleNamespace(),
        VisualizeDataArgs(filename="query_results.csv", title="Sales by region"),
    )

    assert result.success is True
    assert isinstance(result.ui_component.rich_component, ChartComponent)
    assert result.ui_component.rich_component.chart_type == "plotly"
    assert result.ui_component.rich_component.data["data"][0]["type"] == "bar"
    assert result.ui_component.rich_component.config["source_file"] == "query_results.csv"
    assert result.ui_component.rich_component.config["responsive"] is True
    assert isinstance(result.ui_component.simple_component, SimpleTextComponent)
    assert result.metadata["source_format"] == "csv"
    assert result.metadata["visualization_format"] == "plotly_json"
