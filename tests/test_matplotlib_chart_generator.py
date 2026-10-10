"""Tests for deterministic Matplotlib visualization output."""

import base64
import io

import pandas as pd
from PIL import Image

from vanna.integrations.matplotlib import MatplotlibChartGenerator


def test_generates_png_bar_chart() -> None:
    result = MatplotlibChartGenerator().generate_chart(
        pd.DataFrame({"region": ["East", "West"], "sales": [10, 20]}),
        "Sales by region",
    )

    assert result["chart_type"] == "bar"
    assert result["format"] == "png"
    prefix, encoded = result["data_url"].split(",", 1)
    assert prefix == "data:image/png;base64"
    assert base64.b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n")


def test_infers_named_date_column_as_time_series() -> None:
    result = MatplotlibChartGenerator().generate_chart(
        pd.DataFrame(
            {
                "order_date": ["2026-01-01", "2026-02-01"],
                "sales": [10, 20],
            }
        )
    )

    assert result["chart_type"] == "line"


def test_single_row_numeric_result_uses_kpi_visualization() -> None:
    result = MatplotlibChartGenerator().generate_chart(
        pd.DataFrame({"female": [47_235], "male": [45_965]}),
        "검진 인원",
    )

    assert result["chart_type"] == "kpi"


def test_png_background_is_transparent() -> None:
    result = MatplotlibChartGenerator().generate_chart(
        pd.DataFrame({"category": ["A", "B"], "value": [10, 20]})
    )
    encoded = result["data_url"].split(",", 1)[1]

    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        assert image.mode == "RGBA"
        assert image.getpixel((0, 0))[3] == 0
