"""Tests for deterministic Matplotlib visualization output."""

import base64

import pandas as pd

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
