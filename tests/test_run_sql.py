from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from vanna.capabilities.sql_runner import RunSqlToolArgs
from vanna.capabilities.sql_runner.statement import get_statement_type
from vanna.components import DataFrameComponent
from vanna.integrations.postgres import PostgresRunner
from vanna.integrations.sqlite import SqliteRunner
from vanna.tools.run_sql import RunSqlTool


@pytest.mark.parametrize(
    ("sql", "expected_type"),
    [
        ("SELECT 1", "SELECT"),
        ("WITH result AS (SELECT 1) SELECT * FROM result", "SELECT"),
        (
            "WITH x AS (SELECT 1) SELECT * FROM x UNION ALL SELECT 2",
            "SELECT",
        ),
        (
            "-- comment\nWITH result AS (SELECT 1) SELECT * FROM result",
            "SELECT",
        ),
        (
            "WITH RECURSIVE x(n) AS ("
            "SELECT 1 UNION ALL SELECT n + 1 FROM x WHERE n < 3"
            ") SELECT * FROM x",
            "SELECT",
        ),
        ("WITH source AS (SELECT 1) UPDATE target SET value = 1", "UPDATE"),
        (
            "WITH data AS (SELECT 1) INSERT INTO target SELECT * FROM data",
            "INSERT",
        ),
        ("WITH result AS (SELECT 1) DELETE FROM target", "DELETE"),
        ("", "UNKNOWN"),
    ],
)
def test_get_statement_type(sql: str, expected_type: str) -> None:
    assert get_statement_type(sql) == expected_type


@pytest.mark.asyncio
async def test_cte_select_returns_dataframe_results_to_llm_ui_and_metadata() -> None:
    expected_results = [
        {"year": "2015", "patient_payment_ratio": 12.3},
        {"year": "2021", "patient_payment_ratio": 15.7},
    ]
    runner = SimpleNamespace(
        run_sql=AsyncMock(return_value=pd.DataFrame(expected_results))
    )
    file_system = SimpleNamespace(write_file=AsyncMock())
    context = SimpleNamespace()
    tool = RunSqlTool(runner, file_system=file_system)

    result = await tool.execute(
        context,
        RunSqlToolArgs(
            sql=(
                "WITH result AS ("
                "SELECT '2015' AS year, 12.3 AS patient_payment_ratio "
                "UNION ALL SELECT '2021', 15.7"
                ") SELECT * FROM result"
            )
        ),
    )

    assert result.success is True
    assert result.metadata["query_type"] == "SELECT"
    assert result.metadata["row_count"] == 2
    assert result.metadata["results"] == expected_results
    assert isinstance(result.metadata["output_file"], str)
    assert "2015" in result.result_for_llm
    assert "12.3" in result.result_for_llm
    assert "2021" in result.result_for_llm
    assert "15.7" in result.result_for_llm
    assert "row(s) affected" not in result.result_for_llm

    assert result.ui_component is not None
    dataframe_component = result.ui_component.rich_component
    assert isinstance(dataframe_component, DataFrameComponent)
    assert dataframe_component.rows == expected_results
    assert dataframe_component.columns == ["year", "patient_payment_ratio"]

    file_system.write_file.assert_awaited_once()
    write_call = file_system.write_file.await_args
    assert write_call.args[0] == result.metadata["output_file"]
    assert write_call.args[1].splitlines() == [
        "year,patient_payment_ratio",
        "2015,12.3",
        "2021,15.7",
    ]
    assert write_call.args[2] is context
    assert write_call.kwargs == {"overwrite": True}


@pytest.mark.asyncio
async def test_with_update_keeps_non_select_result_contract() -> None:
    runner = SimpleNamespace(
        run_sql=AsyncMock(return_value=pd.DataFrame([{"rows_affected": 1}]))
    )
    file_system = SimpleNamespace(write_file=AsyncMock())
    tool = RunSqlTool(runner, file_system=file_system)

    result = await tool.execute(
        SimpleNamespace(),
        RunSqlToolArgs(sql="WITH source AS (SELECT 1) UPDATE target SET value = 1"),
    )

    assert result.success is True
    assert result.metadata == {"rows_affected": 1, "query_type": "UPDATE"}
    assert result.result_for_llm == ("Query executed successfully. 1 row(s) affected.")
    file_system.write_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_sqlite_runner_fetches_cte_select_results() -> None:
    runner = SqliteRunner(":memory:")

    result = await runner.run_sql(
        RunSqlToolArgs(
            sql="WITH result AS (SELECT 1 AS value) SELECT value FROM result"
        ),
        SimpleNamespace(),
    )

    assert result.to_dict("records") == [{"value": 1}]


@pytest.mark.asyncio
async def test_postgres_runner_fetches_cte_select_without_commit() -> None:
    cursor = MagicMock()
    cursor.fetchall.return_value = [{"value": 1}]
    connection = MagicMock()
    connection.cursor.return_value = cursor
    real_dict_cursor = object()

    runner = PostgresRunner.__new__(PostgresRunner)
    runner.psycopg2 = SimpleNamespace(
        connect=MagicMock(return_value=connection),
        extras=SimpleNamespace(RealDictCursor=real_dict_cursor),
    )
    runner.connection_string = "postgresql://test"
    runner.connection_params = None
    sql = "WITH result AS (SELECT 1 AS value) SELECT value FROM result"

    result = await runner.run_sql(RunSqlToolArgs(sql=sql), SimpleNamespace())

    assert result.to_dict("records") == [{"value": 1}]
    connection.cursor.assert_called_once_with(cursor_factory=real_dict_cursor)
    cursor.execute.assert_called_once_with(sql)
    cursor.fetchall.assert_called_once_with()
    connection.commit.assert_not_called()
    cursor.close.assert_called_once_with()
    connection.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_postgres_runner_commits_cte_update_without_fetching() -> None:
    cursor = MagicMock()
    cursor.rowcount = 2
    connection = MagicMock()
    connection.cursor.return_value = cursor

    runner = PostgresRunner.__new__(PostgresRunner)
    runner.psycopg2 = SimpleNamespace(
        connect=MagicMock(return_value=connection),
        extras=SimpleNamespace(RealDictCursor=object()),
    )
    runner.connection_string = "postgresql://test"
    runner.connection_params = None
    sql = "WITH source AS (SELECT 1) UPDATE target SET value = 1"

    result = await runner.run_sql(RunSqlToolArgs(sql=sql), SimpleNamespace())

    assert result.to_dict("records") == [{"rows_affected": 2}]
    cursor.execute.assert_called_once_with(sql)
    connection.commit.assert_called_once_with()
    cursor.fetchall.assert_not_called()
    cursor.close.assert_called_once_with()
    connection.close.assert_called_once_with()
