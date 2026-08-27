"""SQL statement type detection helpers."""

from collections.abc import Callable
from typing import cast

import sqlparse


def get_statement_type(sql: str) -> str:
    """Return the type of the first SQL statement."""
    statements = sqlparse.parse(sql)
    if not statements:
        return "UNKNOWN"

    get_type = cast(Callable[[], str], statements[0].get_type)
    return get_type().upper()
