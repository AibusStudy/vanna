"""SQL statement type detection helpers."""

from collections.abc import Callable
from typing import cast

import sqlparse
from sqlparse import tokens
from sqlparse.sql import Token


def get_statement_type(sql: str) -> str:
    """Return the type of the first SQL statement."""
    statements = sqlparse.parse(sql)
    if not statements:
        return "UNKNOWN"

    statement = statements[0]
    token_first = cast(Callable[..., Token | None], statement.token_first)
    first_token = token_first(skip_cm=True)

    if first_token is not None and first_token.ttype is tokens.Keyword.CTE:
        for token in statement.tokens:
            if token.ttype is tokens.Keyword.DML:
                return str(token.normalized).upper()

    get_type = cast(Callable[[], str], statement.get_type)
    return get_type().upper()
