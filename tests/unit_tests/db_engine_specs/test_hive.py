# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


from datetime import datetime
from typing import Optional
from unittest import mock

import pandas as pd
import pytest
from pytest_mock import MockerFixture
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.engine.url import make_url

from superset.exceptions import SupersetException
from superset.sql.parse import Table
from tests.unit_tests.db_engine_specs.utils import assert_convert_dttm
from tests.unit_tests.fixtures.common import dttm  # noqa: F401


@pytest.mark.parametrize(
    "target_type,expected_result",
    [
        ("Date", "CAST('2019-01-02' AS DATE)"),
        (
            "TimeStamp",
            "CAST('2019-01-02 03:04:05.678900' AS TIMESTAMP)",
        ),
        ("UnknownType", None),
    ],
)
def test_convert_dttm(
    target_type: str,
    expected_result: Optional[str],
    dttm: datetime,  # noqa: F811
) -> None:
    from superset.db_engine_specs.hive import HiveEngineSpec as spec  # noqa: N813

    assert_convert_dttm(spec, target_type, expected_result, dttm)


def test_get_schema_from_engine_params() -> None:
    """
    Test the ``get_schema_from_engine_params`` method.
    """
    from superset.db_engine_specs.hive import HiveEngineSpec

    assert (
        HiveEngineSpec.get_schema_from_engine_params(
            make_url("hive://localhost:10000/default"), {}
        )
        == "default"
    )


def test_select_star(mocker: MockerFixture) -> None:
    """
    Test the ``select_star`` method.
    """
    from superset.db_engine_specs.hive import HiveEngineSpec

    database = mocker.MagicMock()
    dialect = mocker.MagicMock()

    def quote_table(table: Table, dialect: Dialect) -> str:
        return ".".join(
            part for part in (table.catalog, table.schema, table.table) if part
        )

    mocker.patch.object(HiveEngineSpec, "quote_table", quote_table)

    HiveEngineSpec.select_star(
        database=database,
        table=Table("my_table", "my_schema", "my_catalog"),
        dialect=dialect,
        limit=100,
        show_cols=False,
        indent=True,
        latest_partition=False,
        cols=None,
    )

    query = database.compile_sqla_query.mock_calls[0][1][0]
    assert (
        str(query)
        == """
SELECT * \nFROM my_schema.my_table
 LIMIT :param_1
    """.strip()
    )


@pytest.mark.parametrize(
    "identifier",
    [
        "good_name",
        "GoodName",
        "_with_leading_underscore",
        "schema1",
        "x",
    ],
)
def test_validate_hive_identifier_accepts_safe_names(identifier: str) -> None:
    from superset.db_engine_specs.hive import _validate_hive_identifier

    assert _validate_hive_identifier(identifier) == identifier


@pytest.mark.parametrize(
    "identifier",
    [
        "1starts_with_digit",
        "has space",
        "has-dash",
        "has.dot",
        "has;semicolon",
        "has`backtick",
        "with'quote",
        "evil); DROP TABLE users;--",
        "",
    ],
)
def test_validate_hive_identifier_rejects_unsafe_names(identifier: str) -> None:
    from superset.db_engine_specs.hive import _validate_hive_identifier

    with pytest.raises(SupersetException):
        _validate_hive_identifier(identifier)


def test_validate_hive_identifier_rejects_non_string() -> None:
    from superset.db_engine_specs.hive import _validate_hive_identifier

    with pytest.raises(SupersetException):
        _validate_hive_identifier(None)


@pytest.mark.parametrize(
    "name,unsafe",
    [
        ("col`backtick", True),
        ("col;semicolon", True),
        ("col\nwith_newline", True),
        ("", True),
        ("normal_col", False),
        ("col with space", False),
        ("col-with-dash", False),
    ],
)
def test_quote_hive_column_rejects_dangerous_chars(name: str, unsafe: bool) -> None:
    from superset.db_engine_specs.hive import _quote_hive_column

    if unsafe:
        with pytest.raises(SupersetException):
            _quote_hive_column(name)
    else:
        assert _quote_hive_column(name) == f"`{name}`"


def test_quote_hive_table_validates_all_components() -> None:
    from superset.db_engine_specs.hive import _quote_hive_table

    assert _quote_hive_table(Table("t")) == "`t`"
    assert _quote_hive_table(Table("t", "s")) == "`s`.`t`"
    assert _quote_hive_table(Table("t", "s", "c")) == "`c`.`s`.`t`"

    with pytest.raises(SupersetException):
        _quote_hive_table(Table("evil; DROP TABLE x; --"))
    with pytest.raises(SupersetException):
        _quote_hive_table(Table("ok", "evil schema"))


@mock.patch("superset.db_engine_specs.hive.g", spec={})
def test_df_to_sql_rejects_sql_injection_in_table_name(mock_g) -> None:
    """
    Verify that ``df_to_sql`` rejects malicious table names instead of
    interpolating them into SQL DDL. Regression test for issue #44 /
    Semgrep ``avoid-sqlalchemy-text`` finding in
    ``superset/db_engine_specs/hive.py``.
    """
    from superset.db_engine_specs.hive import HiveEngineSpec

    mock_g.user = True
    mock_database = mock.MagicMock()
    payload = "users`); DROP TABLE users; --"

    with pytest.raises(SupersetException, match="Invalid Hive identifier"):
        HiveEngineSpec.df_to_sql(
            mock_database,
            Table(table=payload),
            pd.DataFrame(),
            {"if_exists": "replace"},
        )

    # The malicious DDL must never reach the database.
    engine_cm = mock_database.get_sqla_engine.return_value
    engine = engine_cm.__enter__.return_value
    engine.execute.assert_not_called()
    mock_database.get_df.assert_not_called()


@mock.patch("superset.db_engine_specs.hive.g", spec={})
def test_df_to_sql_rejects_sql_injection_in_schema_name(mock_g) -> None:
    from superset.db_engine_specs.hive import HiveEngineSpec

    mock_g.user = True
    mock_database = mock.MagicMock()

    with pytest.raises(SupersetException, match="Invalid Hive identifier"):
        HiveEngineSpec.df_to_sql(
            mock_database,
            Table(table="ok", schema="evil`; DROP TABLE x; --"),
            pd.DataFrame(),
            {"if_exists": "fail"},
        )


@mock.patch("superset.db_engine_specs.hive.g", spec={})
def test_df_to_sql_rejects_sql_injection_in_column_name(mock_g) -> None:
    from superset.db_engine_specs.hive import HiveEngineSpec

    mock_g.user = True
    mock_database = mock.MagicMock()
    mock_database.get_df.return_value.empty = True

    df = pd.DataFrame({"safe_col": [1], "evil`); DROP TABLE x; --": [2]})

    with pytest.raises(SupersetException, match="Invalid Hive column name"):
        HiveEngineSpec.df_to_sql(
            mock_database,
            Table(table="ok"),
            df,
            {"if_exists": "fail"},
        )
