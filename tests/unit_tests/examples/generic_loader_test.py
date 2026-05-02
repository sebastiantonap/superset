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
"""Tests for generic_loader.py UUID threading functionality."""

from unittest.mock import MagicMock, patch

import pytest


@patch("superset.examples.generic_loader.get_example_database")
@patch("superset.examples.generic_loader.db")
def test_load_parquet_table_sets_uuid_on_new_table(mock_db, mock_get_db):
    """Test that load_parquet_table sets UUID on newly created SqlaTable."""
    from superset.examples.generic_loader import load_parquet_table

    mock_database = MagicMock()
    mock_database.id = 1
    mock_database.has_table.return_value = True
    mock_get_db.return_value = mock_database

    mock_engine = MagicMock()
    mock_inspector = MagicMock()
    mock_inspector.default_schema_name = "public"
    mock_database.get_sqla_engine.return_value.__enter__ = MagicMock(
        return_value=mock_engine
    )
    mock_database.get_sqla_engine.return_value.__exit__ = MagicMock(return_value=False)

    # Simulate table not found in metadata
    mock_db.session.query.return_value.filter_by.return_value.first.return_value = None

    test_uuid = "12345678-1234-1234-1234-123456789012"

    with patch("superset.examples.generic_loader.inspect") as mock_inspect:
        mock_inspect.return_value = mock_inspector

        tbl = load_parquet_table(
            parquet_file="test_data",
            table_name="test_table",
            database=mock_database,
            only_metadata=True,
            uuid=test_uuid,
        )

    assert tbl.uuid == test_uuid


@patch("superset.examples.generic_loader.get_example_database")
@patch("superset.examples.generic_loader.db")
def test_load_parquet_table_early_return_does_not_modify_existing_uuid(
    mock_db, mock_get_db
):
    """Test early return path when table exists - UUID is not modified.

    When the physical table exists and force=False, the function returns early
    without going through the full load path. The existing table's UUID is
    preserved as-is (not modified even if different from the provided uuid).
    """
    from superset.examples.generic_loader import load_parquet_table

    mock_database = MagicMock()
    mock_database.id = 1
    mock_database.has_table.return_value = True  # Triggers early return
    mock_get_db.return_value = mock_database

    mock_engine = MagicMock()
    mock_inspector = MagicMock()
    mock_inspector.default_schema_name = "public"
    mock_database.get_sqla_engine.return_value.__enter__ = MagicMock(
        return_value=mock_engine
    )
    mock_database.get_sqla_engine.return_value.__exit__ = MagicMock(return_value=False)

    # Simulate existing table without UUID
    existing_table = MagicMock()
    existing_table.uuid = None
    mock_db.session.query.return_value.filter_by.return_value.first.return_value = (
        existing_table
    )

    test_uuid = "12345678-1234-1234-1234-123456789012"

    with patch("superset.examples.generic_loader.inspect") as mock_inspect:
        mock_inspect.return_value = mock_inspector

        tbl = load_parquet_table(
            parquet_file="test_data",
            table_name="test_table",
            database=mock_database,
            only_metadata=True,
            uuid=test_uuid,
        )

    # Early return path returns existing table as-is
    assert tbl is existing_table
    # UUID was not modified (still None)
    assert tbl.uuid is None


@patch("superset.examples.generic_loader.get_example_database")
@patch("superset.examples.generic_loader.db")
def test_load_parquet_table_preserves_existing_uuid(mock_db, mock_get_db):
    """Test that load_parquet_table does not overwrite existing UUID."""
    from superset.examples.generic_loader import load_parquet_table

    mock_database = MagicMock()
    mock_database.id = 1
    mock_database.has_table.return_value = True
    mock_get_db.return_value = mock_database

    mock_engine = MagicMock()
    mock_inspector = MagicMock()
    mock_inspector.default_schema_name = "public"
    mock_database.get_sqla_engine.return_value.__enter__ = MagicMock(
        return_value=mock_engine
    )
    mock_database.get_sqla_engine.return_value.__exit__ = MagicMock(return_value=False)

    # Simulate existing table with different UUID
    existing_uuid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    existing_table = MagicMock()
    existing_table.uuid = existing_uuid
    mock_db.session.query.return_value.filter_by.return_value.first.return_value = (
        existing_table
    )

    new_uuid = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

    with patch("superset.examples.generic_loader.inspect") as mock_inspect:
        mock_inspect.return_value = mock_inspector

        tbl = load_parquet_table(
            parquet_file="test_data",
            table_name="test_table",
            database=mock_database,
            only_metadata=True,
            uuid=new_uuid,
        )

    # Should preserve original UUID
    assert tbl.uuid == existing_uuid


@patch("superset.examples.generic_loader.get_example_database")
@patch("superset.examples.generic_loader.db")
def test_load_parquet_table_works_without_uuid(mock_db, mock_get_db):
    """Test that load_parquet_table works correctly when no UUID is provided."""
    from superset.examples.generic_loader import load_parquet_table

    mock_database = MagicMock()
    mock_database.id = 1
    mock_database.has_table.return_value = True
    mock_get_db.return_value = mock_database

    mock_engine = MagicMock()
    mock_inspector = MagicMock()
    mock_inspector.default_schema_name = "public"
    mock_database.get_sqla_engine.return_value.__enter__ = MagicMock(
        return_value=mock_engine
    )
    mock_database.get_sqla_engine.return_value.__exit__ = MagicMock(return_value=False)

    # Simulate table not found
    mock_db.session.query.return_value.filter_by.return_value.first.return_value = None

    with patch("superset.examples.generic_loader.inspect") as mock_inspect:
        mock_inspect.return_value = mock_inspector

        tbl = load_parquet_table(
            parquet_file="test_data",
            table_name="test_table",
            database=mock_database,
            only_metadata=True,
            # No uuid parameter
        )

    # UUID should remain None
    assert tbl.uuid is None


def test_create_generic_loader_passes_uuid():
    """Test that create_generic_loader passes UUID to load_parquet_table."""
    from superset.examples.generic_loader import create_generic_loader

    test_uuid = "12345678-1234-1234-1234-123456789012"
    loader = create_generic_loader(
        parquet_file="test_data",
        table_name="test_table",
        uuid=test_uuid,
    )

    # Verify loader was created with UUID in closure
    with patch("superset.examples.generic_loader.load_parquet_table") as mock_load:
        mock_load.return_value = MagicMock()

        loader(only_metadata=True)

        # Verify UUID was passed through
        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args[1]
        assert call_kwargs["uuid"] == test_uuid


def test_create_generic_loader_without_uuid():
    """Test that create_generic_loader works without UUID (backward compat)."""
    from superset.examples.generic_loader import create_generic_loader

    loader = create_generic_loader(
        parquet_file="test_data",
        table_name="test_table",
        # No uuid
    )

    with patch("superset.examples.generic_loader.load_parquet_table") as mock_load:
        mock_load.return_value = MagicMock()

        loader(only_metadata=True)

        mock_load.assert_called_once()
        call_kwargs = mock_load.call_args[1]
        assert call_kwargs["uuid"] is None


# ---------------------------------------------------------------------------
# Schema-name SQL-injection regression tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "schema",
    [
        '"; DROP TABLE users; --',
        '" OR 1=1 --',
        'evil"; DROP TABLE foo; --',
        "schema with space",
        "schema-with-dash",
        "1leading_digit",
        "",
        "schema;",
        "public;DROP",
        '"',
        "schema/comment",
    ],
)
def test_ensure_schema_exists_rejects_injection_payloads(schema):
    """Malicious / malformed schema names must be rejected before any SQL runs."""
    from superset.examples.generic_loader import _ensure_schema_exists

    engine = MagicMock()

    with pytest.raises(ValueError, match="Invalid schema name"):
        _ensure_schema_exists(engine, schema)

    # No DDL should ever be issued for a rejected schema name.
    engine.begin.assert_not_called()


@pytest.mark.parametrize("schema", [None, 123, ["public"], {"name": "public"}])
def test_ensure_schema_exists_rejects_non_string_input(schema):
    """Non-string schema arguments must raise rather than be coerced into SQL."""
    from superset.examples.generic_loader import _ensure_schema_exists

    engine = MagicMock()

    with pytest.raises(ValueError, match="Invalid schema name"):
        _ensure_schema_exists(engine, schema)

    engine.begin.assert_not_called()


def test_ensure_schema_exists_skips_create_when_schema_exists():
    """If the schema already exists, no CREATE SCHEMA DDL should be executed."""
    from superset.examples.generic_loader import _ensure_schema_exists

    engine = MagicMock()
    inspector = MagicMock()
    inspector.get_schema_names.return_value = ["public", "analytics"]

    with patch("superset.examples.generic_loader.inspect", return_value=inspector):
        _ensure_schema_exists(engine, "analytics")

    engine.begin.assert_not_called()


def test_ensure_schema_exists_uses_create_schema_ddl():
    """When the schema is absent, CreateSchema DDL is used instead of raw text()."""
    from superset.examples.generic_loader import _ensure_schema_exists

    engine = MagicMock()
    inspector = MagicMock()
    inspector.get_schema_names.return_value = ["public"]

    conn = MagicMock()
    engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)

    with patch("superset.examples.generic_loader.inspect", return_value=inspector):
        _ensure_schema_exists(engine, "analytics_new")

    # Exactly one DDL execution, dispatched via SQLAlchemy's CreateSchema element.
    conn.execute.assert_called_once()
    (executed,), _ = conn.execute.call_args
    from sqlalchemy.schema import CreateSchema

    assert isinstance(executed, CreateSchema)
    # The element must carry the validated, untouched identifier.
    assert executed.element == "analytics_new"


def test_ensure_schema_exists_swallows_concurrent_create_race():
    """If another process created the schema in the meantime, swallow the error.

    Replicates a TOCTOU race: the inspector reports the schema as missing,
    so we attempt CREATE SCHEMA, the database raises a duplicate-schema
    error (because another process won the race), and a fresh inspector
    confirms the schema now exists. The helper must treat this as success
    and not propagate the error — preserving the IF NOT EXISTS semantics
    of the original raw DDL.
    """
    from sqlalchemy.exc import ProgrammingError

    from superset.examples.generic_loader import _ensure_schema_exists

    engine = MagicMock()
    conn = MagicMock()
    engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.side_effect = ProgrammingError(
        "CREATE SCHEMA analytics", {}, Exception('schema "analytics" already exists')
    )

    inspector_before = MagicMock()
    inspector_before.get_schema_names.return_value = ["public"]
    inspector_after = MagicMock()
    inspector_after.get_schema_names.return_value = ["public", "analytics"]

    with patch(
        "superset.examples.generic_loader.inspect",
        side_effect=[inspector_before, inspector_after],
    ):
        _ensure_schema_exists(engine, "analytics")  # must NOT raise

    conn.execute.assert_called_once()


def test_ensure_schema_exists_reraises_real_db_errors():
    """Genuine database errors (not duplicate-schema races) must propagate."""
    from sqlalchemy.exc import ProgrammingError

    from superset.examples.generic_loader import _ensure_schema_exists

    engine = MagicMock()
    conn = MagicMock()
    engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)
    conn.execute.side_effect = ProgrammingError(
        "CREATE SCHEMA analytics", {}, Exception("permission denied for database")
    )

    inspector_before = MagicMock()
    inspector_before.get_schema_names.return_value = ["public"]
    # Even after the error the schema is still absent -> not a race.
    inspector_after = MagicMock()
    inspector_after.get_schema_names.return_value = ["public"]

    with patch(
        "superset.examples.generic_loader.inspect",
        side_effect=[inspector_before, inspector_after],
    ):
        with pytest.raises(ProgrammingError):
            _ensure_schema_exists(engine, "analytics")


def test_ensure_schema_exists_create_schema_quotes_malicious_identifier_safely():
    """SQLAlchemy's CreateSchema must double-quote embedded quote characters.

    The schema identifier itself is validated against an allow-list, but this
    test additionally pins the underlying SQLAlchemy behaviour we rely on:
    even if the allow-list ever changed, ``CreateSchema`` would still emit
    safely quoted DDL rather than allowing identifier-level injection.
    """
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateSchema

    payload = 'evil"; DROP TABLE users; --'
    compiled = str(CreateSchema(payload).compile(dialect=postgresql.dialect()))
    # Embedded double-quote is doubled, so the entire payload stays inside a
    # single quoted identifier instead of breaking out into a new statement.
    assert compiled == 'CREATE SCHEMA "evil""; DROP TABLE users; --"'
    # Balanced quoting is what guarantees the closing quote isn't escaped
    # away — an odd number would mean the identifier "leaks" into raw SQL.
    assert compiled.count('"') % 2 == 0
    # The compiled DDL must be a single ``CREATE SCHEMA "..."`` statement;
    # any ``;`` characters in the payload remain inside the quoted identifier.
    assert compiled.startswith('CREATE SCHEMA "')
    assert compiled.endswith('"')


@patch("superset.examples.generic_loader.get_example_database")
@patch("superset.examples.generic_loader.db")
def test_load_parquet_table_rejects_injected_schema(mock_db, mock_get_db):
    """End-to-end: a SQL-injection schema payload aborts the loader safely."""
    from superset.examples.generic_loader import load_parquet_table

    mock_database = MagicMock()
    mock_database.id = 1
    mock_get_db.return_value = mock_database

    mock_engine = MagicMock()
    mock_database.get_sqla_engine.return_value.__enter__ = MagicMock(
        return_value=mock_engine
    )
    mock_database.get_sqla_engine.return_value.__exit__ = MagicMock(return_value=False)

    with pytest.raises(ValueError, match="Invalid schema name"):
        load_parquet_table(
            parquet_file="test_data",
            table_name="test_table",
            database=mock_database,
            only_metadata=True,
            schema='"; DROP TABLE users; --',
        )

    # No DDL or table-existence check should have been attempted.
    mock_engine.begin.assert_not_called()
    mock_database.has_table.assert_not_called()
