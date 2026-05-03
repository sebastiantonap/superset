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

import pytest
from sqlalchemy.dialects import postgresql

from superset.migrations.shared.utils import (
    _safe_quoted_identifier,
    cast_json_column_to_text,
    cast_text_column_to_json,
    create_index,
    drop_index,
)


# ----- Dummy classes for capturing calls ----- #
class DummyLogger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        # Handle lazy logging format with multiple arguments
        if args:
            formatted_message = message % args
        else:
            formatted_message = message
        self.messages.append(formatted_message)


class DummyOp:
    def __init__(self):
        self.called = False
        self.call_kwargs = None

    def create_index(self, **kwargs):
        self.called = True
        self.call_kwargs = kwargs

    def drop_index(self, **kwargs):
        self.called = True
        self.call_kwargs = kwargs


# ----- Fake functions to simulate table index checks ----- #
def fake_table_has_index_true(*args, **kwargs):
    return True


def fake_table_has_index_false(*args, **kwargs):
    return False


# ----- Tests for create_index ----- #
def test_create_index_skips_if_index_exists(monkeypatch):
    dummy_logger = DummyLogger()
    dummy_op = DummyOp()

    # Patch globals in the module where create_index is defined.
    monkeypatch.setattr("superset.migrations.shared.utils.logger", dummy_logger)
    monkeypatch.setattr("superset.migrations.shared.utils.op", dummy_op)
    monkeypatch.setattr(
        "superset.migrations.shared.utils.table_has_index", fake_table_has_index_true
    )

    table_name = "test_table"
    index_name = "idx_test"
    columns = ["col1", "col2"]

    create_index(table_name, index_name, columns, unique=True)

    # When the index already exists, op.create_index should not be called.
    assert dummy_op.called is False
    # And a log message mentioning "already has index" should be generated.
    assert any("already has index" in msg for msg in dummy_logger.messages)


def test_create_index_creates_index(monkeypatch):
    dummy_logger = DummyLogger()
    dummy_op = DummyOp()

    monkeypatch.setattr("superset.migrations.shared.utils.logger", dummy_logger)
    monkeypatch.setattr("superset.migrations.shared.utils.op", dummy_op)
    monkeypatch.setattr(
        "superset.migrations.shared.utils.table_has_index", fake_table_has_index_false
    )

    table_name = "test_table"
    index_name = "idx_test"
    columns = ["col1", "col2"]

    create_index(table_name, index_name, columns, unique=False)

    # When the index does not exist, op.create_index should be called.
    assert dummy_op.called is True
    call_kwargs = dummy_op.call_kwargs
    assert call_kwargs.get("table_name") == table_name
    assert call_kwargs.get("index_name") == index_name
    assert call_kwargs.get("unique") is False
    assert call_kwargs.get("columns") == columns
    # And a log message mentioning "Creating index" should be generated.
    assert any("Creating index" in msg for msg in dummy_logger.messages)


def test_create_unique_index_creates_index(monkeypatch):
    dummy_logger = DummyLogger()
    dummy_op = DummyOp()

    monkeypatch.setattr("superset.migrations.shared.utils.logger", dummy_logger)
    monkeypatch.setattr("superset.migrations.shared.utils.op", dummy_op)
    monkeypatch.setattr(
        "superset.migrations.shared.utils.table_has_index", fake_table_has_index_false
    )

    table_name = "test_table"
    index_name = "idx_test"
    columns = ["col1", "col2"]

    create_index(table_name, index_name, columns, unique=True)

    # When the index does not exist, op.create_index should be called.
    assert dummy_op.called is True
    call_kwargs = dummy_op.call_kwargs
    assert call_kwargs.get("table_name") == table_name
    assert call_kwargs.get("index_name") == index_name
    assert call_kwargs.get("unique") is True
    assert call_kwargs.get("columns") == columns
    # And a log message mentioning "Creating index" should be generated.
    assert any("Creating index" in msg for msg in dummy_logger.messages)


def test_create_index_with_not_unique(monkeypatch):
    dummy_logger = DummyLogger()
    dummy_op = DummyOp()

    monkeypatch.setattr("superset.migrations.shared.utils.logger", dummy_logger)
    monkeypatch.setattr("superset.migrations.shared.utils.op", dummy_op)
    monkeypatch.setattr(
        "superset.migrations.shared.utils.table_has_index", fake_table_has_index_false
    )

    table_name = "test_table"
    index_name = "idx_test"
    columns = ["col1", "col2"]

    create_index(table_name, index_name, columns, unique=False)

    # When the index does not exist, op.create_index should be called.
    assert dummy_op.called is True
    call_kwargs = dummy_op.call_kwargs
    assert call_kwargs.get("table_name") == table_name
    assert call_kwargs.get("index_name") == index_name
    assert call_kwargs.get("unique") is False
    assert call_kwargs.get("columns") == columns


# ----- Tests for drop_index ----- #
def test_drop_index_skips_if_index_not_exist(monkeypatch):
    dummy_logger = DummyLogger()
    dummy_op = DummyOp()

    monkeypatch.setattr("superset.migrations.shared.utils.logger", dummy_logger)
    monkeypatch.setattr("superset.migrations.shared.utils.op", dummy_op)
    monkeypatch.setattr(
        "superset.migrations.shared.utils.table_has_index", fake_table_has_index_false
    )

    table_name = "test_table"
    index_name = "idx_test"

    drop_index(table_name, index_name)

    # When the index does not exist, op.drop_index should not be called.
    assert dummy_op.called is False
    # And a log message mentioning "doesn't have index" should be generated.
    assert any("doesn't have index" in msg for msg in dummy_logger.messages)


def test_drop_index_drops_index_when_exists(monkeypatch):
    dummy_logger = DummyLogger()
    dummy_op = DummyOp()

    monkeypatch.setattr("superset.migrations.shared.utils.logger", dummy_logger)
    monkeypatch.setattr("superset.migrations.shared.utils.op", dummy_op)
    monkeypatch.setattr(
        "superset.migrations.shared.utils.table_has_index", fake_table_has_index_true
    )

    table_name = "test_table"
    index_name = "idx_test"

    drop_index(table_name, index_name)

    # When the index exists, op.drop_index should be called.
    assert dummy_op.called is True
    call_kwargs = dummy_op.call_kwargs
    assert call_kwargs.get("table_name") == table_name
    assert call_kwargs.get("index_name") == index_name
    # And a log message mentioning "Dropping index" should be generated.
    assert any("Dropping index" in msg for msg in dummy_logger.messages)


# ----- Tests for SQL identifier validation ----- #
class _DummyConn:
    """Minimal stand-in for an Alembic/SQLAlchemy connection."""

    def __init__(self, dialect):
        self.dialect = dialect


class _RecordingOp:
    """Captures op.execute() calls so we can assert on the SQL emitted."""

    def __init__(self, conn):
        self._conn = conn
        self.executed = []

    def get_bind(self):
        return self._conn

    def execute(self, statement):
        self.executed.append(str(statement))


@pytest.mark.parametrize(
    "name",
    ["users", "_users", "u123", "User_Table"],
)
def test_safe_quoted_identifier_accepts_valid_names(name):
    dialect = postgresql.dialect()
    quoted = _safe_quoted_identifier(dialect, name)
    # The dialect quoting may add double quotes, but the original name
    # must always be embedded somewhere in the result.
    assert name in quoted


@pytest.mark.parametrize(
    "name",
    [
        "1abc",
        "users; DROP TABLE users",
        "'; DROP TABLE users; --",
        "users--",
        "users.column",
        "users column",
        '"users"',
        "",
        None,
    ],
)
def test_safe_quoted_identifier_rejects_unsafe_names(name):
    dialect = postgresql.dialect()
    with pytest.raises(ValueError, match="Unsafe SQL identifier"):
        _safe_quoted_identifier(dialect, name)


def test_cast_text_column_to_json_rejects_sqli_payload(monkeypatch):
    """A SQL injection payload in the table or column name must be rejected
    before any SQL is executed."""
    conn = _DummyConn(postgresql.dialect())
    op_recorder = _RecordingOp(conn)
    monkeypatch.setattr("superset.migrations.shared.utils.op", op_recorder)

    payload = "users; DROP TABLE users; --"
    with pytest.raises(ValueError, match="Unsafe SQL identifier"):
        cast_text_column_to_json(payload, "currency")
    with pytest.raises(ValueError, match="Unsafe SQL identifier"):
        cast_text_column_to_json("sql_metrics", payload)

    # No SQL must have been executed against the connection.
    assert op_recorder.executed == []


def test_cast_json_column_to_text_rejects_sqli_payload(monkeypatch):
    conn = _DummyConn(postgresql.dialect())
    op_recorder = _RecordingOp(conn)
    monkeypatch.setattr("superset.migrations.shared.utils.op", op_recorder)

    payload = "'; DROP TABLE sql_metrics; --"
    with pytest.raises(ValueError, match="Unsafe SQL identifier"):
        cast_json_column_to_text(payload, "currency")
    with pytest.raises(ValueError, match="Unsafe SQL identifier"):
        cast_json_column_to_text("sql_metrics", payload)

    assert op_recorder.executed == []


def test_cast_text_column_to_json_emits_quoted_sql(monkeypatch):
    """Valid identifiers must be quoted by the dialect identifier preparer
    before being embedded in the DDL."""
    conn = _DummyConn(postgresql.dialect())
    op_recorder = _RecordingOp(conn)
    monkeypatch.setattr("superset.migrations.shared.utils.op", op_recorder)

    cast_text_column_to_json("sql_metrics", "currency")

    # Two statements: the helper function, then the ALTER TABLE.
    assert len(op_recorder.executed) == 2
    function_sql, alter_sql = op_recorder.executed
    assert "CREATE OR REPLACE FUNCTION safe_to_jsonb" in function_sql
    assert "ALTER TABLE sql_metrics" in alter_sql
    assert "ALTER COLUMN currency TYPE jsonb" in alter_sql
    # No raw payload-style characters should be present.
    assert "DROP TABLE" not in alter_sql
    assert ";" in alter_sql  # statement terminator


def test_cast_json_column_to_text_emits_quoted_sql(monkeypatch):
    conn = _DummyConn(postgresql.dialect())
    op_recorder = _RecordingOp(conn)
    monkeypatch.setattr("superset.migrations.shared.utils.op", op_recorder)

    cast_json_column_to_text("sql_metrics", "currency")

    assert len(op_recorder.executed) == 1
    alter_sql = op_recorder.executed[0]
    assert "ALTER TABLE sql_metrics" in alter_sql
    assert "ALTER COLUMN currency TYPE text" in alter_sql
    assert "currency::text" in alter_sql
