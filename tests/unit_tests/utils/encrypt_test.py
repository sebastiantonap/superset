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
"""Unit tests for ``superset.utils.encrypt`` (issue #61).

These tests exercise :class:`SecretsMigrator` end-to-end against an
in-memory SQLite database and intentionally use a table name that would
trigger a SQL-injection if identifiers were ever interpolated as raw SQL.
The migrator must rely on SQLAlchemy Core constructs so all identifiers
are resolved from :class:`MetaData` and quoted by the dialect.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import (
    Column,
    create_engine,
    insert,
    Integer,
    LargeBinary,
    MetaData,
    select,
    String,
    Table,
)
from sqlalchemy.engine import Engine

from superset.utils import encrypt as encrypt_module
from superset.utils.encrypt import EncryptedType, SecretsMigrator

PREVIOUS_SECRET_KEY = "previous-secret-key"  # noqa: S105
NEW_SECRET_KEY = "new-secret-key"  # noqa: S105

# A name that is a valid SQLite identifier when properly quoted but that
# would break out of the statement if naively interpolated into SQL.
SUSPICIOUS_TABLE_NAME = 'weird"; DROP TABLE other; --'


def _build_migrator(
    monkeypatch: pytest.MonkeyPatch, table_name: str = "encrypted_secrets"
) -> tuple[SecretsMigrator, Engine, Table]:
    """Build a :class:`SecretsMigrator` wired to an in-memory SQLite DB."""
    engine = create_engine("sqlite://")
    metadata = MetaData()
    table = Table(
        table_name,
        metadata,
        Column("id", Integer, primary_key=True),
        Column(
            "secret",
            EncryptedType(String(1024), NEW_SECRET_KEY),
        ),
        Column(
            "extra",
            LargeBinary,
        ),
    )
    metadata.create_all(engine)

    fake_db = SimpleNamespace(engine=engine, metadata=metadata)

    # ``SecretsMigrator.__init__`` imports ``superset.db`` at call time; patch
    # the module-level ``superset`` package attribute lookup.
    fake_superset = SimpleNamespace(db=fake_db)
    monkeypatch.setitem(__import__("sys").modules, "superset", fake_superset)

    migrator = SecretsMigrator(previous_secret_key=PREVIOUS_SECRET_KEY)
    return migrator, engine, table


def _encrypt_with(key: str, value: str) -> bytes:
    enc = EncryptedType(String(1024), key)
    dialect = create_engine("sqlite://").dialect
    return enc.process_bind_param(value, dialect)


def _raw_view(table_name: str) -> Table:
    """A LargeBinary view of the encrypted column for inserting ciphertext."""
    return Table(
        table_name,
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("secret", LargeBinary),
    )


def _insert_ciphertext(
    engine: Engine, table_name: str, row_id: int, ciphertext: bytes
) -> None:
    """Insert pre-encrypted bytes as-is, simulating production state."""
    raw = _raw_view(table_name)
    with engine.begin() as conn:
        conn.execute(insert(raw).values(id=row_id, secret=ciphertext))


def _read_raw_ciphertext(engine: Engine, table_name: str) -> bytes:
    raw = _raw_view(table_name)
    with engine.begin() as conn:
        value = conn.execute(select(raw.c.secret)).scalar_one()
    if isinstance(value, memoryview):
        value = value.tobytes()
    return value


def test_re_encrypt_row_uses_parameterised_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end key rotation through SQLAlchemy Core (no f-string SQL).

    A table name containing quote characters and a comment terminator is used
    to verify identifiers are quoted by the dialect rather than interpolated
    into a raw SQL string. If the implementation ever regressed to building
    ``text(f"UPDATE {table_name} ...")`` this test would fail with a SQLite
    syntax error from the unescaped ``"`` and ``;`` in the table name.
    """
    migrator, engine, _table = _build_migrator(
        monkeypatch, table_name=SUSPICIOUS_TABLE_NAME
    )

    plaintext = "super_secret_value"
    encrypted_with_old = _encrypt_with(PREVIOUS_SECRET_KEY, plaintext)
    _insert_ciphertext(engine, SUSPICIOUS_TABLE_NAME, 1, encrypted_with_old)

    migrator.run()

    stored = _read_raw_ciphertext(engine, SUSPICIOUS_TABLE_NAME)
    new_enc = EncryptedType(String(1024), NEW_SECRET_KEY)
    old_enc = EncryptedType(String(1024), PREVIOUS_SECRET_KEY)

    # Stored ciphertext must round-trip with the new key …
    assert new_enc.process_result_value(stored, engine.dialect) == plaintext
    # … and must NOT be decryptable with the previous key any more.
    with pytest.raises(ValueError, match="Invalid decryption key"):
        old_enc.process_result_value(stored, engine.dialect)


def test_select_columns_from_table_quotes_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_select_columns_from_table`` must use SQLAlchemy Core to fetch rows.

    Combined with the suspicious table name, this proves identifiers are
    bound through the dialect's identifier preparer rather than concatenated
    into a SQL string.
    """
    migrator, engine, _table = _build_migrator(
        monkeypatch, table_name=SUSPICIOUS_TABLE_NAME
    )

    encrypted_with_old = _encrypt_with(PREVIOUS_SECRET_KEY, "value")
    _insert_ciphertext(engine, SUSPICIOUS_TABLE_NAME, 42, encrypted_with_old)

    with engine.begin() as conn:
        rows = list(
            migrator._select_columns_from_table(  # noqa: SLF001
                conn, ["secret"], SUSPICIOUS_TABLE_NAME
            )
        )

    assert len(rows) == 1
    assert rows[0]._mapping["id"] == 42
    stored = rows[0]._mapping["secret"]
    if isinstance(stored, memoryview):
        stored = stored.tobytes()
    assert stored == encrypted_with_old


def test_run_is_idempotent_on_already_rotated_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running the migrator twice must not corrupt already-rotated rows."""
    migrator, engine, _table = _build_migrator(monkeypatch)
    encrypted_with_old = _encrypt_with(PREVIOUS_SECRET_KEY, "v")
    _insert_ciphertext(engine, "encrypted_secrets", 1, encrypted_with_old)

    migrator.run()
    after_first = _read_raw_ciphertext(engine, "encrypted_secrets")
    migrator.run()
    after_second = _read_raw_ciphertext(engine, "encrypted_secrets")

    new_enc = EncryptedType(String(1024), NEW_SECRET_KEY)
    assert new_enc.process_result_value(after_first, engine.dialect) == "v"
    assert new_enc.process_result_value(after_second, engine.dialect) == "v"


def test_encrypt_module_does_not_use_text_f_strings() -> None:
    """Guard against regressions reintroducing ``text(f"...")`` SQL."""
    import inspect

    source = inspect.getsource(encrypt_module)
    assert 'text(f"' not in source
    assert "text(f'" not in source
