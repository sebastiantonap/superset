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
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from flask import Flask
from flask_babel import lazy_gettext as _
from sqlalchemy import (
    Column,
    LargeBinary,
    MetaData,
    select,
    Table,
    TypeDecorator,
    update,
)
from sqlalchemy.engine import Connection, CursorResult, Dialect, Row
from sqlalchemy_utils import EncryptedType as SqlaEncryptedType


class EncryptedType(SqlaEncryptedType):
    cache_ok = True


ENC_ADAPTER_TAG_ATTR_NAME = "__created_by_enc_field_adapter__"
logger = logging.getLogger(__name__)


class AbstractEncryptedFieldAdapter(ABC):  # pylint: disable=too-few-public-methods
    @abstractmethod
    def create(
        self,
        app_config: Optional[dict[str, Any]],
        *args: list[Any],
        **kwargs: Optional[dict[str, Any]],
    ) -> TypeDecorator:
        pass


class SQLAlchemyUtilsAdapter(  # pylint: disable=too-few-public-methods
    AbstractEncryptedFieldAdapter
):
    def create(
        self,
        app_config: Optional[dict[str, Any]],
        *args: list[Any],
        **kwargs: Optional[dict[str, Any]],
    ) -> TypeDecorator:
        if app_config:
            return EncryptedType(*args, lambda: app_config["SECRET_KEY"], **kwargs)

        raise Exception(  # pylint: disable=broad-exception-raised
            "Missing app_config kwarg"
        )


class EncryptedFieldFactory:
    def __init__(self) -> None:
        self._concrete_type_adapter: Optional[AbstractEncryptedFieldAdapter] = None
        self._config: Optional[dict[str, Any]] = None

    def init_app(self, app: Flask) -> None:
        self._config = app.config
        self._concrete_type_adapter = app.config[
            "SQLALCHEMY_ENCRYPTED_FIELD_TYPE_ADAPTER"
        ]()

    def create(
        self, *args: list[Any], **kwargs: Optional[dict[str, Any]]
    ) -> TypeDecorator:
        if self._concrete_type_adapter:
            adapter = self._concrete_type_adapter.create(self._config, *args, **kwargs)
            setattr(adapter, ENC_ADAPTER_TAG_ATTR_NAME, True)
            return adapter

        raise Exception(  # pylint: disable=broad-exception-raised
            "App not initialized yet. Please call init_app first"
        )

    @staticmethod
    def created_by_enc_field_factory(field: TypeDecorator) -> bool:
        return getattr(field, ENC_ADAPTER_TAG_ATTR_NAME, False)


class SecretsMigrator:
    def __init__(self, previous_secret_key: str) -> None:
        from superset import db  # pylint: disable=import-outside-toplevel

        self._db = db
        self._previous_secret_key = previous_secret_key
        self._dialect: Dialect = db.engine.url.get_dialect()

    def discover_encrypted_fields(self) -> dict[str, dict[str, EncryptedType]]:
        """
        Iterates over SqlAlchemy's metadata, looking for EncryptedType
        columns along the way. Builds up a dict of
        table_name -> dict of col_name: enc type instance
        :return:
        """
        meta_info: dict[str, Any] = {}

        for table_name, table in self._db.metadata.tables.items():
            for col_name, col in table.columns.items():
                if isinstance(col.type, EncryptedType):
                    cols = meta_info.get(table_name, {})
                    cols[col_name] = col.type
                    meta_info[table_name] = cols

        return meta_info

    @staticmethod
    def _read_bytes(col_name: str, value: Any) -> Optional[bytes]:
        if value is None or isinstance(value, bytes):
            return value
        # Note that the Postgres Driver returns memoryview's for BLOB types
        if isinstance(value, memoryview):
            return value.tobytes()
        if isinstance(value, str):
            return bytes(value.encode("utf8"))

        # Just bail if we haven't seen this type before...
        raise ValueError(
            _(
                "DB column %(col_name)s has unknown type: %(value_type)s",
                col_name=col_name,
                value_type=type(value),
            )
        )

    def _build_raw_table(
        self, table_name: str, encrypted_column_names: list[str]
    ) -> Table:
        """
        Build a stand-alone :class:`Table` mirroring ``table_name`` whose
        encrypted columns use the underlying :class:`LargeBinary` storage
        type rather than :class:`EncryptedType`.

        The migrator deliberately handles encryption/decryption itself (using
        both the previous and current ``SECRET_KEY``); going through the
        column's :class:`EncryptedType` processors during I/O would either
        decrypt unexpectedly on read or double-encrypt on write. This raw
        view lets SQLAlchemy treat the values as opaque bytes while still
        building parameterised SQL via the Core expression language, so
        identifiers are quoted by the dialect (no f-string SQL).
        """
        src = self._db.metadata.tables[table_name]
        return Table(
            table_name,
            MetaData(),
            Column("id", src.c["id"].type, primary_key=True),
            *[Column(name, LargeBinary) for name in encrypted_column_names],
        )

    def _select_columns_from_table(
        self, conn: Connection, column_names: list[str], table_name: str
    ) -> CursorResult:
        """
        Select ``id`` and the requested columns from ``table_name`` using the
        SQLAlchemy Core expression language. Resolving the table and columns
        via :class:`MetaData` ensures identifiers are looked up from the
        application's declarative models and quoted by the dialect, so the
        statement cannot be hijacked by attacker-controlled identifiers.
        """
        raw_table = self._build_raw_table(table_name, column_names)
        return conn.execute(select(raw_table))

    def _re_encrypt_row(
        self,
        conn: Connection,
        row: Row,
        table_name: str,
        columns: dict[str, EncryptedType],
    ) -> None:
        """
        Re encrypts all columns in a Row
        :param row: Current row to reencrypt
        :param columns: Meta info from columns
        """
        re_encrypted_columns = {}

        for column_name, encrypted_type in columns.items():
            previous_encrypted_type = EncryptedType(
                type_in=encrypted_type.underlying_type, key=self._previous_secret_key
            )
            try:
                unencrypted_value = previous_encrypted_type.process_result_value(
                    self._read_bytes(column_name, row[column_name]), self._dialect
                )
            except ValueError as ex:
                # Failed to unencrypt
                try:
                    encrypted_type.process_result_value(
                        self._read_bytes(column_name, row[column_name]), self._dialect
                    )
                    logger.info(
                        "Current secret is able to decrypt value on column [%s.%s],"
                        " nothing to do",
                        table_name,
                        column_name,
                    )
                    return
                except Exception:
                    raise Exception from ex  # pylint: disable=broad-exception-raised

            re_encrypted_columns[column_name] = encrypted_type.process_bind_param(
                unencrypted_value,
                self._dialect,
            )

        logger.info("Processing table: %s", table_name)
        raw_table = self._build_raw_table(table_name, list(re_encrypted_columns.keys()))
        conn.execute(
            update(raw_table)
            .where(raw_table.c["id"] == row["id"])
            .values(**re_encrypted_columns)
        )

    def run(self) -> None:
        encrypted_meta_info = self.discover_encrypted_fields()

        with self._db.engine.begin() as conn:
            logger.info("Collecting info for re encryption")
            for table_name, columns in encrypted_meta_info.items():
                column_names = list(columns.keys())
                rows = self._select_columns_from_table(conn, column_names, table_name)

                for row in rows:
                    self._re_encrypt_row(conn, row, table_name, columns)
        logger.info("All tables processed")
