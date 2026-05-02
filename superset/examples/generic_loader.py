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
"""Generic Parquet example data loader."""

import logging
import re
from functools import partial
from typing import Any, Callable, Optional

import numpy as np
from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError
from sqlalchemy.schema import CreateSchema

from superset import db
from superset.connectors.sqla.models import SqlaTable
from superset.examples.helpers import read_example_data
from superset.models.core import Database
from superset.sql.parse import Table
from superset.utils import json
from superset.utils.database import get_example_database

logger = logging.getLogger(__name__)

# Strict allow-list for schema identifiers used in CREATE SCHEMA DDL.
# Limits names to ASCII letters, digits, and underscores starting with a
# letter or underscore. This is intentionally narrower than what most
# databases technically permit so that schema names supplied via dataset
# configuration files cannot be used to construct injected SQL.
_VALID_SCHEMA_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ensure_schema_exists(engine: Engine, schema: str) -> None:
    """Create ``schema`` on ``engine`` if it does not already exist.

    The schema name is validated against a strict allow-list and the DDL
    statement is constructed via SQLAlchemy's :class:`CreateSchema` element
    so that the identifier is quoted by the dialect's ``IdentifierPreparer``
    rather than interpolated into a raw SQL string. Together these guard
    against SQL injection (CWE-89) when the schema name originates from a
    dataset configuration file or other potentially untrusted source.

    SQLAlchemy 1.4's :class:`CreateSchema` does not yet support an
    ``if_not_exists`` flag (added in SQLAlchemy 2.0), so the idempotent
    semantics of the original raw ``CREATE SCHEMA IF NOT EXISTS`` statement
    are preserved by catching the duplicate-schema error that the database
    raises when another process wins a TOCTOU race against the inspector
    pre-check.
    """
    if not isinstance(schema, str) or not _VALID_SCHEMA_NAME.match(schema):
        raise ValueError(
            f"Invalid schema name {schema!r}: schema names must match "
            "[A-Za-z_][A-Za-z0-9_]* to be safely used in DDL."
        )

    if schema in inspect(engine).get_schema_names():
        return

    try:
        with engine.begin() as conn:
            conn.execute(CreateSchema(schema))
    except DBAPIError:
        # Another process created the schema between the inspector check
        # and the CREATE SCHEMA execution. Re-verify via a fresh inspector;
        # if the schema is now present the DDL effectively succeeded and we
        # can swallow the duplicate-schema error. Otherwise re-raise so the
        # caller still sees genuine database failures.
        if schema not in inspect(engine).get_schema_names():
            raise


def serialize_numpy_arrays(obj: Any) -> Any:  # noqa: C901
    """Convert numpy arrays to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        # Handle numpy scalar types
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return [serialize_numpy_arrays(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: serialize_numpy_arrays(val) for key, val in obj.items()}
    return obj


def load_parquet_table(  # noqa: C901
    parquet_file: str,
    table_name: str,
    database: Optional[Database] = None,
    only_metadata: bool = False,
    force: bool = False,
    sample_rows: Optional[int] = None,
    data_file: Optional[Any] = None,
    schema: Optional[str] = None,
    uuid: Optional[str] = None,
) -> SqlaTable:
    """Load a Parquet file into the example database.

    Args:
        parquet_file: Name of the Parquet file (e.g., "birth_names")
        table_name: Name for the table in the target database
        database: Target database (defaults to example database)
        only_metadata: If True, only create metadata without loading data
        force: If True, replace existing table
        sample_rows: If specified, only load this many rows
        data_file: Optional specific file path (Path object) to load from
        schema: Schema to load into (defaults to database default schema)

    Returns:
        The created SqlaTable object
    """
    if database is None:
        database = get_example_database()

    # Determine schema - use provided or fall back to database default
    with database.get_sqla_engine() as engine:
        if schema is None:
            schema = inspect(engine).default_schema_name
        else:
            _ensure_schema_exists(engine, schema)

    table_exists = database.has_table(Table(table_name, schema=schema))
    if table_exists and not force:
        logger.info("Table %s already exists, skipping data load", table_name)
        tbl = (
            db.session.query(SqlaTable)
            .filter_by(table_name=table_name, database_id=database.id)
            .first()
        )
        if tbl:
            return tbl

    # Load data if not metadata only
    if not only_metadata:
        logger.info("Loading data for %s from %s.parquet", table_name, parquet_file)

        # Read from Parquet - use specific file path if provided
        if data_file is not None:
            pdf = read_example_data(f"file://{data_file}")
        else:
            pdf = read_example_data(f"examples://{parquet_file}")

        # Sample if requested (handle sample_rows=0 correctly)
        if sample_rows is not None:
            pdf = pdf.head(sample_rows)

        # Check for columns with complex types (numpy arrays, nested structures)
        for col in pdf.columns:
            # Check if any value in the column is a numpy array or nested structure
            if pdf[col].dtype == object:
                try:
                    # Check if the first non-null value is complex
                    sample_val = (
                        pdf[col].dropna().iloc[0]
                        if not pdf[col].dropna().empty
                        else None
                    )
                    if sample_val is not None and isinstance(
                        sample_val, (np.ndarray, list, dict)
                    ):
                        logger.info("Converting complex column %s to JSON string", col)

                        # Convert to JSON string for database storage
                        def safe_serialize(x: Any, column_name: str) -> Optional[str]:
                            if x is None:
                                return None
                            try:
                                return json.dumps(serialize_numpy_arrays(x))
                            except (TypeError, ValueError) as e:
                                logger.warning(
                                    "Failed to serialize value in column %s: %s",
                                    column_name,
                                    e,
                                )
                                # Convert to string representation as fallback
                                return str(x)

                        # Avoid loop variable binding issues with partial
                        serialize_col = partial(safe_serialize, column_name=col)
                        pdf[col] = pdf[col].apply(serialize_col)
                except Exception as e:
                    logger.warning("Could not process column %s: %s", col, e)

        # Write to target database
        with database.get_sqla_engine() as engine:
            pdf.to_sql(
                table_name,
                engine,
                schema=schema,
                if_exists="replace",
                chunksize=500,
                method="multi",
                index=False,
            )

        logger.info("Loaded %d rows into %s", len(pdf), table_name)

    # Create or update SqlaTable metadata
    tbl = (
        db.session.query(SqlaTable)
        .filter_by(table_name=table_name, database_id=database.id)
        .first()
    )

    if not tbl:
        tbl = SqlaTable(table_name=table_name, database_id=database.id)
        # Set the database reference
        tbl.database = database

    # Set UUID from YAML config so the YAML import path can find this dataset
    if uuid and not tbl.uuid:
        tbl.uuid = uuid

    if not only_metadata:
        # Ensure database reference is set before fetching metadata
        if not tbl.database:
            tbl.database = database
        tbl.fetch_metadata()

    db.session.merge(tbl)
    # pylint: disable=consider-using-transaction
    db.session.commit()

    return tbl


def create_generic_loader(
    parquet_file: str,
    table_name: Optional[str] = None,
    description: Optional[str] = None,
    sample_rows: Optional[int] = None,
    data_file: Optional[Any] = None,
    schema: Optional[str] = None,
    uuid: Optional[str] = None,
) -> Callable[[Database, SqlaTable], None]:
    """Create a loader function for a specific Parquet file.

    This factory function creates loaders that match the existing pattern
    used by Superset examples.

    Args:
        parquet_file: Name of the Parquet file (without .parquet extension)
        table_name: Table name (defaults to parquet_file)
        description: Description for the dataset
        sample_rows: Default number of rows to sample
        data_file: Optional specific file path (Path object) for data/ folder pattern
        schema: Schema to load into (defaults to database default schema)

    Returns:
        A loader function with the standard signature
    """
    if table_name is None:
        table_name = parquet_file

    def loader(
        only_metadata: bool = False,
        force: bool = False,
        sample: bool = False,
    ) -> None:
        """Load the dataset."""
        rows = sample_rows if sample and sample_rows is not None else None

        tbl = load_parquet_table(
            parquet_file=parquet_file,
            table_name=table_name,
            only_metadata=only_metadata,
            force=force,
            sample_rows=rows,
            data_file=data_file,
            schema=schema,
            uuid=uuid,
        )

        if description and tbl:
            tbl.description = description
            db.session.merge(tbl)
            # pylint: disable=consider-using-transaction
            db.session.commit()

    # Set function name and docstring
    loader.__name__ = f"load_{parquet_file}"
    loader.__doc__ = description or f"Load {parquet_file} dataset"

    return loader
