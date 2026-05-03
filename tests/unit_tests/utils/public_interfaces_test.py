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

from superset.utils.public_interfaces import (
    compute_class_hash,
    compute_func_hash,
    compute_hash,
)


def test_compute_func_hash_is_deterministic() -> None:
    """The signature fingerprint must be stable across calls."""

    def example(a: int, b: int) -> int:
        return a + b

    assert compute_func_hash(example) == compute_func_hash(example)


def test_compute_func_hash_changes_with_signature() -> None:
    """Adding a parameter must change the fingerprint."""

    def example(a: int, b: int) -> int:
        return a + b

    original = compute_func_hash(example)

    def example(a: int, b: int, c: int) -> int:  # type: ignore[no-redef]
        return a + b + c

    assert compute_func_hash(example) != original


def test_compute_class_hash_is_deterministic() -> None:
    """The class fingerprint must be stable across calls."""

    class Example:
        def __init__(self, a: int) -> None:
            self.a = a

        def value(self) -> int:
            return self.a

    assert compute_class_hash(Example) == compute_class_hash(Example)


def test_compute_hash_dispatch() -> None:
    """compute_hash should dispatch to the function or class variant."""

    def example(a: int) -> int:
        return a

    class Example:
        def __init__(self, a: int) -> None:
            self.a = a

    assert compute_hash(example) == compute_func_hash(example)
    assert compute_hash(Example) == compute_class_hash(Example)


def test_compute_hash_rejects_other_objects() -> None:
    """compute_hash should raise for unsupported object types."""

    with pytest.raises(Exception, match="Invalid object"):
        compute_hash(42)  # type: ignore[arg-type]
