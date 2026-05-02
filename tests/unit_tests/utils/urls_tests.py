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

from unittest import mock

import pytest

from superset.utils.urls import (
    modify_url_query,
    SupersetUrlValidationError,
    validate_url_for_ssrf,
)

EXPLORE_CHART_LINK = "http://localhost:9000/explore/?form_data=%7B%22slice_id%22%3A+76%7D&standalone=true&force=false"

EXPLORE_DASHBOARD_LINK = "http://localhost:9000/superset/dashboard/3/?standalone=3"


def test_convert_chart_link() -> None:
    test_url = modify_url_query(EXPLORE_CHART_LINK, standalone="0")
    assert (
        test_url
        == "http://localhost:9000/explore/?form_data=%7B%22slice_id%22%3A%2076%7D&standalone=0&force=false"
    )


def test_convert_dashboard_link() -> None:
    test_url = modify_url_query(EXPLORE_DASHBOARD_LINK, standalone="0")
    assert test_url == "http://localhost:9000/superset/dashboard/3/?standalone=0"


def test_convert_dashboard_link_with_integer() -> None:
    test_url = modify_url_query(EXPLORE_DASHBOARD_LINK, standalone=0)
    assert test_url == "http://localhost:9000/superset/dashboard/3/?standalone=0"


@pytest.mark.parametrize(
    "url",
    [
        # Cloud-metadata endpoint (the canonical SSRF target).
        "http://169.254.169.254/latest/meta-data/",
        "https://169.254.169.254/",
        # Loopback / RFC1918 / link-local IP literals.
        "http://127.0.0.1/",
        "http://127.1.2.3/admin",
        "http://10.0.0.1/",
        "http://172.16.0.1/",
        "http://192.168.1.1/",
        "http://[::1]/",
        "http://[fe80::1]/",
        # Disallowed schemes that must never be dereferenced server-side.
        "file:///etc/passwd",
        "gopher://example.com/",
        "ftp://example.com/",
        # Malformed URLs.
        "http:///no-host",
        "not-a-url",
    ],
)
def test_validate_url_for_ssrf_blocks_non_public_targets(url: str) -> None:
    with pytest.raises(SupersetUrlValidationError):
        validate_url_for_ssrf(url)


def test_validate_url_for_ssrf_blocks_hostname_resolving_to_metadata_ip() -> None:
    # Simulate DNS rebinding: a benign-looking hostname resolves to the
    # cloud-metadata IP address. The validator must reject the request.
    fake_addr_info = [(2, 1, 6, "", ("169.254.169.254", 0))]
    with mock.patch(
        "superset.utils.urls.socket.getaddrinfo",
        return_value=fake_addr_info,
    ):
        with pytest.raises(SupersetUrlValidationError):
            validate_url_for_ssrf("http://internal.attacker.example/")


def test_validate_url_for_ssrf_allows_public_hostname() -> None:
    fake_addr_info = [(2, 1, 6, "", ("93.184.216.34", 0))]
    with mock.patch(
        "superset.utils.urls.socket.getaddrinfo",
        return_value=fake_addr_info,
    ):
        assert (
            validate_url_for_ssrf("https://example.com/path")
            == "https://example.com/path"
        )


def test_validate_url_for_ssrf_respects_host_allowlist() -> None:
    fake_addr_info = [(2, 1, 6, "", ("93.184.216.34", 0))]
    with mock.patch(
        "superset.utils.urls.socket.getaddrinfo",
        return_value=fake_addr_info,
    ):
        # Hostname is in the allow-list -> accepted.
        validate_url_for_ssrf(
            "https://example.com/",
            allowed_hosts=frozenset({"example.com"}),
        )
        # Hostname is NOT in the allow-list -> rejected, even when public.
        with pytest.raises(SupersetUrlValidationError):
            validate_url_for_ssrf(
                "https://example.com/",
                allowed_hosts=frozenset({"superset.example.org"}),
            )
