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
import ipaddress
import socket
import urllib
from typing import Any, Optional, Union
from urllib.parse import urlparse

from flask import current_app as app, has_request_context, url_for


class SupersetUrlValidationError(ValueError):
    """Raised when a URL fails SSRF validation."""


# Schemes that the application is willing to dereference server-side.
_DEFAULT_ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

# IPv4 + IPv6 address types implement the ``is_*`` properties used by
# :func:`_is_disallowed_ip` (the abstract :class:`ipaddress._BaseAddress` base
# class does not — using this union lets mypy recognise them).
_IpAddress = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]


def get_url_host(user_friendly: bool = False) -> str:
    if user_friendly:
        return app.config["WEBDRIVER_BASEURL_USER_FRIENDLY"]
    return app.config["WEBDRIVER_BASEURL"]


def headless_url(path: str, user_friendly: bool = False) -> str:
    return urllib.parse.urljoin(get_url_host(user_friendly=user_friendly), path)


def get_url_path(view: str, user_friendly: bool = False, **kwargs: Any) -> str:
    in_request_context = has_request_context()

    # When already in a request context, Flask's url_for respects SCRIPT_NAME from
    # the WSGI environment, so the prefix is already included. Only add APPLICATION_ROOT
    # prefix when creating a new request context.
    if in_request_context:
        url = url_for(view, **kwargs)
    else:
        with app.test_request_context():
            url = url_for(view, **kwargs)
            app_root = app.config.get("APPLICATION_ROOT", "/")
            if app_root != "/" and not url.startswith(app_root):
                url = app_root.rstrip("/") + url

    return headless_url(url, user_friendly=user_friendly)


def modify_url_query(url: str, **kwargs: Any) -> str:
    """
    Replace or add parameters to a URL.
    """
    parts = list(urllib.parse.urlsplit(url))
    params = urllib.parse.parse_qs(parts[3])
    for k, v in kwargs.items():
        if not isinstance(v, list):
            v = [v]
        params[k] = v

    parts[3] = "&".join(
        f"{k}={urllib.parse.quote(str(v[0]))}" for k, v in params.items()
    )
    return urllib.parse.urlunsplit(parts)


def is_secure_url(url: str) -> bool:
    """
    Validates if a URL is secure (uses HTTPS).

    :param url: The URL to validate.
    :return: True if the URL uses HTTPS (secure), False if it uses HTTP (non-secure).
    """
    parsed_url = urlparse(url)
    return parsed_url.scheme == "https"


def _is_disallowed_ip(ip: _IpAddress) -> bool:
    """
    Return ``True`` if ``ip`` falls in any range that should never be the target
    of an outbound request originated by the server (loopback, link-local,
    private networks, multicast, reserved, unspecified, etc.).

    The cloud-metadata IPs (``169.254.169.254`` for EC2/GCP/Azure IMDS and the
    IPv6 ``fd00:ec2::254`` address) are link-local / private and therefore
    already covered by the standard library properties below, but they are
    listed here explicitly to make the intent obvious to reviewers.
    """
    return (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def validate_url_for_ssrf(  # noqa: C901
    url: str,
    *,
    allowed_schemes: Optional[frozenset[str]] = None,
    allowed_hosts: Optional[frozenset[str]] = None,
) -> str:
    """
    Validate ``url`` before it is dereferenced by a server-side HTTP client.

    The check is intentionally conservative and is intended to mitigate
    Server-Side Request Forgery (SSRF, CWE-918) attacks against outbound
    requests where the URL (or any component of it) may be influenced by
    configuration or untrusted input. The caller is still responsible for
    proper authentication and transport-layer security on the request itself.

    The validation performs the following checks:

    * The URL must be parseable and contain a scheme and a hostname.
    * The scheme must be in ``allowed_schemes`` (defaults to ``http``/``https``)
      — disallowing schemes such as ``file://``, ``gopher://`` and ``ftp://``.
    * If ``allowed_hosts`` is provided, the hostname must match (case-insensitive)
      one of the entries in the allow-list.
    * Every IP address that the hostname resolves to must be a public address.
      Loopback (``127.0.0.0/8``), link-local (``169.254.0.0/16`` — including the
      cloud-metadata endpoint ``169.254.169.254``), RFC1918 private ranges
      (``10.0.0.0/8``, ``172.16.0.0/12``, ``192.168.0.0/16``), multicast,
      reserved and unspecified addresses are all rejected.

    :param url: The fully-qualified URL to validate.
    :param allowed_schemes: Optional override for the scheme allow-list.
    :param allowed_hosts: Optional case-insensitive hostname allow-list.
    :returns: ``url`` unchanged when validation succeeds.
    :raises SupersetUrlValidationError: If the URL fails any of the checks.
    """
    schemes = allowed_schemes or _DEFAULT_ALLOWED_SCHEMES

    try:
        parsed = urlparse(url)
    except ValueError as ex:
        raise SupersetUrlValidationError(f"Invalid URL: {url}") from ex

    if parsed.scheme.lower() not in schemes:
        raise SupersetUrlValidationError(
            f"URL scheme {parsed.scheme!r} is not allowed for outbound requests"
        )

    hostname = parsed.hostname
    if not hostname:
        raise SupersetUrlValidationError(
            f"URL must include a hostname for outbound requests: {url}"
        )

    if allowed_hosts is not None:
        normalized_allowed = {host.lower() for host in allowed_hosts}
        if hostname.lower() not in normalized_allowed:
            raise SupersetUrlValidationError(
                f"Hostname {hostname!r} is not in the configured allow-list"
            )

    # If the hostname is a literal IP address, validate it directly.
    try:
        literal_ip = ipaddress.ip_address(hostname)
    except ValueError:
        literal_ip = None

    if literal_ip is not None:
        if _is_disallowed_ip(literal_ip):
            raise SupersetUrlValidationError(
                f"Refusing to make a request to non-public IP address {hostname}"
            )
        return url

    # Otherwise resolve the hostname and validate every returned address. This
    # protects against DNS rebinding-style abuse where a hostname resolves to a
    # private/internal IP at request time.
    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as ex:
        raise SupersetUrlValidationError(
            f"Unable to resolve hostname {hostname!r}"
        ) from ex

    for info in addr_infos:
        sockaddr = info[4]
        try:
            resolved = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            continue
        if _is_disallowed_ip(resolved):
            raise SupersetUrlValidationError(
                f"Hostname {hostname!r} resolved to non-public IP {resolved.compressed}"
            )

    return url
