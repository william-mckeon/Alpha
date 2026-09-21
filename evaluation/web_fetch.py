"""Bounded BFCL page fetching with DNS-pinned public HTTPS connections."""
from __future__ import annotations

from importlib import import_module
import ipaddress
import socket
import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

from evaluation.worker_adapters import isolated_worker_environment


class FetchFailure(ValueError):
    """Sanitized fetch outcome; never includes response bodies, URLs or credentials."""
    def __init__(self, code, http_status=None):
        self.details = {"code": code, "http_status": http_status,
                        "recoverable": code == "http_status" and type(http_status) is int
                        and 300 <= http_status < 500 and http_status not in {408, 429}}
        super().__init__(json.dumps(self.details, sort_keys=True))


def valid_fetch_error(value):
    if not isinstance(value, dict) or set(value) != {"code", "http_status", "recoverable"}:
        return False
    if value["code"] not in {"http_status", "timeout", "dns", "tls", "transport", "security", "size_limit", "internal"}:
        return False
    status = value["http_status"]
    if (value["code"] == "http_status" and (type(status) is not int or not 100 <= status <= 599)) or (value["code"] != "http_status" and status is not None):
        return False
    return type(value["recoverable"]) is bool and value == FetchFailure(value["code"], status).details


def safe_fetch_failure(exc):
    from urllib3 import exceptions
    if isinstance(exc, FetchFailure):
        return exc
    if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired, exceptions.TimeoutError)):
        return FetchFailure("timeout")
    if isinstance(exc, socket.gaierror):
        return FetchFailure("dns")
    if isinstance(exc, exceptions.SSLError):
        return FetchFailure("tls")
    if isinstance(exc, (ConnectionError, exceptions.HTTPError)):
        return FetchFailure("transport")
    return FetchFailure("internal")


def bounded_fetch(url: str, mode: str = "raw") -> dict[str, str]:
    """Kill the fetch subprocess on a wall deadline, including DNS and slow bodies."""
    environment = isolated_worker_environment(os.environ)
    environment.pop("ARCUS_WORKER_TRIAL_ID", None)
    environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    try:
        completed = subprocess.run([sys.executable, "-m", "evaluation.web_fetch"], input=json.dumps({"url": url, "mode": mode}), capture_output=True, text=True, encoding="utf-8", cwd=Path(__file__).resolve().parents[1], env=environment, timeout=45, check=False)
    except subprocess.TimeoutExpired:
        raise FetchFailure("timeout") from None
    if completed.returncode:
        try:
            envelope = json.loads(completed.stdout)
            detail = envelope["error"]
            if completed.returncode != 2 or set(envelope) != {"error"} or not valid_fetch_error(detail):
                raise ValueError()
        except (ValueError, KeyError, TypeError):
            raise FetchFailure("internal") from None
        raise FetchFailure(detail["code"], detail["http_status"])
    result = json.loads(completed.stdout)
    if set(result) != {"content"} or not isinstance(result["content"], str):
        raise ValueError("invalid fetch subprocess response")
    return result


def public_target(url: str) -> tuple[str, str, str]:
    if not isinstance(url, str) or len(url) > 8192:
        raise ValueError("invalid fetch URL")
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.port not in (None, 443):
        raise FetchFailure("security")
    host = parsed.hostname.encode("idna").decode("ascii")
    addresses = sorted({row[4][0] for row in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)})
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise FetchFailure("security")
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    return host, addresses[0], path


def fetch_url_content(url: str, mode: str = "raw", *, max_bytes: int = 1_048_576) -> dict[str, str]:
    """No redirects, retries, ambient proxies or second DNS lookup; TLS checks hostname."""
    if mode not in {"raw", "markdown", "text", "truncate"} or type(max_bytes) is not int or not 0 < max_bytes <= 1_048_576:
        raise ValueError("unsupported fetch mode or size ceiling")
    host, address, path = public_target(url)
    urllib3 = import_module("urllib3")
    pool = urllib3.HTTPSConnectionPool(address, port=443, server_hostname=host, assert_hostname=host, cert_reqs="CERT_REQUIRED", timeout=urllib3.Timeout(connect=5, read=10), retries=False)
    response = None
    try:
        response = pool.request("GET", path, headers={"Host": host, "User-Agent": "Arcus-BFCL-adapter/1"}, redirect=False, retries=False, preload_content=False)
        if response.status != 200:
            raise FetchFailure("http_status", response.status)
        payload = response.read(max_bytes + 1, decode_content=True)
        if len(payload) > max_bytes:
            raise FetchFailure("size_limit")
        content = payload.decode("utf-8", errors="replace")
        if mode == "markdown":
            content = import_module("html2text").html2text(content)
        elif mode in {"text", "truncate"}:
            soup = import_module("bs4").BeautifulSoup(content, "html.parser")
            for element in soup(["script", "style"]):
                element.decompose()
            content = soup.get_text(separator=" ", strip=True)
        return {"content": content}
    finally:
        if response is not None:
            response.close()
        pool.close()


if __name__ == "__main__":
    try:
        arguments = json.loads(sys.stdin.read(16384))
        if set(arguments) != {"url", "mode"}:
            raise ValueError("invalid fetch request")
        print(json.dumps(fetch_url_content(**arguments), ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"error": safe_fetch_failure(exc).details}))
        sys.exit(2)
