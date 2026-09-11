"""Validate redirect destinations before network access, without contacting remote hosts."""

import hashlib
import io
from typing import ClassVar
from unittest.mock import Mock
from urllib.request import Request

import pytest

from tools import model_zoo_automation as zoo


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/a.onnx",
        "https://127.0.0.1/a.onnx",
        "https://github.com:8443/a.onnx",
        "https://user:pass@github.com/a.onnx",
    ],
)
def test_reject_redirect_before_following(url):
    """No Request for a disallowed Location may be returned to urllib."""
    with pytest.raises(ValueError):
        zoo.CheckedRedirectHandler().redirect_request(Request("https://github.com/a.onnx"), None, 302, "Found", {}, url)


def test_allowed_redirect_stays_supported():
    """Release assets on allowed HTTPS hosts must continue working."""
    redirected = zoo.CheckedRedirectHandler().redirect_request(
        Request("https://github.com/a.onnx"), None, 302, "Found", {}, "https://objects.githubusercontent.com/a.onnx"
    )
    assert redirected.full_url == "https://objects.githubusercontent.com/a.onnx"


@pytest.mark.parametrize("bad_digest", [False, True])
def test_download_uses_checked_opener_and_keeps_integrity_checks(tmp_path, monkeypatch, bad_digest):
    """Changing redirect handling must not drop length and SHA256 checks."""

    class Response(io.BytesIO):
        headers: ClassVar[dict] = {"Content-Length": "4"}

        def geturl(self):
            return "https://github.com/a.onnx"

    opener = Mock()
    opener.open.return_value = Response(b"test")
    factory = Mock(return_value=opener)
    monkeypatch.setattr(zoo.urllib.request, "build_opener", factory)
    data = {
        "weights": {
            "url": "https://github.com/a.onnx",
            "size_bytes": 4,
            "sha256": "0" * 64 if bad_digest else hashlib.sha256(b"test").hexdigest(),
        }
    }
    if bad_digest:
        with pytest.raises(ValueError, match="SHA-256"):
            zoo.download_weights(data, tmp_path / "model.onnx")
    else:
        assert zoo.download_weights(data, tmp_path / "model.onnx").read_bytes() == b"test"
    assert isinstance(factory.call_args.args[0], zoo.CheckedRedirectHandler)
