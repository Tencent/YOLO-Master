"""Second server instance with API keys, a rate limit and the JSON access log (tests/run_server_tests.sh
phase 2). Skipped entirely unless YM_SERVER_URL2 is set."""
import json, os, sys, time, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "clients" / "python"))
from yolomaster_client import Client, ApiError  # noqa: E402

URL2 = os.environ.get("YM_SERVER_URL2", "")
KEY = os.environ.get("YM_API_KEY", "")
LOG2 = os.environ.get("YM_SERVER_LOG2", "")
MODEL = os.environ.get("YM_TEST_MODEL", "esmoe")
IMAGES = Path(os.environ.get("YM_TEST_IMAGES", ROOT / "visdrone50" / "images" / "val"))
IMGS = sorted(IMAGES.glob("*.jpg"))[:2]


def _skip(msg):
    print("  SKIP ", msg)
    return None


def _get(path, headers=None):
    req = urllib.request.Request(URL2 + path, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_auth_required():
    if not URL2:
        return _skip("YM_SERVER_URL2 not set")
    st, h, body = _get("/v1/models")
    assert st == 401 and "WWW-Authenticate" in h and json.loads(body)["code"] == 401
    st, _, _ = _get("/metrics")
    assert st == 401
    st, _, _ = _get("/healthz")
    assert st == 200                                   # exempt
    st, _, _ = _get("/openapi.json")
    assert st == 200                                   # auth off on the document itself
    st, _, _ = _get("/v1/models", {"X-API-Key": "wrong-key-wrong-key-wrong"})
    assert st == 401


def test_auth_header_and_bearer():
    if not URL2:
        return _skip("YM_SERVER_URL2 not set")
    st, _, _ = _get("/v1/models", {"X-API-Key": KEY})
    assert st == 200
    st, _, _ = _get("/v1/models", {"Authorization": "Bearer " + KEY})
    assert st == 200
    c = Client(URL2, api_key=KEY)
    r = c.infer(IMGS[0], model=MODEL)
    assert r["count"] >= 0 and "traceparent" in {k.lower() for k in c.last_headers}


def test_ws_upgrade_requires_key():
    if not URL2:
        return _skip("YM_SERVER_URL2 not set")
    try:
        import websocket  # noqa
    except ImportError:
        return _skip("websocket-client missing")
    ws_url = URL2.replace("http://", "ws://") + f"/v1/stream?model={MODEL}"
    try:
        websocket.create_connection(ws_url, timeout=10).close()
        assert False, "upgrade without a key should be rejected"
    except websocket.WebSocketBadStatusException as e:
        assert e.status_code == 401
    ws = websocket.create_connection(ws_url, timeout=10, header=[f"X-API-Key: {KEY}"])
    assert "hello" in json.loads(ws.recv())
    ws.close()


def test_rate_limit_burst_then_429():
    if not URL2:
        return _skip("YM_SERVER_URL2 not set")
    time.sleep(2.5)                                    # refill the bucket from earlier tests
    codes = []
    for _ in range(20):
        st, h, _ = _get("/v1/models", {"X-API-Key": KEY})
        codes.append((st, h))
    n429 = sum(1 for st, _ in codes if st == 429)
    assert codes[0][0] == 200 and n429 >= 3, [c[0] for c in codes]
    first429 = next(h for st, h in codes if st == 429)
    assert int(first429.get("Retry-After", "0")) >= 1 and "X-RateLimit-Remaining" in first429
    assert "X-RateLimit-Limit" in codes[0][1]
    time.sleep(1.5)
    st, _, _ = _get("/v1/models", {"X-API-Key": KEY})
    assert st == 200                                   # recovered
    st, _, body = _get("/metrics", {"X-API-Key": KEY})
    assert "yolomaster_rate_limited_total" in body.decode() and "yolomaster_auth_failed_total" in body.decode()


def test_json_access_log():
    if not URL2 or not LOG2:
        return _skip("YM_SERVER_URL2 / YM_SERVER_LOG2 not set")
    time.sleep(1.2)
    rid = "hardened-test-" + str(int(time.time()))
    st, h, _ = _get("/v1/models", {"X-API-Key": KEY, "X-Request-Id": rid})
    assert st == 200 and h.get("X-Request-Id") == rid
    c = Client(URL2, api_key=KEY); c.request_id = rid + "-infer"
    c.infer(IMGS[0], model=MODEL)
    time.sleep(0.5)
    lines = [json.loads(l) for l in open(LOG2, errors="replace") if l.startswith("{")]
    mine = [l for l in lines if l.get("rid") == rid]
    assert mine and mine[0]["status"] == 200 and mine[0]["path"] == "/v1/models" and len(mine[0]["trace_id"]) == 32
    inf = [l for l in lines if l.get("rid") == rid + "-infer"]
    assert inf and inf[0]["model"] == MODEL and set(inf[0]["stages"]) >= {"queue", "pre", "infer", "post", "total"}


if __name__ == "__main__":
    fails = 0
    for name, fn in [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]:
        try:
            fn(); print("  PASS ", name)
        except Exception as e:  # noqa
            fails += 1; print("  FAIL ", name, repr(e)[:300])
    print(f"RESULT: {fails} failed")
    sys.exit(1 if fails else 0)
