"""API server integration tests (stdlib urllib; optional cv2/websocket-client for the video/WS cases).

Run via tests/run_server_tests.sh (boots the server on a free port with two CPU models) or:
    YM_SERVER_URL=http://localhost:8080 YM_TEST_MODEL=esmoe YM_TEST_MODEL2=esmoe-ncnn \
    YM_TEST_IMAGES=visdrone50/images/val YM_CLI=cpp/build/yolomaster_edge YM_CLI_MODEL=models/esmoe_n_visdrone_sim.onnx \
    python -m pytest tests/server -q      (or: python tests/server/test_api.py)
"""
import json, os, subprocess, sys, tempfile, time
from pathlib import Path
import urllib.request, urllib.error

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "clients" / "python"))
from yolomaster_client import Client, ApiError  # noqa: E402

URL = os.environ.get("YM_SERVER_URL", "http://localhost:8080")
MODEL = os.environ.get("YM_TEST_MODEL", "esmoe")
MODEL2 = os.environ.get("YM_TEST_MODEL2", "")
IMAGES = Path(os.environ.get("YM_TEST_IMAGES", ROOT / "visdrone50" / "images" / "val"))
CLI = os.environ.get("YM_CLI", "")
CLI_MODEL = os.environ.get("YM_CLI_MODEL", "")
IMGS = sorted(IMAGES.glob("*.jpg"))[:8]
c = Client(URL, timeout=60)


def _skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        print("  SKIP", msg)


def test_health_ready():
    assert c.health()["status"] == "ok"
    assert c.wait_ready(120), "server never became ready"


def test_models_listed():
    ids = {m["id"] for m in c.models()}
    assert MODEL in ids
    m = next(m for m in c.models() if m["id"] == MODEL)
    assert m["loaded"] and m["ready"] and m["nc"] > 0 and m["imgsz"] > 0


def test_infer_json_shape():
    r = c.infer(IMGS[0], model=MODEL)
    assert r["model"] == MODEL and r["count"] == len(r["detections"]) > 0
    d = r["detections"][0]
    assert set(d) >= {"class_id", "conf", "box", "name"} and len(d["box"]) == 4
    assert d["box"][0] <= d["box"][2] and d["box"][1] <= d["box"][3]
    for k in ("queue", "decode", "pre", "infer", "post", "total"):
        assert r["timings_ms"][k] >= 0
    assert r["image"]["width"] > 0 and "X-Request-Id" in c.last_headers


def test_txt_matches_json():
    j = c.infer(IMGS[1], model=MODEL)
    t = c.infer(IMGS[1], model=MODEL, ret="txt").strip().splitlines()
    assert len(t) == j["count"]
    cls, conf, x1, y1, x2, y2 = t[0].split()
    assert int(cls) == j["detections"][0]["class_id"]
    assert abs(float(x1) - j["detections"][0]["box"][0]) < 0.01


def test_txt_parity_with_cli():
    if not (CLI and CLI_MODEL and Path(CLI).exists()):
        return _skip("YM_CLI/YM_CLI_MODEL not set")
    with tempfile.TemporaryDirectory() as td:
        # YM_CLI_ARGS: backend / precision flags so the CLI runs the same engine as the server model
        # (e.g. "-b trt --precision fp16" on a GPU pod); the default compares the CPU ONNX paths.
        extra = os.environ.get("YM_CLI_ARGS", "").split()
        subprocess.run([CLI, "-m", CLI_MODEL, "-s", str(IMGS[2]), "--no-save", "--quiet", "--save-txt", td] + extra, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cli = sorted(Path(td).glob("*.txt"))[0].read_text().split()
    api = c.infer(IMGS[2], model=MODEL, ret="txt").split()
    assert len(api) == len(cli)
    for a, b in zip(api, cli):
        assert abs(float(a) - float(b)) < 1e-3, (a, b)


def test_param_overrides():
    lo = c.infer(IMGS[0], model=MODEL, conf=0.05)["count"]
    hi = c.infer(IMGS[0], model=MODEL, conf=0.8)["count"]
    assert lo >= hi
    capped = c.infer(IMGS[0], model=MODEL, conf=0.05, max_det=3)
    assert capped["count"] <= 3 and capped["params"]["max_det"] == 3


def test_coco_format():
    r = c.infer(IMGS[0], model=MODEL, ret="coco", image_id=42, coco91=0)
    assert isinstance(r, list) and r[0]["image_id"] == 42 and len(r[0]["bbox"]) == 4 and r[0]["bbox"][2] > 0


def test_annotated_jpeg():
    b = c.infer(IMGS[0], model=MODEL, ret="annotated", masks="overlay")
    assert isinstance(b, bytes) and b[:2] == b"\xff\xd8" and int(c.last_headers.get("X-Detections", "0")) > 0


def test_multipart_and_batch():
    rs = c.infer_many(IMGS[:3], model=MODEL)
    assert len(rs) == 3 and all(r["count"] >= 0 for r in rs) and rs[0]["file"] == IMGS[0].name


def test_second_model_if_any():
    if not MODEL2:
        return _skip("no second model")
    r = c.infer(IMGS[0], model=MODEL2)
    assert r["model"] == MODEL2 and r["count"] > 0


def test_errors():
    for kwargs, code in (({"model": "does-not-exist"}, 404),):
        try:
            c.infer(IMGS[0], **kwargs); assert False
        except ApiError as e:
            assert e.status == code and "error" in e.body
    try:
        c.infer(b"definitely not an image", model=MODEL); assert False
    except ApiError as e:
        assert e.status == 400
    try:
        c.infer(IMGS[0], model=MODEL, ret="xml"); assert False
    except ApiError as e:
        assert e.status == 400
    try:
        c._req("GET", "/no/such/route"); assert False
    except ApiError as e:
        assert e.status == 404
    try:  # body cap: 64 MB of zeros -> 413 (server default max_body_mb=32, tests use 4)
        c._req("POST", "/v1/infer", b"\0" * (8 << 20), query={"model": MODEL}); assert False
    except ApiError as e:
        assert e.status == 413


def test_queue_full_503_or_ok():
    """Flood the queue: every answer must be 200 or 503 (never a hang or a 500)."""
    import concurrent.futures as cf
    data = IMGS[0].read_bytes()
    def one(_):
        try:
            c.infer(data, model=MODEL); return 200
        except ApiError as e:
            return e.status
    with cf.ThreadPoolExecutor(32) as ex:
        codes = list(ex.map(one, range(96)))
    assert set(codes) <= {200, 503}, codes
    assert codes.count(200) >= 1


def test_stats_and_metrics():
    s = c.stats()
    assert s["models"][MODEL]["requests_ok"] > 0 and "p50" in s["models"][MODEL]["latency_ms"]["5m"]
    m = c.metrics()
    assert "yolomaster_requests_total{model=\"%s\",code=\"200\"}" % MODEL in m
    assert "yolomaster_stage_seconds_bucket" in m and "yolomaster_queue_depth" in m


def _make_video(path, n=6):
    import cv2
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (640, 480))
    for p in IMGS[:n]:
        vw.write(cv2.resize(cv2.imread(str(p)), (640, 480)))
    vw.release()


def test_video_ndjson():
    try:
        import cv2  # noqa
    except ImportError:
        return _skip("opencv-python missing")
    with tempfile.TemporaryDirectory() as td:
        vp = Path(td) / "t.mp4"; _make_video(vp)
        try:
            frames = list(c.video(vp, model=MODEL, every=2))
        except ApiError as e:
            if e.status == 501:
                return _skip("server built without videoio")
            raise
    assert frames[-1]["done"] and frames[-1]["frames"] == 3 and frames[0]["frame"] == 0 and frames[1]["frame"] == 2
    assert all("count" in f for f in frames[:-1])


def test_ws_stream():
    try:
        import cv2, websocket  # noqa
    except ImportError:
        return _skip("opencv-python/websocket-client missing")
    with tempfile.TemporaryDirectory() as td:
        vp = Path(td) / "t.mp4"; _make_video(vp)
        out = list(c.stream_video(vp, model=MODEL))
    assert len(out) == 6 and all("count" in o and "seq" in o for o in out)
    assert [o["seq"] for o in out] == list(range(1, 7))


def test_slicing_param():
    r = c.infer(IMGS[0], model=MODEL, slicing="dense")
    assert "slicing" in r and r["slicing"]["tiles_total"] > 0 and r["slicing"]["tiles_run"] >= 1
    r2 = c.infer(IMGS[0], model=MODEL, slicing="off")
    assert "slicing" not in r2


def test_bench_endpoint():
    j = c.bench(model=MODEL, warmup=1, iters=4)
    assert j["schema_version"] == "yolomaster-bench/v1" and j["tool"] == "server"
    assert j["cold"]["infer_ms"]["n"] == 4 and j["cold"]["probe_mode"] in ("infer_only", "full")
    assert j["model"]["id"] == MODEL and j["environment"]["version"] and "request_id" in j
    for k in ("mean", "median", "p90", "p95", "p99", "min", "max"):
        assert k in j["cold"]["infer_ms"]


def _make_pan_video(path, frames=12):
    import cv2, numpy as np
    img = cv2.resize(cv2.imread(str(IMGS[0])), (640, 480))
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
    for f in range(frames):
        m = np.float32([[1, 0, 3 * f], [0, 1, 0]])
        vw.write(cv2.warpAffine(img, m, (640, 480), borderValue=(114, 114, 114)))
    vw.release()


def test_video_track():
    try:
        import cv2  # noqa
    except ImportError:
        return _skip("opencv-python missing")
    with tempfile.TemporaryDirectory() as td:
        vp = Path(td) / "pan.mp4"; _make_pan_video(vp)
        try:
            frames = list(c.video(vp, model=MODEL, track="bytetrack"))
        except ApiError as e:
            if e.status == 501:
                return _skip("server built without videoio")
            raise
    assert frames[-1]["done"] and frames[-1]["track"] == "bytetrack" and frames[-1]["frames"] == 12
    body = frames[:-1]
    assert all(all("track_id" in d for d in f["detections"]) for f in body)
    ids = [d["track_id"] for f in body for d in f["detections"]]
    assert ids and len(set(ids)) < len(ids)          # ids repeat across frames
    # tracks persist: the most frequent id is present on nearly every frame (the highest-conf
    # detection may alternate between two objects, so the top-1 id alone is not the measure)
    per_frame = [set(d["track_id"] for d in f["detections"]) for f in body]
    best = max(ids, key=lambda i: sum(i in s for s in per_frame))
    assert sum(best in s for s in per_frame) >= len(body) - 2


def test_ws_track():
    try:
        import cv2, websocket  # noqa
    except ImportError:
        return _skip("opencv-python/websocket-client missing")
    with tempfile.TemporaryDirectory() as td:
        vp = Path(td) / "pan.mp4"; _make_pan_video(vp, 8)
        ws = websocket.create_connection(URL.replace("http://", "ws://") + f"/v1/stream?model={MODEL}&track=botsort", timeout=60)
        json.loads(ws.recv())                                    # hello
        cap = cv2.VideoCapture(str(vp)); got = []
        for i in range(8):
            ok, frame = cap.read(); assert ok
            if i == 5:                                           # switch tracking off mid-stream
                ws.send(json.dumps({"track": "off"})); ack = json.loads(ws.recv()); assert ack["params"]["track"] == "off"
            ws.send_binary(cv2.imencode(".jpg", frame)[1].tobytes())
            got.append(json.loads(ws.recv()))
        ws.close(); cap.release()
    assert len(got) == 8 and all("count" in g for g in got)
    assert all("track_id" in d for g in got[:5] for d in g["detections"])
    assert all("track_id" not in d for g in got[5:] for d in g["detections"])


def test_request_id_and_traceparent():
    rid = "client-rid-42"
    c.request_id = rid
    try:
        r = c.infer(IMGS[0], model=MODEL)
    finally:
        c.request_id = None
    h = {k.lower(): v for k, v in c.last_headers.items()}
    assert r["request_id"] == rid and h["x-request-id"] == rid
    tp = h["traceparent"]; parts = tp.split("-")
    assert len(parts) == 4 and parts[0] == "00" and len(parts[1]) == 32 and len(parts[2]) == 16
    # an unsafe inbound id is replaced, an inbound traceparent keeps its trace id with a new span id
    req = urllib.request.Request(URL + "/healthz", headers={"X-Request-Id": "bad id with spaces" * 8,
                                                          "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        hh = {k.lower(): v for k, v in resp.headers.items()}
    assert hh["x-request-id"] != "bad id with spaces" * 8
    assert hh["traceparent"].startswith("00-0af7651916cd43dd8448eb211c80319c-") and hh["traceparent"].split("-")[2] != "b7ad6b7169203331"


def test_openapi_document():
    doc = json.loads(c._req("GET", "/openapi.json")[2])
    assert doc["openapi"] == "3.1.0" and doc["info"]["version"]
    paths = set(doc["paths"])
    listed = set(e.replace(" (ws)", "") for e in c._req("GET", "/")[2] and json.loads(c._req("GET", "/")[2])["endpoints"])
    assert {"/v1/infer", "/v1/infer/batch", "/v1/video", "/v1/bench", "/v1/stream", "/v1/models/{id}/load", "/metrics", "/openapi.json"} <= paths
    assert listed <= {p.replace("{id}", ":id") for p in paths} | {"/v1/models/:id/load", "/v1/models/:id/unload"}
    assert doc["paths"]["/v1/stream"]["get"].get("x-websocket") is True
    assert "InferResult" in doc["components"]["schemas"] and "ApiKeyHeader" in doc["components"]["securitySchemes"]
    try:
        from openapi_spec_validator import validate  # type: ignore
        validate(doc)
    except ImportError:
        pass


def test_load_unload_roundtrip():
    if not MODEL2:
        return _skip("no second model")
    assert c.unload(MODEL2)["loaded"] is False
    assert c.load(MODEL2)["loaded"] is True
    assert c.wait_ready(120)
    assert c.infer(IMGS[0], model=MODEL2)["count"] > 0


if __name__ == "__main__":
    fails = 0
    for name, fn in [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]:
        try:
            fn(); print("  PASS ", name)
        except Exception as e:  # noqa
            fails += 1; print("  FAIL ", name, repr(e)[:300])
    print(f"RESULT: {fails} failed")
    sys.exit(1 if fails else 0)
