#!/usr/bin/env python3
"""Python client for the yolomaster_server API (REST + WebSocket).

    from yolomaster_client import Client
    c = Client("http://localhost:8080")
    r = c.infer("img.jpg", model="v01n-trt", conf=0.25)        # dict (JSON response)
    txt = c.infer("img.jpg", model="v01n-trt", ret="txt")     # 'class conf x1 y1 x2 y2' lines
    for frame in c.stream_video("clip.mp4", model="v01n-trt"):  # WebSocket, one dict per frame
        ...

CLI:  python yolomaster_client.py --url http://host:8080 --model v01n-trt infer img.jpg [--return json|txt|coco|annotated]
      python yolomaster_client.py ... models | stats | ready
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
from typing import Iterator, Optional, Union
import urllib.request, urllib.error

Bytes = Union[bytes, str, Path]


class ApiError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status, self.body = status, body


class Client:
    def __init__(self, base_url: str = "http://localhost:8080", timeout: float = 30.0, api_key: Optional[str] = None):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key                # sent as X-API-Key on every request (and the WS upgrade)
        self.request_id: Optional[str] = None  # optional X-Request-Id to send with the next request(s)
        self.last_headers: dict = {}

    # ---- helpers ----
    def _headers(self) -> dict:
        h = {}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        if self.request_id:
            h["X-Request-Id"] = self.request_id
        return h

    @staticmethod
    def _bytes(x: Bytes) -> bytes:
        return x if isinstance(x, (bytes, bytearray)) else Path(x).read_bytes()

    def _req(self, method: str, path: str, data: Optional[bytes] = None, ctype: str = "application/octet-stream",
             query: Optional[dict] = None):
        url = self.base + path
        if query:
            q = "&".join(f"{k}={v}" for k, v in query.items() if v is not None and v != "")
            if q:
                url += ("&" if "?" in url else "?") + q
        req = urllib.request.Request(url, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", ctype)
        for k, v in self._headers().items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                self.last_headers = dict(r.headers)
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            raise ApiError(e.code, e.read().decode(errors="replace")) from None

    # ---- endpoints ----
    def health(self) -> dict:
        return json.loads(self._req("GET", "/healthz")[2])

    def ready(self) -> bool:
        try:
            return json.loads(self._req("GET", "/readyz")[2]).get("ready", False)
        except ApiError as e:
            return False

    def wait_ready(self, timeout: float = 120.0, poll: float = 0.5) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if self.ready():
                    return True
            except Exception:
                pass
            time.sleep(poll)
        return False

    def models(self) -> list:
        return json.loads(self._req("GET", "/v1/models")[2])["models"]

    def load(self, model: str) -> dict:
        return json.loads(self._req("POST", f"/v1/models/{model}/load", b"")[2])

    def unload(self, model: str) -> dict:
        return json.loads(self._req("POST", f"/v1/models/{model}/unload", b"")[2])

    def stats(self) -> dict:
        return json.loads(self._req("GET", "/v1/stats")[2])

    def metrics(self) -> str:
        return self._req("GET", "/metrics")[2].decode()

    def bench(self, model: Optional[str] = None, warmup: int = 10, iters: int = 50) -> dict:
        """POST /v1/bench: gray-probe run on one worker of a loaded model -> yolomaster-bench/v1 JSON."""
        _, _, body = self._req("POST", "/v1/bench", b"", "application/octet-stream",
                               {"model": model, "warmup": warmup, "iters": iters})
        return json.loads(body)

    def infer(self, image: Bytes, model: Optional[str] = None, ret: str = "json", **params):
        """params: conf, iou, max_det, multi_label, slicing, tile_size, masks=overlay, mask_coeffs, image_id, coco91, names, quality.
        Returns dict for json/coco, str for txt, bytes for annotated. Also sets self.last_headers."""
        q = {"model": model, "return": ret, **params}
        status, headers, body = self._req("POST", "/v1/infer", self._bytes(image), "image/*", q)
        self.last_headers = headers
        if ret == "txt":
            return body.decode()
        if ret == "annotated":
            return body
        return json.loads(body)

    def infer_many(self, images: list, model: Optional[str] = None, **params) -> list:
        """One multipart request, K files -> K results (independent batch-1 jobs on the server)."""
        boundary = "----ymclient" + str(int(time.time() * 1000))
        parts = []
        for i, img in enumerate(images):
            name = Path(img).name if not isinstance(img, (bytes, bytearray)) else f"img{i}.bin"
            parts.append((f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
                          f"Content-Type: application/octet-stream\r\n\r\n").encode() + self._bytes(img) + b"\r\n")
        body = b"".join(parts) + f"--{boundary}--\r\n".encode()
        q = {"model": model, **params}
        _, _, out = self._req("POST", "/v1/infer/batch", body, f"multipart/form-data; boundary={boundary}", q)
        return json.loads(out)["results"]

    def video(self, path: Bytes, model: Optional[str] = None, every: int = 1, max_frames: int = 0, **params) -> Iterator[dict]:
        """POST a video file; yields one dict per processed frame (NDJSON stream), last item has done=True.
        track="botsort"|"bytetrack" adds a track_id to every detection."""
        q = {"model": model, "every": every, "max_frames": max_frames, **params}
        url = self.base + "/v1/video?" + "&".join(f"{k}={v}" for k, v in q.items() if v is not None)
        req = urllib.request.Request(url, data=self._bytes(path), method="POST")
        req.add_header("Content-Type", "application/octet-stream")
        for k, v in self._headers().items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=max(self.timeout, 600)) as r:
            for line in r:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def stream_video(self, path: Bytes, model: Optional[str] = None, every: int = 1, jpeg_quality: int = 90,
                     max_frames: int = 0, **params) -> Iterator[dict]:
        """WebSocket /v1/stream: sends JPEG frames (needs opencv-python + websocket-client), yields result dicts.
        Frames are sent without waiting; the server keeps only the latest queued frame per connection."""
        import cv2, websocket  # optional deps
        ws_url = self.base.replace("http://", "ws://").replace("https://", "wss://") + "/v1/stream?" + \
            "&".join(f"{k}={v}" for k, v in {"model": model, **params}.items() if v is not None)
        ws = websocket.create_connection(ws_url, timeout=self.timeout, header=[f"{k}: {v}" for k, v in self._headers().items()])
        hello = json.loads(ws.recv())
        cap = cv2.VideoCapture(str(path))
        sent = 0
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or (max_frames and sent >= max_frames):
                    break
                idx += 1
                if (idx - 1) % every:
                    continue
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                ws.send_binary(buf.tobytes())
                sent += 1
                msg = ws.recv()
                if isinstance(msg, bytes):   # annotated image frame precedes its JSON
                    msg = ws.recv()
                yield json.loads(msg)
        finally:
            ws.close()
            cap.release()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--api-key", default=os.environ.get("YM_API_KEY"), help="X-API-Key (or env YM_API_KEY)")
    ap.add_argument("--model", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("infer"); p.add_argument("image"); p.add_argument("--return", dest="ret", default="json")
    p.add_argument("--conf", type=float); p.add_argument("--iou", type=float); p.add_argument("--out", help="write body to file")
    sub.add_parser("models"); sub.add_parser("stats"); sub.add_parser("ready"); sub.add_parser("metrics")
    v = sub.add_parser("video"); v.add_argument("path"); v.add_argument("--every", type=int, default=1); v.add_argument("--ws", action="store_true")
    a = ap.parse_args(argv)
    c = Client(a.url, api_key=a.api_key)
    if a.cmd == "models":
        for m in c.models():
            print(f"{m['id']:24s} {m.get('backend','?'):6s} {m.get('ep',''):18s} loaded={m.get('loaded')} ready={m.get('ready')} imgsz={m.get('imgsz')} nc={m.get('nc')}")
    elif a.cmd == "stats":
        print(json.dumps(c.stats(), indent=2))
    elif a.cmd == "metrics":
        print(c.metrics())
    elif a.cmd == "ready":
        ok = c.ready(); print("ready" if ok else "not ready"); return 0 if ok else 1
    elif a.cmd == "infer":
        r = c.infer(a.image, model=a.model, ret=a.ret, conf=a.conf, iou=a.iou)
        if a.out:
            Path(a.out).write_bytes(r if isinstance(r, bytes) else (r if isinstance(r, str) else json.dumps(r, indent=2)).encode())
            print(f"wrote {a.out}")
        elif isinstance(r, bytes):
            sys.stdout.buffer.write(r)
        elif isinstance(r, str):
            print(r, end="")
        else:
            print(json.dumps(r, indent=2))
    elif a.cmd == "video":
        it = c.stream_video(a.path, model=a.model, every=a.every) if a.ws else c.video(a.path, model=a.model, every=a.every)
        for fr in it:
            print(json.dumps({k: fr[k] for k in fr if k in ("frame", "seq", "count", "timings_ms", "done", "error", "frames")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
