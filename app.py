"""Zero-dependency web UI for the conveyor inspection pipeline.

The brief says a polished UI is not required, but one was asked for, so it is
built without adding a single dependency: stdlib http.server plus the pipeline
modules that already exist. requirements.txt pins upper version bounds so the
project installs reproducibly on a lab machine, and a dashboard does not justify
a framework that can break that install.

Run:  python app.py  ->  http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from pipeline import PipelineConfig, PipelineProgress, run_pipeline

ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "webui"
JOB_ROOT = ROOT / "outputs" / "ui_jobs"
UPLOAD_ROOT = ROOT / "data" / "uploads"
MAX_PREVIEW_EDGE = 720
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_stem(name: str) -> str:
    """Reduce a client-supplied filename to a safe stem.

    The name arrives from the browser and must never be able to steer the write
    outside UPLOAD_ROOT, so separators and traversal sequences are stripped
    before it is used as a path component.
    """
    stem = Path(name.replace("\\", "/")).name
    stem = _SAFE_NAME.sub("_", stem).strip("._-")
    return stem[:64] or "upload"


def parse_multipart(body: bytes, content_type: str) -> dict[str, tuple[str | None, bytes]]:
    """Parse a multipart/form-data body into {field: (filename, data)}.

    The stdlib cgi module was removed in Python 3.13, so this uses email.parser,
    which still parses RFC 2046 multipart correctly.
    """
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
    )
    fields: dict[str, tuple[str | None, bytes]] = {}
    if not message.is_multipart():
        return fields
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        fields[str(name)] = (part.get_filename(), payload)
    return fields


def _json_safe(value: object) -> object:
    """Replace non-finite floats with None so the payload is strict JSON.

    The pipeline reports missing sizes and speeds as NaN. Python's json module
    would happily emit a bare NaN token, but JSON.parse rejects it, so the
    browser would fail on the whole response. Emitting null instead keeps the
    document valid and lets the UI show a dash for the affected cell.
    """
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return _json_safe(value.item())
    return value


def _json_bytes(payload: object) -> bytes:
    return json.dumps(_json_safe(payload), allow_nan=False, default=str).encode("utf-8")


@dataclass
class Job:
    id: str
    config: PipelineConfig
    status: str = "queued"
    error: str | None = None
    processed: int = 0
    max_frames: int = 0
    detections: int = 0
    tracks: int = 0
    latest_speed: float = float("nan")
    result: dict | None = None
    preview_jpeg: bytes | None = None
    thread: threading.Thread | None = field(default=None, repr=False)

    def snapshot(self, include_preview: bool = True) -> dict:
        payload = {
            "id": self.id,
            "status": self.status,
            "error": self.error,
            "processed": self.processed,
            "max_frames": self.max_frames,
            "detections": self.detections,
            "tracks": self.tracks,
            "latest_speed": self.latest_speed,
            "result": self.result,
        }
        if include_preview:
            payload["preview_url"] = f"/api/preview/{self.id}"
            payload["has_preview"] = self.preview_jpeg is not None
        return payload


class JobRegistry:
    def __init__(self, limit: int = 8) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._limit = limit

    def create(self, config: PipelineConfig, job_id: str | None = None) -> Job:
        job = Job(id=job_id or uuid.uuid4().hex[:12], config=config, max_frames=config.max_frames)
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > self._limit:
                oldest = next(
                    (j for j in self._jobs.values() if j.status in {"done", "error"}),
                    None,
                )
                if oldest is None or oldest.id == job.id:
                    break
                self._jobs.pop(oldest.id, None)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.id)

    def start(self, job: Job) -> None:
        def on_progress(progress: PipelineProgress) -> None:
            job.processed = progress.processed
            job.detections = progress.detections
            job.tracks = progress.tracks
            job.latest_speed = progress.latest_speed
            if progress.preview is not None:
                job.preview_jpeg = _encode_jpeg(progress.preview)

        def worker() -> None:
            job.status = "running"
            try:
                result = run_pipeline(job.config, on_progress=on_progress)
                payload = result.as_json()
                artifacts = payload.get("artifacts") or {}
                artifacts["annotated_video"] = _relative_to_root(result.annotated_video)
                artifacts["results_csv"] = _relative_to_root(result.results_csv)
                artifacts["flow_preview"] = _relative_to_root(result.flow_preview)
                payload["artifacts"] = artifacts
                job.result = payload
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI verbatim
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()

        thread = threading.Thread(target=worker, daemon=True, name=f"job-{job.id}")
        job.thread = thread
        thread.start()


def _encode_jpeg(frame: np.ndarray, quality: int = 70) -> bytes:
    height, width = frame.shape[:2]
    scale = min(1.0, MAX_PREVIEW_EDGE / max(height, width))
    if scale < 1.0:
        frame = cv2.resize(
            frame, (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return b""
    return encoded.tobytes()


ARTIFACT_GROUPS = (
    ("Segmentation (A)", "outputs/segmentation", ("segmentation_comparison.png",)),
    ("Segmentation, touching (A)", "outputs/segmentation_touching", ("segmentation_comparison.png",)),
    ("Calibration (B)", "outputs", ("undistortion.png",)),
    ("Optical flow (C)", "outputs/optical_flow", ("lucas_kanade_flow.png", "sparse_flow.png", "affine_ransac_flow.png")),
    ("Tracking (D)", "outputs/kalman", ("tracking_plot.png",)),
    ("Recognition (E)", "outputs/recognition", ("eigenvectors.png",)),
)

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".csv": "text/csv; charset=utf-8",
    ".mp4": "video/mp4",
}


def _relative_to_root(path: Path | None) -> str | None:
    """Render an artifact path relative to the project root.

    The pipeline records absolute paths, which are correct on disk but useless
    to a browser and not portable. Emitting a root-relative path keeps the JSON
    host-independent and lets the client build /api/artifact/<path> directly.
    """
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


class Handler(BaseHTTPRequestHandler):
    registry: JobRegistry
    protocol_version = "HTTP/1.1"

    def handle_one_request(self) -> None:
        """Swallow client disconnects instead of dumping a traceback.

        With HTTP/1.1 keep-alive the browser holds the socket open, and a page
        refresh or an aborted poll closes it mid-read. That surfaces as
        ConnectionResetError / BrokenPipeError raised from readline() before any
        handler runs, which socketserver would otherwise log as an unhandled
        exception with a full stack trace. It is normal traffic, not a fault.
        """
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            self.close_connection = True

    def log_message(self, fmt: str, *args: object) -> None:  # quieter console
        print(f"[ui] {self.address_string()} {fmt % args}")

    # -- helpers ---------------------------------------------------------
    def _send(self, status: HTTPStatus, body: bytes, content_type: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(status, _json_bytes(payload), CONTENT_TYPES[".json"])

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status)

    def _resolve(self, relative: str) -> Path | None:
        """Resolve a path for serving, refusing anything outside the project.

        Accepts both root-relative ("outputs/x.png") and absolute paths, because
        the pipeline reports absolute artifact locations. Anything resolving
        outside ROOT is rejected, so the endpoint cannot be used to read
        arbitrary files off the machine.
        """
        candidate = Path(relative)
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(ROOT.resolve())
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    # -- routing ---------------------------------------------------------
    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route in {"/", "/index.html"}:
                self._serve_static("index.html")
            elif route == "/api/jobs":
                self._json([job.snapshot(include_preview=False) for job in self.registry.all()])
            elif route == "/api/artifacts":
                self._json(self._artifact_index())
            elif route.startswith("/api/status/"):
                job = self.registry.get(route.rsplit("/", 1)[-1])
                if job is None:
                    self._error(HTTPStatus.NOT_FOUND, "unknown job")
                else:
                    self._json(job.snapshot())
            elif route.startswith("/api/preview/"):
                self._serve_preview(route.rsplit("/", 1)[-1])
            elif route.startswith("/api/artifact/"):
                self._serve_file(route[len("/api/artifact/"):])
            else:
                self._serve_static(route.lstrip("/"))
        except BrokenPipeError:
            pass
        except Exception as exc:  # noqa: BLE001
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if parsed.path == "/api/upload":
                self._handle_upload(length)
                return
            raw = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw or b"{}")
            if parsed.path == "/api/run":
                self._start_job(payload)
            else:
                self._error(HTTPStatus.NOT_FOUND, "unknown route")
        except json.JSONDecodeError as exc:
            self._error(HTTPStatus.BAD_REQUEST, f"invalid JSON: {exc}")
        except Exception as exc:  # noqa: BLE001
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    # -- upload ----------------------------------------------------------
    def _handle_upload(self, length: int) -> None:
        content_type = self.headers.get("Content-Type") or ""
        if "multipart/form-data" not in content_type:
            self._error(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "expected multipart/form-data"
            )
            return
        if length <= 0:
            self._error(HTTPStatus.BAD_REQUEST, "empty upload")
            return
        if length > MAX_UPLOAD_BYTES:
            self._error(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
            )
            return
        body = self.rfile.read(length)
        fields = parse_multipart(body, content_type)
        entry = fields.get("file")
        if entry is None or entry[1] == b"":
            self._error(HTTPStatus.BAD_REQUEST, "no file field named 'file'")
            return
        filename, data = entry
        suffix = Path(filename or "").suffix.lower()
        if suffix not in ALLOWED_VIDEO_SUFFIXES:
            self._error(
                HTTPStatus.BAD_REQUEST,
                f"unsupported type '{suffix or filename}'; expected one of "
                f"{', '.join(sorted(ALLOWED_VIDEO_SUFFIXES))}",
            )
            return
        UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        target = UPLOAD_ROOT / f"{_safe_stem(Path(filename or 'upload').stem)}{suffix}"
        counter = 1
        while target.exists():
            target = UPLOAD_ROOT / f"{_safe_stem(Path(filename or 'upload').stem)}_{counter}{suffix}"
            counter += 1
        target.write_bytes(data)
        try:
            relative = target.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            relative = str(target)
        self._json(
            {
                "filename": target.name,
                "path": relative,
                "bytes": len(data),
                "url": f"/api/artifact/{relative}",
            },
            HTTPStatus.CREATED,
        )

    # -- static ----------------------------------------------------------
    def _serve_static(self, name: str) -> None:
        path = (WEB_ROOT / name).resolve()
        try:
            path.relative_to(WEB_ROOT.resolve())
        except ValueError:
            self._error(HTTPStatus.FORBIDDEN, "forbidden")
            return
        if not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, f"no such asset: {name}")
            return
        body = path.read_bytes()
        self._send(HTTPStatus.OK, body, CONTENT_TYPES.get(path.suffix, "application/octet-stream"))

    def _serve_preview(self, job_id: str) -> None:
        job = self.registry.get(job_id)
        if job is None or job.preview_jpeg is None:
            self._error(HTTPStatus.NOT_FOUND, "no preview yet")
            return
        self._send(HTTPStatus.OK, job.preview_jpeg, CONTENT_TYPES[".jpg"])

    def _artifact_index(self) -> list[dict]:
        entries = []
        for title, folder, names in ARTIFACT_GROUPS:
            images = []
            for name in names:
                path = ROOT / folder / name
                if path.is_file():
                    images.append({"name": name, "url": f"/api/artifact/{folder}/{name}"})
            entries.append({"title": title, "images": images})
        return entries

    def _serve_file(self, relative: str) -> None:
        path = self._resolve(relative)
        if path is None:
            self._error(HTTPStatus.NOT_FOUND, f"no such artifact: {relative}")
            return
        content_type = CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        size = path.stat().st_size
        range_header = self.headers.get("Range")
        if range_header and path.suffix == ".mp4":
            self._serve_range(path, range_header, content_type, size)
            return
        self._send(HTTPStatus.OK, path.read_bytes(), content_type)

    def _serve_range(self, path: Path, range_header: str, content_type: str, size: int) -> None:
        start_text, _, end_text = range_header.removeprefix("bytes=").partition("-")
        try:
            start = int(start_text) if start_text else 0
            end = int(end_text) if end_text else size - 1
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "malformed Range header")
            return
        start = max(0, start)
        end = min(size - 1, end)
        if start > end:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        with path.open("rb") as handle:
            handle.seek(start)
            chunk = handle.read(end - start + 1)
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)

    # -- jobs ------------------------------------------------------------
    def _start_job(self, payload: dict) -> None:
        video = str(payload.get("video") or "").strip()
        if not video:
            self._error(HTTPStatus.BAD_REQUEST, "'video' is required")
            return
        video_path = Path(video)
        if not video_path.is_absolute():
            video_path = ROOT / video_path
        if not video_path.is_file():
            self._error(HTTPStatus.BAD_REQUEST, f"video not found: {video}")
            return

        def optional_path(key: str) -> Path | None:
            value = str(payload.get(key) or "").strip()
            if not value:
                return None
            candidate = Path(value)
            return (candidate if candidate.is_absolute() else ROOT / candidate)

        def as_int(key: str, default: int, low: int, high: int) -> int:
            try:
                return max(low, min(high, int(payload.get(key, default))))
            except (TypeError, ValueError):
                return default

        def as_float(key: str) -> float | None:
            value = payload.get(key)
            if value in (None, ""):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        belt = payload.get("belt_velocity")
        belt_velocity = None
        if isinstance(belt, (list, tuple)) and len(belt) == 2:
            try:
                belt_velocity = (float(belt[0]), float(belt[1]))
            except (TypeError, ValueError):
                belt_velocity = None

        job_id = str(payload.get("job_id") or uuid.uuid4().hex[:12])
        config = PipelineConfig(
            video=video_path,
            calibration=optional_path("calibration"),
            recognition_root=optional_path("recognition_root"),
            cm_per_pixel=as_float("cm_per_pixel"),
            output=JOB_ROOT / job_id,
            max_frames=as_int("max_frames", 300, 1, 100000),
            stride=as_int("stride", 1, 1, 240),
            flow_every=as_int("flow_every", 5, 1, 100000),
            max_boxes=as_int("max_boxes", 12, 1, 400),
            keep_static=bool(payload.get("keep_static")),
            belt_velocity=belt_velocity,
            write_video=True,
        )
        job = self.registry.create(config, job_id=job_id)
        self.registry.start(job)
        self._json(job.snapshot(), HTTPStatus.ACCEPTED)

    def _serve_annotated_video(self, job_id: str) -> None:  # pragma: no cover - helper
        job = self.registry.get(job_id)
        if job is None:
            self._error(HTTPStatus.NOT_FOUND, "unknown job")
            return
        result = job.result or {}
        relative = (result.get("artifacts") or {}).get("annotated_video")
        if not relative:
            self._error(HTTPStatus.NOT_FOUND, "no video for this job")
            return
        self._serve_file(relative)


class QuietThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address) -> None:
        """Report only genuine faults; client disconnects are not errors."""
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return
        traceback.print_exc()


def build_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    Handler.registry = JobRegistry()
    return QuietThreadingHTTPServer((host, port), Handler)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Serve the conveyor inspection UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    started = time.perf_counter()
    server = build_server(args.host, args.port)
    elapsed = (time.perf_counter() - started) * 1000

    print(f"Conveyor inspection UI ready in {elapsed:.0f} ms")
    print(f"  Open  http://{args.host}:{args.port}")
    print("  Press Ctrl+C to stop.")
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()