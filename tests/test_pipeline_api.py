"""Tests for the reusable pipeline API and the UI server helpers.

These guard the refactor that extracted run_pipeline() out of the argparse CLI,
and the JSON sanitisation the browser depends on. The pipeline equivalence tests
pin the frame and record counts that the CLI produced before the refactor, so a
future change to the loop cannot silently alter the documented behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app import (
    ALLOWED_VIDEO_SUFFIXES,
    JobRegistry,
    _json_bytes,
    _json_safe,
    _relative_to_root,
    _safe_stem,
    parse_multipart,
)
from pipeline import PipelineConfig, PipelineProgress, run_pipeline

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "data" / "smoke.mp4"


@pytest.fixture(scope="module")
def smoke_video() -> Path:
    if not SMOKE.is_file():
        pytest.skip(f"missing fixture: {SMOKE}")
    return SMOKE


def test_run_pipeline_returns_a_result_and_writes_artifacts(smoke_video, tmp_path) -> None:
    result = run_pipeline(
        PipelineConfig(video=smoke_video, output=tmp_path / "run", cm_per_pixel=0.05, max_frames=6)
    )
    assert result.frames_processed == 6
    assert result.rows
    assert (result.output / "pipeline_summary.json").is_file()
    assert result.results_csv is not None and result.results_csv.is_file()
    assert result.annotated_video is not None and result.annotated_video.is_file()


def test_max_frames_bounds_the_loop(smoke_video, tmp_path) -> None:
    result = run_pipeline(
        PipelineConfig(video=smoke_video, output=tmp_path / "bounded", max_frames=3, write_video=False)
    )
    assert result.frames_processed == 3
    assert len(result.rows) <= 3 * 12


@pytest.mark.parametrize(
    ("stride", "frames", "records"),
    [(1, 45, 88), (2, 23, 44), (3, 15, 28)],
)
def test_stride_counts_match_the_pre_refactor_cli(smoke_video, tmp_path, stride, frames, records) -> None:
    """Counts pinned from the CLI before run_pipeline() existed (see WORK_LOG_2)."""
    result = run_pipeline(
        PipelineConfig(
            video=smoke_video,
            output=tmp_path / f"stride{stride}",
            stride=stride,
            cm_per_pixel=0.05,
        )
    )
    assert result.frames_processed == frames
    assert len(result.rows) == records


def test_write_video_false_skips_the_mp4_but_keeps_the_csv(smoke_video, tmp_path) -> None:
    result = run_pipeline(
        PipelineConfig(video=smoke_video, output=tmp_path / "novideo", max_frames=4, write_video=False)
    )
    assert result.annotated_video is None
    assert result.results_csv is not None and result.results_csv.is_file()
    assert not (result.output / "annotated.mp4").exists()


def test_progress_callback_fires_once_per_frame_with_a_preview(smoke_video, tmp_path) -> None:
    seen: list[PipelineProgress] = []
    run_pipeline(
        PipelineConfig(video=smoke_video, output=tmp_path / "progress", max_frames=5, write_video=False),
        on_progress=seen.append,
    )
    assert len(seen) == 5
    assert [event.processed for event in seen] == [1, 2, 3, 4, 5]
    assert all(event.preview is not None for event in seen)
    assert all(event.preview.ndim == 3 for event in seen)


def test_invalid_stride_and_flow_every_are_rejected(smoke_video, tmp_path) -> None:
    for field in ("stride", "flow_every"):
        config = PipelineConfig(video=smoke_video, output=tmp_path / field, max_frames=1)
        setattr(config, field, 0)
        with pytest.raises(ValueError):
            run_pipeline(config)


def test_summary_reports_the_features_actually_in_use(smoke_video, tmp_path) -> None:
    result = run_pipeline(
        PipelineConfig(video=smoke_video, output=tmp_path / "summary", max_frames=3)
    )
    assert result.summary["calibrated"] is False
    assert result.summary["recognition_enabled"] is False
    assert result.as_json()["calibrated"] is False


def test_as_json_is_strict_json_despite_nan(smoke_video, tmp_path) -> None:
    """NaN sizes reach the payload; json.dumps would emit a bare NaN token."""
    result = run_pipeline(
        PipelineConfig(video=smoke_video, output=tmp_path / "nan", max_frames=4)
    )
    payload = result.as_json()
    assert json.loads(json.dumps(_json_safe(payload)))["frames_processed"] == 4


class TestJsonSafe:
    def test_non_finite_floats_become_null(self) -> None:
        cleaned = _json_safe({"a": float("nan"), "b": float("inf"), "c": 1.5})
        assert cleaned == {"a": None, "b": None, "c": 1.5}

    def test_numpy_scalars_are_unwrapped(self) -> None:
        cleaned = _json_safe({"x": np.float64(2.5), "i": np.int64(7)})
        assert cleaned == {"x": 2.5, "i": 7}
        assert isinstance(cleaned["i"], int)

    def test_nested_containers_are_walked(self) -> None:
        cleaned = _json_safe([{"v": [float("nan"), 2.0]}])
        assert cleaned == [{"v": [None, 2.0]}]

    def test_serialised_payload_has_no_nan_token(self) -> None:
        raw = _json_bytes({"speed": float("nan")})
        assert b"NaN" not in raw
        assert json.loads(raw) == {"speed": None}


class TestArtifactPaths:
    def test_relative_to_root_renders_posix_paths(self, tmp_path) -> None:
        assert _relative_to_root(ROOT / "outputs" / "x.png") == "outputs/x.png"

    def test_relative_to_root_passes_through_none(self) -> None:
        assert _relative_to_root(None) is None

    def test_path_outside_root_is_returned_verbatim(self, tmp_path) -> None:
        outside = tmp_path / "elsewhere.png"
        assert _relative_to_root(outside) == str(outside)


class TestJobRegistry:
    def test_explicit_id_is_used_and_matches_output_folder(self, tmp_path) -> None:
        registry = JobRegistry()
        config = PipelineConfig(video=SMOKE, output=tmp_path / "abc123", max_frames=1)
        job = registry.create(config, job_id="abc123")
        assert job.id == "abc123"
        assert job.config.output.name == "abc123"

    def test_generated_id_is_used_when_absent(self, tmp_path) -> None:
        registry = JobRegistry()
        job = registry.create(PipelineConfig(video=SMOKE, output=tmp_path / "j", max_frames=1))
        assert len(job.id) == 12

    def test_unknown_job_lookup_returns_none(self) -> None:
        assert JobRegistry().get("nope") is None

    def test_snapshot_exposes_progress_fields(self, tmp_path) -> None:
        registry = JobRegistry()
        job = registry.create(PipelineConfig(video=SMOKE, output=tmp_path / "s", max_frames=9))
        job.processed, job.tracks = 4, 2
        snapshot = job.snapshot()
        assert snapshot["processed"] == 4
        assert snapshot["max_frames"] == 9
        assert snapshot["tracks"] == 2
        assert snapshot["has_preview"] is False

class TestMultipartUpload:
    @staticmethod
    def build(fields: dict[str, tuple[str | None, bytes]]) -> tuple[str, bytes]:
        boundary = "----cvtest"
        chunks = []
        for name, (filename, data) in fields.items():
            disposition = f'form-data; name="{name}"'
            if filename is not None:
                disposition += f'; filename="{filename}"'
            chunks.append(f"--{boundary}\r\n".encode())
            chunks.append(f"Content-Disposition: {disposition}\r\n".encode())
            chunks.append(b"Content-Type: application/octet-stream\r\n\r\n")
            chunks.append(data)
            chunks.append(b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode())
        return f"multipart/form-data; boundary={boundary}", b"".join(chunks)

    def test_parses_a_file_field(self) -> None:
        content_type, body = self.build({"file": ("clip.mp4", b"binary-payload")})
        fields = parse_multipart(body, content_type)
        assert fields["file"] == ("clip.mp4", b"binary-payload")

    def test_parses_a_plain_text_field(self) -> None:
        content_type, body = self.build({"video": (None, b"data/smoke.mp4")})
        fields = parse_multipart(body, content_type)
        assert fields["video"] == (None, b"data/smoke.mp4")

    def test_handles_binary_bytes_exactly(self) -> None:
        payload = bytes(range(256)) * 8
        content_type, body = self.build({"file": ("x.mp4", payload)})
        assert parse_multipart(body, content_type)["file"][1] == payload

    def test_empty_body_yields_no_fields(self) -> None:
        content_type, body = self.build({})
        assert parse_multipart(body, content_type) == {}


class TestSafeStem:
    def test_plain_name_is_kept(self) -> None:
        assert _safe_stem("clip.mp4") == "clip.mp4"

    def test_spaces_and_specials_become_underscores(self) -> None:
        assert _safe_stem("my clip.mp4") == "my_clip.mp4"

    def test_traversal_is_flattened(self) -> None:
        for hostile in ("../../../evil.mp4", "..\\..\\evil.mp4", "/etc/passwd.mp4"):
            assert "/" not in _safe_stem(hostile)
            assert "\\" not in _safe_stem(hostile)
            assert ".." not in _safe_stem(hostile)

    def test_empty_name_gets_a_placeholder(self) -> None:
        assert _safe_stem("") == "upload"
        assert _safe_stem("///") == "upload"

    def test_long_names_are_truncated(self) -> None:
        assert len(_safe_stem("a" * 300 + ".mp4")) <= 64


class TestAllowedVideoSuffixes:
    def test_common_video_types_are_allowed(self) -> None:
        for suffix in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"):
            assert suffix in ALLOWED_VIDEO_SUFFIXES

    def test_executables_and_archives_are_not_allowed(self) -> None:
        for suffix in (".exe", ".py", ".zip", ".sh", ".bat"):
            assert suffix not in ALLOWED_VIDEO_SUFFIXES
