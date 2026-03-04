#!/usr/bin/env python

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm.auto import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames, encode_video_frames


def _select_video_key(camera_keys: list[str], requested_video_key: str | None) -> str:
    if len(camera_keys) == 0:
        raise ValueError("No camera key found in dataset.")
    if requested_video_key is not None:
        if requested_video_key not in camera_keys:
            raise ValueError(
                f"Unknown video_key '{requested_video_key}'. Available camera keys: {camera_keys}"
            )
        return requested_video_key

    for key in camera_keys:
        lower = key.lower()
        if ".front" in lower or "_front" in lower or lower.endswith("front"):
            return key
    return camera_keys[0]


def _parse_episodes_arg(episodes_arg: str, total_episodes: int) -> list[int]:
    value = episodes_arg.strip().lower()
    if value == "all":
        return list(range(total_episodes))

    parsed: set[int] = set()
    for token in episodes_arg.split(","):
        part = token.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid episode range '{part}'.")
            parsed.update(range(start, end + 1))
        else:
            parsed.add(int(part))

    episodes = sorted(parsed)
    for ep in episodes:
        if ep < 0 or ep >= total_episodes:
            raise ValueError(f"Episode index out of range: {ep}, total_episodes={total_episodes}.")
    return episodes


def _to_1d_float(values: list | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)[:, 0]
    return arr.reshape(-1)


def _to_1d_int(values: list | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)[:, 0]
    return arr.reshape(-1)


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    font_candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/local/share/fonts/DejaVuSans.ttf"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def _curve_points(
    values: np.ndarray,
    current_step: int,
    x0: int,
    y0: int,
    width: int,
    height: int,
    y_min: float,
    y_max: float,
) -> list[tuple[int, int]]:
    n = len(values)
    if n == 0:
        return []

    denom_x = max(1, n - 1)
    denom_y = max(1e-6, y_max - y_min)

    points = []
    last_step = min(current_step, n - 1)
    for i in range(last_step + 1):
        x = int(round(x0 + width * (i / denom_x)))
        y_norm = np.clip((float(values[i]) - y_min) / denom_y, 0.0, 1.0)
        y = int(round(y0 + (1.0 - y_norm) * height))
        points.append((x, y))
    return points


def _draw_overlay(
    frame: Image.Image,
    values: np.ndarray,
    current_step: int,
    advantage_t: float,
    acp_t: int,
    highlight_current_point: bool,
    y_min: float,
    y_max: float,
    indicators: np.ndarray | None = None,
) -> Image.Image:
    rgba = frame.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width, height = rgba.size
    margin = max(10, width // 80)
    chart_h = max(72, height // 4)
    chart_w = width - 2 * margin
    chart_x0 = margin
    chart_y0 = height - margin - chart_h

    draw.rectangle(
        (chart_x0 - 4, chart_y0 - 4, chart_x0 + chart_w + 4, chart_y0 + chart_h + 4),
        fill=(0, 0, 0, 110),
    )
    draw.line((chart_x0, chart_y0, chart_x0, chart_y0 + chart_h), fill=(200, 200, 200, 160), width=1)
    draw.line(
        (chart_x0, chart_y0 + chart_h, chart_x0 + chart_w, chart_y0 + chart_h),
        fill=(200, 200, 200, 160),
        width=1,
    )

    mid_y = chart_y0 + chart_h // 2
    draw.line((chart_x0, mid_y, chart_x0 + chart_w, mid_y), fill=(120, 120, 120, 120), width=1)

    points = _curve_points(values, current_step, chart_x0, chart_y0, chart_w, chart_h, y_min, y_max)
    curve_width = max(2, width // 320)
    white = (255, 255, 255, 255)
    red = (255, 80, 80, 255)
    if len(points) >= 2:
        if indicators is not None:
            for i in range(len(points) - 1):
                seg_color = red if int(indicators[i]) == 1 or int(indicators[i + 1]) == 1 else white
                draw.line([points[i], points[i + 1]], fill=seg_color, width=curve_width)
        else:
            draw.line(points, fill=white, width=curve_width)
    point_color = red if highlight_current_point else white
    if len(points) == 1:
        x, y = points[0]
        draw.ellipse((x - curve_width, y - curve_width, x + curve_width, y + curve_width), fill=point_color)
    elif len(points) > 1:
        x, y = points[-1]
        radius = max(2, curve_width + 1)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=point_color)

    font_size = max(18, height // 26)
    font = _load_font(font_size)
    lines = [f"advantage: {advantage_t:+.4f}", f"acp_indicator: {int(acp_t)}"]
    line_sizes = [draw.textbbox((0, 0), text, font=font) for text in lines]
    text_w = max(box[2] - box[0] for box in line_sizes)
    text_h = sum(box[3] - box[1] for box in line_sizes) + max(4, font_size // 4)
    box_pad = max(8, font_size // 3)
    box_x1 = width - margin
    box_x0 = box_x1 - text_w - 2 * box_pad
    box_y0 = margin
    box_y1 = box_y0 + text_h + 2 * box_pad
    draw.rectangle((box_x0, box_y0, box_x1, box_y1), fill=(0, 0, 0, 150))

    text_y = box_y0 + box_pad
    for idx, text in enumerate(lines):
        box = line_sizes[idx]
        line_h = box[3] - box[1]
        draw.text((box_x0 + box_pad, text_y), text, fill=(255, 255, 255, 255), font=font)
        text_y += line_h + max(4, font_size // 4)

    return Image.alpha_composite(rgba, overlay).convert("RGB")


def _build_output_video_path(output_dir: Path, repo_id: str, video_key: str, episode_index: int) -> Path:
    repo_tag = repo_id.replace("/", "_")
    key_tag = video_key.replace(".", "_")
    return output_dir / f"{repo_tag}_episode_{episode_index:04d}_{key_tag}.mp4"


def _decode_frames_at_timestamps(
    video_file: Path,
    timestamps_s: np.ndarray,
    tolerance_s: float,
    backend: str | None,
) -> list[Image.Image]:
    if timestamps_s.size == 0:
        return []
    frames = decode_video_frames(
        video_path=video_file,
        timestamps=timestamps_s.tolist(),
        tolerance_s=tolerance_s,
        backend=backend,
    )
    np_frames = frames.detach().cpu().numpy()
    if np_frames.ndim != 4:
        raise ValueError(f"Unexpected decoded frame tensor shape: {np_frames.shape}")
    if np_frames.shape[1] in (1, 3):
        np_frames = np.transpose(np_frames, (0, 2, 3, 1))
    if np_frames.dtype != np.uint8:
        if np.issubdtype(np_frames.dtype, np.floating):
            max_val = float(np.max(np_frames)) if np_frames.size > 0 else 1.0
            if max_val <= 1.0 + 1e-6:
                np_frames = np.clip(np_frames, 0.0, 1.0) * 255.0
            else:
                np_frames = np.clip(np_frames, 0.0, 255.0)
        else:
            np_frames = np.clip(np_frames, 0, 255)
        np_frames = np_frames.astype(np.uint8)
    return [Image.fromarray(np_frames[i]) for i in range(np_frames.shape[0])]


def _get_episode_video_time_bounds(
    dataset: LeRobotDataset,
    episode_index: int,
    video_key: str,
) -> tuple[float, float | None]:
    episodes = getattr(dataset.meta, "episodes", None)
    if episodes is None:
        return 0.0, None
    episodes_ds = episodes.with_format(None)
    if "episode_index" not in episodes_ds.column_names:
        return 0.0, None

    episode_indices = np.asarray(episodes_ds["episode_index"], dtype=np.int64).reshape(-1)
    matched = np.flatnonzero(episode_indices == episode_index)
    if matched.size == 0:
        return 0.0, None
    row = int(matched[0])

    from_col = f"videos/{video_key}/from_timestamp"
    to_col = f"videos/{video_key}/to_timestamp"
    from_ts = 0.0
    to_ts: float | None = None
    if from_col in episodes_ds.column_names:
        from_ts = float(episodes_ds[from_col][row])
    if to_col in episodes_ds.column_names:
        to_ts = float(episodes_ds[to_col][row])
    return from_ts, to_ts


def _get_video_encode_options(vcodec: str) -> tuple[dict[str, str], str]:
    if vcodec == "libsvtav1":
        return {"g": "2", "crf": "30", "preset": "12"}, "yuv420p"
    if vcodec == "h264_nvenc":
        return {
            "preset": "p4",
            "rc": "vbr",
            "cq": "28",
            "b": "0",
            "g": "60",
        }, "yuv420p"
    return {"g": "2", "crf": "30"}, "yuv420p"


def _get_episode_value_bounds(ep_values: np.ndarray) -> tuple[float, float]:
    if ep_values.shape[0] == 0:
        raise ValueError("Cannot determine value bounds from an empty episode.")
    return float(np.min(ep_values)), float(np.max(ep_values))


def _encode_pil_to_video(
    frames: list[Image.Image],
    video_path: Path,
    fps: int,
    vcodec: str,
) -> None:
    """Encode PIL frames directly to video without writing intermediate PNG files."""
    import av as _av

    video_options, pix_fmt = _get_video_encode_options(vcodec)

    video_path.parent.mkdir(parents=True, exist_ok=True)
    with _av.open(str(video_path), "w") as output:
        out_stream = output.add_stream(vcodec, fps, options=video_options)
        out_stream.pix_fmt = pix_fmt
        out_stream.width = frames[0].width
        out_stream.height = frames[0].height
        for pil_img in frames:
            av_frame = _av.VideoFrame.from_image(pil_img.convert("RGB"))
            for packet in out_stream.encode(av_frame):
                output.mux(packet)
        for packet in out_stream.encode():
            output.mux(packet)


def _export_single_episode(
    src_video_path: Path,
    dst_video_path: Path,
    ep_values: np.ndarray,
    ep_advantages: np.ndarray,
    ep_indicators: np.ndarray,
    episode_timestamps_s: np.ndarray,
    fps: int,
    vcodec: str,
    tolerance_s: float,
    video_backend: str | None,
    frame_storage_mode: str = "memory",
    temp_dir_root: Path | None = None,
) -> Path:
    y_min, y_max = _get_episode_value_bounds(ep_values)
    decoded_frames = _decode_frames_at_timestamps(
        video_file=src_video_path,
        timestamps_s=episode_timestamps_s,
        tolerance_s=tolerance_s,
        backend=video_backend,
    )
    n_frames = min(len(decoded_frames), len(ep_values))
    if n_frames == 0:
        raise ValueError(f"No decoded frames for video: {src_video_path}")

    if frame_storage_mode == "disk":
        with tempfile.TemporaryDirectory(
            dir=str(temp_dir_root) if temp_dir_root is not None else None,
            prefix=f"{dst_video_path.stem}-frames-",
        ) as temp_dir:
            temp_path = Path(temp_dir)
            for i in range(n_frames):
                frame = decoded_frames[i]
                composed = _draw_overlay(
                    frame=frame,
                    values=ep_values,
                    current_step=i,
                    advantage_t=float(ep_advantages[i]),
                    acp_t=int(ep_indicators[i]),
                    highlight_current_point=(int(ep_indicators[i]) == 1),
                    y_min=y_min,
                    y_max=y_max,
                    indicators=ep_indicators,
                )
                composed.save(temp_path / f"frame-{i:06d}.png")

            encode_video_frames(
                imgs_dir=temp_path,
                video_path=dst_video_path,
                fps=fps,
                vcodec=vcodec,
                overwrite=True,
            )
        return dst_video_path

    composed_frames: list[Image.Image] = []
    for i in range(n_frames):
        composed = _draw_overlay(
            frame=decoded_frames[i],
            values=ep_values,
            current_step=i,
            advantage_t=float(ep_advantages[i]),
            acp_t=int(ep_indicators[i]),
            highlight_current_point=(int(ep_indicators[i]) == 1),
            y_min=y_min,
            y_max=y_max,
            indicators=ep_indicators,
        )
        composed_frames.append(composed)

    _encode_pil_to_video(composed_frames, dst_video_path, fps, vcodec)
    return dst_video_path


def _export_overlay_videos(
    dataset: LeRobotDataset,
    value_field: str,
    advantage_field: str,
    indicator_field: str,
    viz_episodes: str,
    video_key: str | None,
    output_dir: Path,
    overwrite: bool,
    vcodec: str,
    frame_storage_mode: str = "memory",
) -> list[Path]:
    selected_video_key = _select_video_key(dataset.meta.camera_keys, video_key)

    raw_dataset = dataset.hf_dataset.with_format(None)
    column_names = set(raw_dataset.column_names)

    if value_field not in column_names:
        raise KeyError(f"Missing value field '{value_field}' in dataset.")

    values_all = _to_1d_float(raw_dataset[value_field])
    if advantage_field in column_names:
        advantages_all = _to_1d_float(raw_dataset[advantage_field])
    else:
        advantages_all = np.zeros_like(values_all, dtype=np.float32)

    if indicator_field in column_names:
        indicators_all = _to_1d_int(raw_dataset[indicator_field])
    else:
        indicators_all = np.zeros(values_all.shape[0], dtype=np.int64)
    episode_indices_all = np.asarray(raw_dataset["episode_index"], dtype=np.int64).reshape(-1)
    frame_indices_all = np.asarray(raw_dataset["frame_index"], dtype=np.int64).reshape(-1)
    if "timestamp" in column_names:
        timestamps_all = np.asarray(raw_dataset["timestamp"], dtype=np.float64).reshape(-1)
    else:
        timestamps_all = frame_indices_all.astype(np.float64) / float(dataset.fps)

    if dataset.episodes is not None:
        available_episodes = sorted(dataset.episodes)
    else:
        available_episodes = list(range(dataset.meta.total_episodes))

    if viz_episodes.strip().lower() == "all":
        episodes = available_episodes
    else:
        requested = _parse_episodes_arg(viz_episodes, dataset.meta.total_episodes)
        episodes = [ep for ep in requested if ep in set(available_episodes)]

    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    fps = int(dataset.fps)
    for ep in episodes:
        ep_positions = np.flatnonzero(episode_indices_all == ep)
        if ep_positions.shape[0] > 1:
            ep_frame_indices = frame_indices_all[ep_positions]
            if bool(np.any(np.diff(ep_frame_indices) < 0)):
                ep_positions = ep_positions[np.argsort(ep_frame_indices, kind="stable")]

        ep_values = values_all[ep_positions]
        if ep_values.shape[0] == 0:
            continue

        ep_timestamps = timestamps_all[ep_positions]
        from_ts, to_ts = _get_episode_video_time_bounds(dataset, ep, selected_video_key)
        ep_video_timestamps = from_ts + ep_timestamps
        if to_ts is not None:
            ep_video_timestamps = np.minimum(ep_video_timestamps, to_ts)

        dst_path = _build_output_video_path(output_dir, dataset.repo_id, selected_video_key, ep)
        if dst_path.exists() and not overwrite:
            continue

        src_path = Path(dataset.root) / dataset.meta.get_video_file_path(ep, selected_video_key)
        tasks.append(
            (
                src_path,
                dst_path,
                ep_values,
                advantages_all[ep_positions],
                indicators_all[ep_positions],
                ep_video_timestamps,
            )
        )

    written_paths: list[Path] = []
    for src, dst, vals, advs, inds, ts in tqdm(
        tasks, total=len(tasks), desc="Export value overlay videos", leave=False
    ):
        written_paths.append(
            _export_single_episode(
                src,
                dst,
                vals,
                advs,
                inds,
                ts,
                fps,
                vcodec,
                float(getattr(dataset, "tolerance_s", 1e-4)),
                getattr(dataset, "video_backend", None),
                frame_storage_mode,
                output_dir,
            )
        )

    return written_paths
