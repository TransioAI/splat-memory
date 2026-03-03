"""Extract keyframes from video files or load from image directories."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".bmp", ".tiff")


def extract_keyframes_from_video(
    video_path: str | Path,
    every_n_frames: int = 30,
    max_frames: int = 40,
) -> list[tuple[int, Image.Image]]:
    """Extract keyframes from an MP4/MOV video file.

    Parameters
    ----------
    video_path:
        Path to the video file.
    every_n_frames:
        Extract one frame every N frames.
    max_frames:
        Maximum number of keyframes to extract.

    Returns
    -------
    list[tuple[int, Image.Image]]
        List of (frame_index, PIL_image) tuples.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Adjust stride if we'd exceed max_frames
    candidate_count = total_frames // every_n_frames
    if candidate_count > max_frames:
        every_n_frames = max(1, total_frames // max_frames)

    logger.info(
        "Video: %s — %d frames, %.1f fps, extracting every %d frames (max %d)",
        video_path.name, total_frames, fps, every_n_frames, max_frames,
    )

    keyframes: list[tuple[int, Image.Image]] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n_frames == 0:
            # BGR -> RGB -> PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            keyframes.append((frame_idx, pil_img))

            if len(keyframes) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    logger.info("Extracted %d keyframes from %s", len(keyframes), video_path.name)
    return keyframes


def load_keyframes_from_directory(
    directory: str | Path,
    max_frames: int = 40,
    extensions: tuple[str, ...] = IMAGE_EXTENSIONS,
) -> list[tuple[int, Image.Image]]:
    """Load images from a directory as keyframes.

    Parameters
    ----------
    directory:
        Path to directory containing image files.
    max_frames:
        Maximum number of images to load (evenly spaced if exceeds).
    extensions:
        File extensions to include.

    Returns
    -------
    list[tuple[int, Image.Image]]
        List of (index, PIL_image) tuples, sorted by filename.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Register HEIC support
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass

    # Collect and sort image files
    image_files = sorted(
        f for f in directory.iterdir()
        if f.suffix.lower() in extensions
    )

    if not image_files:
        raise ValueError(f"No image files found in {directory}")

    # Subsample evenly if exceeding max_frames
    if len(image_files) > max_frames:
        indices = np.linspace(0, len(image_files) - 1, max_frames, dtype=int)
        image_files = [image_files[i] for i in indices]

    logger.info("Loading %d images from %s", len(image_files), directory)

    keyframes: list[tuple[int, Image.Image]] = []
    for idx, filepath in enumerate(image_files):
        img = Image.open(filepath)
        img = img.convert("RGB")
        keyframes.append((idx, img))

    return keyframes


def save_keyframes_to_temp(
    keyframes: list[tuple[int, Image.Image]],
    cache_dir: str | Path,
) -> list[str]:
    """Save keyframes as JPEG files for MASt3R input.

    MASt3R's load_images() expects file paths, not PIL Images.

    Returns
    -------
    list[str]
        Ordered list of file paths to saved keyframe images.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    for frame_idx, img in keyframes:
        filename = f"frame_{frame_idx:05d}.jpg"
        filepath = cache_dir / filename
        img.save(str(filepath), "JPEG", quality=95)
        paths.append(str(filepath))

    logger.info("Saved %d keyframes to %s", len(paths), cache_dir)
    return paths
