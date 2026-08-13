import math
import os
import re
import uuid
from pathlib import Path

from PIL import Image


PROVIDER_MAX_MEGAPIXELS = {"NanoGPT": 1.0}
IMAGE_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
_WINDOWS_ILLEGAL_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_FILENAME = re.compile(
    r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)", re.IGNORECASE
)


def scale_for_provider(image, provider):
    limit = PROVIDER_MAX_MEGAPIXELS.get(str(provider))
    if limit is None:
        return image

    width, height = image.size
    maximum_pixels = int(limit * 1_000_000)
    if width * height <= maximum_pixels:
        return image

    scale = math.sqrt(maximum_pixels / (width * height))
    target = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(target, Image.Resampling.LANCZOS)


def _normalized_path(value):
    text = os.path.expandvars(os.path.expanduser(str(value).strip().strip('"')))
    if not text:
        raise ValueError("Path cannot be empty")
    return Path(text).resolve(strict=False)


def _sidecar_stem(filename):
    name = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    path = Path(name)
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        name = path.stem

    name = _WINDOWS_ILLEGAL_FILENAME_CHARS.sub("_", name).rstrip(" .")
    if not name:
        raise ValueError("filename must contain at least one valid character")
    if _WINDOWS_RESERVED_FILENAME.match(name):
        name = f"_{name}"
    return name


class DatasetSidecarWriter:
    def write(self, text, filename, folder, existing_file="overwrite"):
        if existing_file not in {"overwrite", "skip"}:
            raise ValueError("existing_file must be 'overwrite' or 'skip'")

        destination = _normalized_path(folder)
        destination.mkdir(parents=True, exist_ok=True)
        sidecar = destination / f"{_sidecar_stem(filename)}.txt"
        if existing_file == "skip" and sidecar.exists():
            return {"status": "skipped", "path": str(sidecar)}

        temporary = sidecar.with_name(f".{sidecar.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(str(text), encoding="utf-8", newline="")
            os.replace(temporary, sidecar)
        finally:
            if temporary.exists():
                temporary.unlink()
        return {"status": "written", "path": str(sidecar)}
