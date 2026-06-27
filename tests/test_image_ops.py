"""Preprocessing + augmentation sanity checks (sizes, RGB, deterministic JPEG aug)."""
from PIL import Image

from lib import image_ops


def _img(w=64, h=48, color=(120, 30, 200)):
    return Image.new("RGB", (w, h), color)


def test_scale_to_square():
    out = image_ops.scale_to(_img(64, 48), 32)
    assert out.size == (32, 32)


def test_center_crop_large_image():
    out = image_ops.center_crop(_img(100, 80), 32)
    assert out.size == (32, 32)


def test_center_crop_upscales_small_image():
    out = image_ops.center_crop(_img(10, 10), 32)
    assert out.size == (32, 32)


def test_load_rgb_forces_rgb(tmp_path):
    p = tmp_path / "gray.png"
    Image.new("L", (16, 16), 128).save(p)
    img = image_ops.load_rgb(str(p))
    assert img.mode == "RGB"


def test_jpeg_augmenter_is_per_path_deterministic():
    aug = image_ops.make_jpeg_augmenter((30, 90), seed=42)
    img = _img(128, 128)
    a = aug(img, "some/path.png")
    b = aug(img, "some/path.png")
    # Same path -> same quality -> byte-identical output (reproducible).
    assert a.tobytes() == b.tobytes()


def test_jpeg_augmenter_preserves_shape_and_mode():
    aug = image_ops.make_jpeg_augmenter((30, 90), seed=7)
    img = _img(96, 96)
    out = aug(img, "other/path.png")
    assert out.size == img.size and out.mode == "RGB"
