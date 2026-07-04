"""
Image preprocessing and perturbation operations.

Three preprocessing strategies are provided (scaling vs cropping vs aspect-preserving
resize+crop); all write lossless PNG to avoid stacking JPEG artifacts. `scale_to` (squash)
DISTORTS non-square images and can turn the format/resolution confound into an aspect-ratio
confound (supervisor feedback); `resize_shortest_center_crop` is the aspect-preserving option
that avoids that. Robustness perturbations deliberately DO introduce controlled degradations
and are applied to held-out test images only.

ASCII-only; Python 3.9. Uses Pillow only (no GUI).
"""
import io
from typing import Tuple

from PIL import Image, ImageFilter


def load_rgb(path: str) -> Image.Image:
    """Open an image and force 3-channel RGB."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# --- Preprocessing strategies (GOLD concern #2) ------------------------------

def scale_to(img: Image.Image, size: int, resample=Image.BICUBIC) -> Image.Image:
    """Strategy A: squash/scale the whole image to size x size.

    Touches every pixel (interpolation artifacts) but preserves global content.
    """
    return img.resize((size, size), resample=resample)


def center_crop(img: Image.Image, size: int) -> Image.Image:
    """Strategy B: take a center crop of size x size.

    Preserves native pixel statistics (no interpolation) but loses surrounding content.
    If the image is smaller than `size`, it is first scaled up just enough so a crop fits.
    """
    width, height = img.size
    if width < size or height < size:
        scale = size / float(min(width, height))
        new_size = (max(size, int(round(width * scale))),
                    max(size, int(round(height * scale))))
        img = img.resize(new_size, resample=Image.BICUBIC)
        width, height = img.size
    left = (width - size) // 2
    top = (height - size) // 2
    return img.crop((left, top, left + size, top + size))


def resize_shortest_center_crop(img: Image.Image, size: int,
                                resample=Image.BICUBIC) -> Image.Image:
    """Strategy C (aspect-PRESERVING): resize so the SHORTER side == size, then center-crop
    size x size.

    Motivation (supervisor feedback, Dennis): `scale_to` squashes to size x size and therefore
    DISTORTS the aspect ratio of non-square images. Our fakes are square (512x512) so they
    downscale cleanly, but non-square reals (CelebA 178x218, London-DB) get stretched. That
    stretch is itself label-correlated, so a squash pipeline risks REPLACING the format/size
    confound with an aspect-distortion confound instead of removing it.

    This strategy scales uniformly (shorter side -> size, aspect kept) and then crops the
    centered size x size window, so NO image is stretched. A square input (e.g. our fakes) is
    simply downscaled - identical treatment to the reals' scale step - and its crop is a no-op,
    so real and fake go through the exact same uniform resample. Faces stay centered (unlike
    `center_crop` on a large native image, which can crop into a tiny non-face patch).
    """
    width, height = img.size
    scale = size / float(min(width, height))
    new_size = (max(size, int(round(width * scale))),
                max(size, int(round(height * scale))))
    img = img.resize(new_size, resample=resample)
    width, height = img.size
    left = (width - size) // 2
    top = (height - size) // 2
    return img.crop((left, top, left + size, top + size))


def save_png(img: Image.Image, path: str) -> None:
    """Write a lossless PNG."""
    img.save(path, format="PNG", optimize=False)


# --- Robustness perturbations (applied to test images only) ------------------

def jpeg_recompress(img: Image.Image, quality: int) -> Image.Image:
    """Round-trip the image through JPEG at the given quality, return decoded RGB."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


# --- Training-time augmentation (format/compression confound control) --------
# Source format correlates with label (reals CelebA + London-DB are JPEG; all fakes PNG).
# Re-encoding JPEG->PNG does NOT remove baked-in JPEG artifacts, so during TRAINING we push
# every image through a random JPEG quality. After that, "has been JPEG-compressed" no longer
# separates real from fake (Frank 2020 / Wang 2020).

class JpegAugmenter:
    """Picklable, per-path-deterministic JPEG augmentation callable.

    Applied as (img, path) -> img: pushes every image through a random JPEG quality seeded by
    a hash of `path`, so the same image always gets the same quality regardless of iteration
    order or skipped files (Frank2020 / Wang2020 confound control: "has been JPEG-compressed"
    no longer separates real from fake).

    Implemented as a top-level class (NOT a nested closure) so it pickles cleanly under
    Windows/torch-spawn DataLoader workers. The previous closure form
    (make_jpeg_augmenter.<locals>._aug) raised AttributeError when the Dataset+augmenter were
    pickled to spawn workers with num_workers>0 on win32.
    """

    def __init__(self, quality_range=(30, 100), seed: int = 42):
        self.qmin = int(quality_range[0])
        self.qmax = int(quality_range[1])
        self.seed = int(seed)

    def __call__(self, img: Image.Image, path: str = "") -> Image.Image:
        import random as _random
        import zlib
        h = zlib.crc32(str(path).encode("utf-8")) & 0xFFFFFFFF
        rng = _random.Random((self.seed << 32) ^ h)
        return jpeg_recompress(img, rng.randint(self.qmin, self.qmax))

    # __getstate__/__setstate__ are trivial but explicit: the closure form failed to pickle,
    # so we make the round-trip explicit (qmin/qmax/seed are plain ints).
    def __getstate__(self):
        return {"qmin": self.qmin, "qmax": self.qmax, "seed": self.seed}

    def __setstate__(self, state):
        self.qmin = state["qmin"]
        self.qmax = state["qmax"]
        self.seed = state["seed"]


def make_jpeg_augmenter(quality_range=(30, 100), seed: int = 42):
    """Return a picklable callable(img, path) -> img that applies a per-path-deterministic
    random JPEG quality. Thin factory over JpegAugmenter (kept for call-site compatibility)."""
    return JpegAugmenter(quality_range, seed)


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    """Apply Gaussian blur with the given standard deviation (radius)."""
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def resize_roundtrip(img: Image.Image, factor: float) -> Image.Image:
    """Downscale by `factor` then upscale back to the original size (resampling artifacts)."""
    width, height = img.size
    small = img.resize((max(1, int(width * factor)), max(1, int(height * factor))),
                       resample=Image.BICUBIC)
    return small.resize((width, height), resample=Image.BICUBIC)


def sharpen(img: Image.Image, amount: float = 1.0) -> Image.Image:
    """Unsharp-mask style sharpening; `amount` scales the percent strength."""
    percent = int(150 * float(amount))
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=percent, threshold=3))


def image_size(path: str) -> Tuple[int, int]:
    """Return (width, height) without fully decoding pixels."""
    with Image.open(path) as img:
        return img.size
