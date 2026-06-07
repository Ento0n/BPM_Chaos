from pathlib import Path
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

DATASET_DIR = Path("/Users/antonspannagl/PythonProjects/BPM_Chaos/data/synthetic_bw_256")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

def threshold_bw(img: Image.Image, threshold: int = 128) -> Image.Image:
    return img.convert("L").point(lambda p: 255 if p > threshold else 0).convert("RGB")

def generate_candidate(seed: int, size: int) -> Image.Image:
    rng = np.random.default_rng(seed)

    bg = int(rng.choice([0, 255]))
    fg = 255 - bg

    img = Image.new("L", (size, size), bg)
    draw = ImageDraw.Draw(img)

    mode = int(rng.integers(0, 6))

    if mode == 0:
        # Random geometric shapes
        count = int(rng.integers(12, 40))
        for _ in range(count):
            x1 = int(rng.integers(0, size))
            y1 = int(rng.integers(0, size))
            w = int(rng.integers(size // 20, size // 3))
            h = int(rng.integers(size // 20, size // 3))
            x2 = min(size, x1 + w)
            y2 = min(size, y1 + h)

            if rng.random() < 0.5:
                draw.rectangle([x1, y1, x2, y2], outline=fg, width=int(rng.integers(1, 6)))
            else:
                draw.ellipse([x1, y1, x2, y2], outline=fg, width=int(rng.integers(1, 6)))

    elif mode == 1:
        # Line field
        count = int(rng.integers(40, 140))
        for _ in range(count):
            x1 = float(rng.integers(-size // 2, size + size // 2))
            y1 = float(rng.integers(-size // 2, size + size // 2))
            angle = float(rng.uniform(0, math.pi))
            length = float(rng.uniform(size // 5, size))
            x2 = x1 + math.cos(angle) * length
            y2 = y1 + math.sin(angle) * length
            draw.line([x1, y1, x2, y2], fill=fg, width=int(rng.integers(1, 5)))

    elif mode == 2:
        # Blocky thresholded noise
        grid = int(rng.choice([8, 12, 16, 24, 32, 48]))
        arr = rng.random((grid, grid))
        arr = (arr > rng.uniform(0.35, 0.65)).astype(np.uint8) * 255
        noise_img = Image.fromarray(arr, mode="L").resize((size, size), Image.Resampling.NEAREST)

        if rng.random() < 0.5:
            noise_img = noise_img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.5, 2.5))))
            noise_img = threshold_bw(noise_img, int(rng.integers(80, 180))).convert("L")

        img = noise_img if bg == 0 else Image.fromarray(255 - np.asarray(noise_img), mode="L")

    elif mode == 3:
        # Concentric rings
        cx = int(rng.integers(size // 5, 4 * size // 5))
        cy = int(rng.integers(size // 5, 4 * size // 5))
        step = int(rng.integers(8, 24))
        width = int(rng.integers(2, 8))

        for r in range(step, size, step):
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=fg, width=width)

    elif mode == 4:
        # Mirrored random polygon strokes
        count = int(rng.integers(8, 24))
        for _ in range(count):
            points = []
            for _ in range(int(rng.integers(3, 8))):
                x = int(rng.integers(0, size // 2))
                y = int(rng.integers(0, size))
                points.append((x, y))

            mirrored = [(size - x, y) for x, y in reversed(points)]
            shape = points + mirrored

            if rng.random() < 0.5:
                draw.polygon(shape, outline=fg)
            else:
                draw.line(shape + [shape[0]], fill=fg, width=int(rng.integers(1, 5)))

    else:
        # Stripes / grid
        spacing = int(rng.integers(8, 32))
        width = int(rng.integers(2, 10))
        angle_mode = int(rng.integers(0, 3))

        for offset in range(-size, size * 2, spacing):
            if angle_mode == 0:
                draw.line([offset, 0, offset, size], fill=fg, width=width)
            elif angle_mode == 1:
                draw.line([0, offset, size, offset], fill=fg, width=width)
            else:
                draw.line([offset, 0, offset + size, size], fill=fg, width=width)

        if rng.random() < 0.5:
            for offset in range(-size, size * 2, spacing * 2):
                draw.line([0, offset, size, offset + size], fill=fg, width=max(1, width // 2))

    # Occasional inversion
    if rng.random() < 0.25:
        img = Image.fromarray(255 - np.asarray(img), mode="L")

    return threshold_bw(img, int(rng.integers(96, 160)))

if __name__ == "__main__":
    for i in range(50_000):
        img = generate_candidate(seed=i, size=256)
        img.save(DATASET_DIR / f"{i:06d}.png")