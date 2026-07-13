from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "centered_geometric_bw_256"
DEFAULT_COUNT = 50_000
DEFAULT_IMAGE_SIZE = 256
SUPERSAMPLE = 2

Point = tuple[float, float]


def threshold_bw(img: Image.Image, threshold: int = 128) -> Image.Image:
    return img.convert("L").point(lambda p: 255 if p > threshold else 0).convert("RGB")


def polar(center: Point, radius: float, angle: float) -> Point:
    return (
        center[0] + math.cos(angle) * radius,
        center[1] + math.sin(angle) * radius,
    )


def regular_polygon(center: Point, radius: float, sides: int, rotation: float) -> list[Point]:
    return [
        polar(center, radius, rotation + 2.0 * math.pi * i / sides)
        for i in range(sides)
    ]


def star_polygon(
    center: Point,
    outer_radius: float,
    inner_radius: float,
    points: int,
    rotation: float,
) -> list[Point]:
    vertices = []
    for i in range(points * 2):
        radius = outer_radius if i % 2 == 0 else inner_radius
        vertices.append(polar(center, radius, rotation + math.pi * i / points))
    return vertices


def rectangle_polygon(center: Point, width: float, height: float, rotation: float) -> list[Point]:
    cos_a = math.cos(rotation)
    sin_a = math.sin(rotation)
    vertices = []
    for x, y in [
        (-width / 2.0, -height / 2.0),
        (width / 2.0, -height / 2.0),
        (width / 2.0, height / 2.0),
        (-width / 2.0, height / 2.0),
    ]:
        vertices.append(
            (
                center[0] + x * cos_a - y * sin_a,
                center[1] + x * sin_a + y * cos_a,
            )
        )
    return vertices


def ellipse_polygon(
    center: Point,
    radius_x: float,
    radius_y: float,
    rotation: float,
    segments: int = 72,
) -> list[Point]:
    cos_a = math.cos(rotation)
    sin_a = math.sin(rotation)
    vertices = []
    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        x = math.cos(angle) * radius_x
        y = math.sin(angle) * radius_y
        vertices.append(
            (
                center[0] + x * cos_a - y * sin_a,
                center[1] + x * sin_a + y * cos_a,
            )
        )
    return vertices


def draw_closed_line(
    draw: ImageDraw.ImageDraw,
    points: list[Point],
    color: int,
    width: int,
) -> None:
    draw.line(points + [points[0]], fill=color, width=max(1, width))


def draw_shape(
    draw: ImageDraw.ImageDraw,
    name: str,
    center: Point,
    radius: float,
    rotation: float,
    color: int,
    width: int,
    filled: bool,
) -> None:
    if name == "circle":
        points = ellipse_polygon(center, radius, radius, rotation)
    elif name == "ellipse":
        points = ellipse_polygon(center, radius * 1.35, radius * 0.65, rotation)
    elif name == "triangle":
        points = regular_polygon(center, radius, 3, rotation)
    elif name == "square":
        points = regular_polygon(center, radius, 4, rotation + math.pi / 4.0)
    elif name == "rectangle":
        points = rectangle_polygon(center, radius * 1.75, radius * 0.85, rotation)
    elif name == "diamond":
        points = regular_polygon(center, radius, 4, rotation)
    elif name == "pentagon":
        points = regular_polygon(center, radius, 5, rotation)
    elif name == "hexagon":
        points = regular_polygon(center, radius, 6, rotation)
    elif name == "octagon":
        points = regular_polygon(center, radius, 8, rotation)
    else:
        points = star_polygon(center, radius, radius * 0.43, 5, rotation)

    if filled:
        draw.polygon(points, fill=color)
    else:
        draw_closed_line(draw, points, color, width)


def draw_concentric_forms(
    draw: ImageDraw.ImageDraw,
    rng: np.random.Generator,
    center: Point,
    size: int,
    color: int,
) -> None:
    shapes = [
        "circle",
        "ellipse",
        "triangle",
        "square",
        "rectangle",
        "diamond",
        "pentagon",
        "hexagon",
        "octagon",
        "star",
    ]
    rings = int(rng.integers(5, 18))
    max_radius = size * float(rng.uniform(0.31, 0.48))
    base_rotation = float(rng.uniform(0.0, 2.0 * math.pi))
    line_width = int(rng.integers(size // 140 + 1, size // 45 + 2))

    for index in range(rings):
        progress = (index + 1) / rings
        wave = 1.0 + 0.14 * math.sin(progress * math.tau * float(rng.integers(1, 5)))
        radius = max_radius * progress * wave
        shape = str(rng.choice(shapes))
        rotation = base_rotation + index * float(rng.uniform(0.18, 0.74))
        filled = bool(rng.random() < 0.18 and index < rings - 1)
        draw_shape(draw, shape, center, radius, rotation, color, line_width, filled)

    if rng.random() < 0.8:
        draw_shape(
            draw,
            str(rng.choice(shapes)),
            center,
            size * float(rng.uniform(0.025, 0.075)),
            base_rotation,
            color,
            max(1, line_width),
            bool(rng.random() < 0.5),
        )


def draw_orbital_wave_forms(
    draw: ImageDraw.ImageDraw,
    rng: np.random.Generator,
    center: Point,
    size: int,
    color: int,
) -> None:
    shapes = [
        "circle",
        "triangle",
        "square",
        "rectangle",
        "diamond",
        "pentagon",
        "hexagon",
        "star",
    ]
    ring_count = int(rng.integers(2, 7))
    max_orbit = size * 0.43
    phase = float(rng.uniform(0.0, math.tau))
    wave_frequency = int(rng.integers(2, 9))
    base_width = int(rng.integers(size // 160 + 1, size // 55 + 2))

    for ring_index in range(ring_count):
        count = int(rng.choice([3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 32]))
        orbit = max_orbit * (ring_index + 1) / (ring_count + 1)
        shape_radius = size * float(rng.uniform(0.018, 0.065)) * (1.0 - ring_index * 0.055)
        amplitude = orbit * float(rng.uniform(0.03, 0.18))
        shape = str(rng.choice(shapes))
        ring_phase = phase + float(rng.uniform(-0.55, 0.55))

        for item_index in range(count):
            angle = ring_phase + math.tau * item_index / count
            wavy_orbit = orbit + amplitude * math.sin(angle * wave_frequency + phase)
            shape_center = polar(center, wavy_orbit, angle)
            rotation = angle + math.pi / 2.0
            if rng.random() < 0.35:
                rotation += float(rng.uniform(-math.pi / 3.0, math.pi / 3.0))
            draw_shape(
                draw,
                shape,
                shape_center,
                max(2.0, shape_radius),
                rotation,
                color,
                base_width,
                bool(rng.random() < 0.22),
            )

        if rng.random() < 0.55:
            ring_points = [
                polar(
                    center,
                    orbit + amplitude * math.sin((ring_phase + math.tau * i / 180) * wave_frequency + phase),
                    ring_phase + math.tau * i / 180,
                )
                for i in range(180)
            ]
            draw_closed_line(draw, ring_points, color, max(1, base_width // 2))


def draw_radial_bars_and_spokes(
    draw: ImageDraw.ImageDraw,
    rng: np.random.Generator,
    center: Point,
    size: int,
    color: int,
) -> None:
    count = int(rng.choice([8, 10, 12, 16, 18, 24, 28, 32, 40, 48]))
    phase = float(rng.uniform(0.0, math.tau))
    wave_frequency = int(rng.integers(2, 8))
    inner = size * float(rng.uniform(0.035, 0.16))
    outer_base = size * float(rng.uniform(0.28, 0.46))
    bar_width = size * float(rng.uniform(0.012, 0.035))
    line_width = int(rng.integers(size // 150 + 1, size // 70 + 2))

    for index in range(count):
        angle = phase + math.tau * index / count
        outer = outer_base * (1.0 + 0.22 * math.sin(angle * wave_frequency + phase))
        length = max(size * 0.04, outer - inner)
        bar_center = polar(center, inner + length / 2.0, angle)
        points = rectangle_polygon(bar_center, length, bar_width, angle)
        if rng.random() < 0.55:
            draw.polygon(points, fill=color)
        else:
            draw_closed_line(draw, points, color, line_width)

    if rng.random() < 0.9:
        for radius in np.linspace(inner, outer_base, int(rng.integers(2, 6))):
            points = ellipse_polygon(center, radius, radius, 0.0, segments=160)
            draw_closed_line(draw, points, color, max(1, line_width))

    if rng.random() < 0.7:
        shape_count = count if count <= 24 else count // 2
        shape = str(rng.choice(["circle", "triangle", "diamond", "hexagon", "star"]))
        for index in range(shape_count):
            angle = phase + math.tau * index / shape_count
            orbit = outer_base * float(rng.uniform(0.62, 0.96))
            draw_shape(
                draw,
                shape,
                polar(center, orbit, angle),
                size * float(rng.uniform(0.018, 0.045)),
                angle,
                color,
                line_width,
                bool(rng.random() < 0.25),
            )


def draw_rotated_polygon_mandala(
    draw: ImageDraw.ImageDraw,
    rng: np.random.Generator,
    center: Point,
    size: int,
    color: int,
) -> None:
    symmetry = int(rng.choice([3, 4, 5, 6, 8, 10, 12, 16]))
    layers = int(rng.integers(3, 9))
    phase = float(rng.uniform(0.0, math.tau))
    line_width = int(rng.integers(size // 160 + 1, size // 45 + 2))

    for layer in range(layers):
        radius = size * (0.07 + 0.38 * (layer + 1) / layers)
        shape_radius = size * float(rng.uniform(0.018, 0.07))
        shape = str(rng.choice(["triangle", "square", "rectangle", "diamond", "pentagon", "hexagon", "octagon"]))
        layer_phase = phase + layer * math.pi / symmetry
        for index in range(symmetry):
            angle = layer_phase + math.tau * index / symmetry
            wave_radius = radius * (1.0 + 0.08 * math.sin(angle * layers + phase))
            draw_shape(
                draw,
                shape,
                polar(center, wave_radius, angle),
                shape_radius,
                angle + layer * 0.38,
                color,
                line_width,
                bool(rng.random() < 0.16),
            )

    for sides in rng.choice([3, 4, 5, 6, 8], size=int(rng.integers(1, 4)), replace=False):
        radius = size * float(rng.uniform(0.12, 0.45))
        points = regular_polygon(center, radius, int(sides), phase + radius * 0.01)
        draw_closed_line(draw, points, color, max(1, line_width))


def draw_centered_wave_grid(
    draw: ImageDraw.ImageDraw,
    rng: np.random.Generator,
    center: Point,
    size: int,
    color: int,
) -> None:
    arms = int(rng.choice([4, 6, 8, 10, 12, 16]))
    steps = int(rng.integers(4, 12))
    phase = float(rng.uniform(0.0, math.tau))
    shape = str(rng.choice(["circle", "ellipse", "triangle", "square", "rectangle", "diamond"]))
    line_width = int(rng.integers(size // 170 + 1, size // 65 + 2))

    for arm in range(arms):
        arm_angle = phase + math.tau * arm / arms
        for step in range(1, steps + 1):
            progress = step / steps
            side_wave = math.sin(progress * math.tau * float(rng.integers(1, 4)) + arm_angle)
            radius = size * (0.045 + 0.405 * progress)
            offset_angle = arm_angle + side_wave * float(rng.uniform(0.02, 0.18))
            shape_center = polar(center, radius, offset_angle)
            shape_radius = size * float(rng.uniform(0.012, 0.042)) * (1.1 - progress * 0.35)
            draw_shape(
                draw,
                shape,
                shape_center,
                max(2.0, shape_radius),
                arm_angle + progress * math.tau,
                color,
                line_width,
                bool(rng.random() < 0.2),
            )

    if rng.random() < 0.75:
        for arm in range(arms):
            angle = phase + math.tau * arm / arms
            endpoint = polar(center, size * float(rng.uniform(0.32, 0.47)), angle)
            draw.line([center, endpoint], fill=color, width=max(1, line_width // 2))


def draw_lissajous_orbit(
    draw: ImageDraw.ImageDraw,
    rng: np.random.Generator,
    center: Point,
    size: int,
    color: int,
) -> None:
    samples = int(rng.integers(80, 220))
    a = int(rng.integers(2, 7))
    b = int(rng.integers(3, 9))
    phase = float(rng.uniform(0.0, math.tau))
    radius = size * float(rng.uniform(0.24, 0.43))
    line_width = int(rng.integers(size // 170 + 1, size // 65 + 2))
    points = []

    for index in range(samples):
        t = math.tau * index / samples
        x = math.sin(a * t + phase) * radius
        y = math.sin(b * t) * radius
        points.append((center[0] + x, center[1] + y))

    draw_closed_line(draw, points, color, line_width)

    shape_count = int(rng.choice([6, 8, 10, 12, 16, 20, 24]))
    shape = str(rng.choice(["circle", "triangle", "square", "diamond", "pentagon", "hexagon", "star"]))
    for index in range(shape_count):
        t = math.tau * index / shape_count
        x = math.sin(a * t + phase) * radius
        y = math.sin(b * t) * radius
        angle = math.atan2(y, x)
        draw_shape(
            draw,
            shape,
            (center[0] + x, center[1] + y),
            size * float(rng.uniform(0.018, 0.05)),
            angle,
            color,
            line_width,
            bool(rng.random() < 0.2),
        )


def draw_center_anchor(
    draw: ImageDraw.ImageDraw,
    rng: np.random.Generator,
    center: Point,
    size: int,
    color: int,
) -> None:
    shape = str(rng.choice(["circle", "triangle", "square", "diamond", "hexagon", "star"]))
    radius = size * float(rng.uniform(0.015, 0.055))
    width = int(rng.integers(size // 180 + 1, size // 70 + 2))
    draw_shape(
        draw,
        shape,
        center,
        radius,
        float(rng.uniform(0.0, math.tau)),
        color,
        width,
        bool(rng.random() < 0.55),
    )


def generate_candidate(seed: int, size: int = DEFAULT_IMAGE_SIZE) -> Image.Image:
    rng = np.random.default_rng(seed)
    canvas_size = size * SUPERSAMPLE
    center = ((canvas_size - 1) / 2.0, (canvas_size - 1) / 2.0)

    bg = int(rng.choice([0, 255]))
    fg = 255 - bg
    img = Image.new("L", (canvas_size, canvas_size), bg)
    draw = ImageDraw.Draw(img)

    mode = int(rng.integers(0, 6))
    if mode == 0:
        draw_concentric_forms(draw, rng, center, canvas_size, fg)
    elif mode == 1:
        draw_orbital_wave_forms(draw, rng, center, canvas_size, fg)
    elif mode == 2:
        draw_radial_bars_and_spokes(draw, rng, center, canvas_size, fg)
    elif mode == 3:
        draw_rotated_polygon_mandala(draw, rng, center, canvas_size, fg)
    elif mode == 4:
        draw_centered_wave_grid(draw, rng, center, canvas_size, fg)
    else:
        draw_lissajous_orbit(draw, rng, center, canvas_size, fg)

    if rng.random() < 0.35:
        extra_drawers = [
            draw_concentric_forms,
            draw_orbital_wave_forms,
            draw_radial_bars_and_spokes,
            draw_rotated_polygon_mandala,
            draw_centered_wave_grid,
        ]
        extra_drawer = extra_drawers[int(rng.integers(0, len(extra_drawers)))]
        extra_drawer(draw, rng, center, canvas_size, fg)

    draw_center_anchor(draw, rng, center, canvas_size, fg)

    if rng.random() < 0.22:
        img = Image.fromarray(255 - np.asarray(img), mode="L")

    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return threshold_bw(img, int(rng.integers(96, 160)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a centered black-and-white geometric training dataset."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete existing PNG files in the output directory before generating.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be greater than 0.")
    if args.image_size <= 0:
        raise ValueError("--image-size must be greater than 0.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.clear:
        for path in args.output_dir.glob("*.png"):
            path.unlink()

    for index in range(args.count):
        img = generate_candidate(seed=args.seed_offset + index, size=args.image_size)
        img.save(args.output_dir / f"{index:06d}.png")
        if (index + 1) % 1000 == 0 or index + 1 == args.count:
            print(f"Generated {index + 1}/{args.count} images in {args.output_dir}")


if __name__ == "__main__":
    main()
