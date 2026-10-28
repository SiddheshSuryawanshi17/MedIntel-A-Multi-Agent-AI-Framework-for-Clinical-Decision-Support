"""Small SVG charts for the evaluation report: reliability diagrams and grouped bars. No plotting dependency.

Colours are the validated categorical slots (blue, orange, aqua) in fixed order; every series is also labelled in a
legend and every figure has a table beside it in EVALUATION.md, so no reading depends on colour alone."""

from html import escape

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e0"
W, H = 640, 420
LEFT, RIGHT, TOP, BOTTOM = 64, 24, 80, 64
MIN_BIN = 3


def _frame(title: str, xlabel: str, ylabel: str, body: list[str], legend: list[tuple[str, str]]) -> str:
    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
        f'aria-label="{escape(title)}" font-family="system-ui, -apple-system, Segoe UI, Arial, sans-serif">',
        f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
        f'<text x="{LEFT}" y="26" font-size="15" font-weight="600" fill="{TEXT}">{escape(title)}</text>',
        *body,
        f'<text x="{(LEFT + W - RIGHT) / 2}" y="{H - 16}" font-size="12" fill="{TEXT_2}" '
        f'text-anchor="middle">{escape(xlabel)}</text>',
        f'<text transform="translate(16 {(TOP + H - BOTTOM) / 2}) rotate(-90)" font-size="12" fill="{TEXT_2}" '
        f'text-anchor="middle">{escape(ylabel)}</text>',
    ]
    x = LEFT
    for color, name in legend:
        items.append(f'<rect x="{x}" y="36" width="10" height="10" rx="2" fill="{color}"/>')
        items.append(f'<text x="{x + 15}" y="45" font-size="12" fill="{TEXT_2}">{escape(name)}</text>')
        x += 24 + 7 * len(name)
    items.append("</svg>")
    return "\n".join(items)


def _axes(ticks: list[float], scale_x: object = None) -> list[str]:
    body: list[str] = []
    plot_w, plot_h = W - LEFT - RIGHT, H - TOP - BOTTOM
    for t in ticks:
        y = TOP + plot_h * (1 - t)
        body.append(
            f'<line x1="{LEFT}" y1="{y:.1f}" x2="{LEFT + plot_w}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        body.append(
            f'<text x="{LEFT - 8}" y="{y + 4:.1f}" font-size="11" fill="{TEXT_2}" text-anchor="end">{t:.1f}</text>'
        )
    return body


def reliability(
    title: str, series: list[tuple[str, list[dict[str, float]]]], xlabel: str, ylabel: str, x_min: float = 0.0
) -> str:
    """Observed rate against mean predicted value per bin; a dashed diagonal marks perfect calibration. Marker area
    grows with the number of cases in the bin; bins with fewer than MIN_BIN cases are hollow and left off the line."""
    plot_w, plot_h = W - LEFT - RIGHT, H - TOP - BOTTOM
    span = 1.0 - x_min

    def sx(v: float) -> float:
        return LEFT + plot_w * (v - x_min) / span

    body = _axes([i / 5 for i in range(6)])
    for i in range(int(round(span * 10)) + 1):
        t = x_min + i / 10
        if i % 2 == 0 or span <= 0.6:
            body.append(
                f'<text x="{sx(t):.1f}" y="{TOP + plot_h + 18}" font-size="11" fill="{TEXT_2}" '
                f'text-anchor="middle">{t:.1f}</text>'
            )
    body.append(
        f'<line x1="{sx(x_min):.1f}" y1="{TOP + plot_h * (1 - x_min):.1f}" x2="{sx(1.0):.1f}" y2="{TOP}" '
        f'stroke="{TEXT_2}" stroke-width="1" stroke-dasharray="4 4"/>'
    )
    legend: list[tuple[str, str]] = []
    for i, (name, rows) in enumerate(series):
        color = SERIES[i % len(SERIES)]
        legend.append((color, name))
        points = [
            (sx(float(r["mean_predicted"])), TOP + plot_h * (1 - float(r["observed"])), int(r["n"]))
            for r in rows
            if float(r["mean_predicted"]) >= x_min
        ]
        solid = [(x, y) for x, y, n in points if n >= MIN_BIN]
        if len(solid) > 1:
            path = " ".join(f"{'M' if j == 0 else 'L'}{x:.1f},{y:.1f}" for j, (x, y) in enumerate(solid))
            body.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y, n in points:
            radius = 4 + min(6, n**0.5)
            fill = color if n >= MIN_BIN else SURFACE
            body.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" stroke="{color}" '
                f'stroke-width="2"><title>{escape(name)}: n = {n}</title></circle>'
            )
    return _frame(title, xlabel, ylabel, body, legend)


def bars(title: str, categories: list[str], series: list[tuple[str, list[float | None]]], ylabel: str) -> str:
    """Grouped bars on a 0 to 1 scale with a value label on each bar."""
    plot_w, plot_h = W - LEFT - RIGHT, H - TOP - BOTTOM
    body = _axes([i / 5 for i in range(6)])
    group = plot_w / max(1, len(categories))
    bar = min(28.0, (group - 16) / max(1, len(series)))
    legend: list[tuple[str, str]] = []
    for c, category in enumerate(categories):
        x0 = LEFT + c * group + (group - bar * len(series)) / 2
        body.append(
            f'<text x="{LEFT + c * group + group / 2:.1f}" y="{TOP + plot_h + 18}" font-size="11" fill="{TEXT_2}" '
            f'text-anchor="middle">{escape(category)}</text>'
        )
        for s, (name, values) in enumerate(series):
            color = SERIES[s % len(SERIES)]
            if c == 0:
                legend.append((color, name))
            value = values[c]
            if value is None:
                continue
            height = plot_h * max(0.0, min(1.0, value))
            x = x0 + s * bar
            y = TOP + plot_h - height
            body.append(
                f'<rect x="{x + 1:.1f}" y="{y:.1f}" width="{bar - 2:.1f}" height="{height:.1f}" rx="3" fill="{color}">'
                f"<title>{escape(name)}, {escape(category)}: {value:.3f}</title></rect>"
            )
            body.append(
                f'<text x="{x + bar / 2:.1f}" y="{y - 5:.1f}" font-size="10.5" fill="{TEXT}" '
                f'text-anchor="middle">{value:.2f}</text>'
            )
    return _frame(title, "", ylabel, body, legend)
