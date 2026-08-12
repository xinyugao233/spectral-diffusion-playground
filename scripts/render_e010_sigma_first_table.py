"""Render the sigma-first E010 summary figure in PNG and PDF formats."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = REPOSITORY_ROOT / "figures" / "experiment_10"

INK = "#17212b"
MUTED = "#5d6875"
RULE = "#cbd3dc"
LOW = "#2563a6"
LOW_PALE = "#eaf2fb"
HIGH = "#b45309"
HIGH_PALE = "#fff1dc"
PAPER = "#ffffff"


def add_result_card(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    radius: str,
    geometry: str,
    selection: str,
    rows: list[tuple[str, str, str, bool]],
    accent: str,
    pale: str,
    result: str,
) -> None:
    """Add one aligned low- or high-frequency result card."""
    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.25,
        edgecolor=RULE,
        facecolor=PAPER,
        transform=ax.transAxes,
    )
    ax.add_patch(card)

    pad = 0.035 * width
    left = x + pad
    right = x + width - pad
    top = y + height - 0.045

    ax.text(
        left,
        top,
        label,
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        color=accent,
        va="top",
    )
    ax.text(
        right,
        top,
        radius,
        transform=ax.transAxes,
        fontsize=17,
        fontweight="bold",
        color=INK,
        va="top",
        ha="right",
    )
    ax.text(
        left,
        top - 0.055,
        geometry,
        transform=ax.transAxes,
        fontsize=11.5,
        color=MUTED,
        va="top",
    )
    ax.text(
        left,
        top - 0.095,
        selection,
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        color=INK,
        va="top",
    )

    table_top = top - 0.16
    table_bottom = y + 0.095
    condition_x = left
    sigma_x = x + width * 0.66
    rate_x = right

    ax.plot(
        [left, right],
        [table_top, table_top],
        color=INK,
        linewidth=1.1,
        transform=ax.transAxes,
    )
    header_y = table_top - 0.035
    ax.text(
        condition_x,
        header_y,
        "Condition",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=MUTED,
        va="center",
    )
    ax.text(
        sigma_x,
        header_y,
        "Noise level (sigma)",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=MUTED,
        va="center",
        ha="right",
    )
    ax.text(
        rate_x,
        header_y,
        "Final memorization",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        color=MUTED,
        va="center",
        ha="right",
    )

    body_top = table_top - 0.065
    row_height = (body_top - table_bottom) / len(rows)
    for index, (condition, sigma, rate, selected) in enumerate(rows):
        row_top = body_top - index * row_height
        row_bottom = row_top - row_height
        row_y = (row_top + row_bottom) / 2
        if selected:
            highlight = FancyBboxPatch(
                (left - 0.01, row_bottom + 0.006),
                right - left + 0.02,
                row_height - 0.012,
                boxstyle="round,pad=0.006,rounding_size=0.009",
                linewidth=0,
                facecolor=pale,
                transform=ax.transAxes,
            )
            ax.add_patch(highlight)
        weight = "bold" if selected else "normal"
        color = accent if selected else INK
        ax.text(
            condition_x,
            row_y,
            condition,
            transform=ax.transAxes,
            fontsize=12,
            fontweight=weight,
            color=color,
            va="center",
        )
        ax.text(
            sigma_x,
            row_y,
            sigma,
            transform=ax.transAxes,
            fontsize=12,
            fontweight=weight,
            color=color,
            va="center",
            ha="right",
        )
        ax.text(
            rate_x,
            row_y,
            rate,
            transform=ax.transAxes,
            fontsize=12,
            fontweight=weight,
            color=color,
            va="center",
            ha="right",
        )
        if index < len(rows) - 1:
            ax.plot(
                [left, right],
                [row_bottom, row_bottom],
                color=RULE,
                linewidth=0.75,
                transform=ax.transAxes,
            )

    ax.text(
        left,
        y + 0.04,
        result,
        transform=ax.transAxes,
        fontsize=10.8,
        fontweight="bold",
        color=accent,
        va="center",
    )


def add_process_box(
    ax: plt.Axes, *, x: float, y: float, width: float, height: float, text: str
) -> None:
    """Add one box in the centered three-step process row."""
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.01,rounding_size=0.014",
        facecolor="#f2f5f8",
        edgecolor="#7b8794",
        linewidth=1,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=11.2,
        fontweight="bold",
        color=INK,
        linespacing=1.25,
        transform=ax.transAxes,
    )


def render() -> None:
    """Render both publication formats from one layout definition."""
    # The prior layout used 13.2 x 10.2 inches. Extra height provides explicit
    # whitespace between title, subtitle, process row, cards, and footer.
    figure = plt.figure(figsize=(13.2, 11.4), facecolor=PAPER)
    ax = figure.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.text(
        0.5,
        0.968,
        "Frequency geometry selects when to swap the denoiser",
        ha="center",
        va="top",
        fontsize=25,
        fontweight="bold",
        color=INK,
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.915,
        "Fourier radius defines the measurement subspace; sigma defines intervention time.",
        ha="center",
        va="top",
        fontsize=13.5,
        color=MUTED,
        transform=ax.transAxes,
    )

    process_y = 0.79
    process_height = 0.085
    process_boxes = [
        (0.05, 0.235, "Freeze split\nat r = 4"),
        (0.35, 0.30, "Measure C(sigma) and W(sigma)\nin each subspace"),
        (0.715, 0.235, "Select sampled sigma levels\nand swap the whole denoiser"),
    ]
    for x, width, text in process_boxes:
        add_process_box(
            ax,
            x=x,
            y=process_y,
            width=width,
            height=process_height,
            text=text,
        )

    process_midpoint = process_y + process_height / 2
    for start, end in ((0.298, 0.337), (0.663, 0.702)):
        ax.annotate(
            "",
            xy=(end, process_midpoint),
            xytext=(start, process_midpoint),
            arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.4},
            xycoords=ax.transAxes,
        )

    card_y = 0.275
    card_height = 0.455
    add_result_card(
        ax,
        x=0.045,
        y=card_y,
        width=0.44,
        height=card_height,
        label="LOW FREQUENCY",
        radius="r ≤ 4",
        geometry="Geometry: C_low(σ), W_low(σ)",
        selection="selects one sampled candidate at σ = 3.257",
        rows=[
            ("Before control", "5.315", "80.08%", False),
            ("Candidate", "3.257", "73.44%", True),
            ("After control", "1.923", "71.09%", False),
        ],
        accent=LOW,
        pale=LOW_PALE,
        result="Result: low-derived suppression was not supported.",
    )
    add_result_card(
        ax,
        x=0.515,
        y=card_y,
        width=0.44,
        height=card_height,
        label="HIGH FREQUENCY",
        radius="r > 4",
        geometry="Geometry: C_high(σ), W_high(σ)",
        selection="selects two sampled candidates at σ = 1.923, 1.088",
        rows=[
            ("Before control", "3.257, 5.315", "65.23%", False),
            ("Candidate", "1.088, 1.923", "55.08%", True),
            ("After control", "0.296, 0.585", "66.41%", False),
        ],
        accent=HIGH,
        pale=HIGH_PALE,
        result="Result: only high-derived suppression was supported.",
    )

    note = FancyBboxPatch(
        (0.045, 0.095),
        0.91,
        0.115,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1,
        edgecolor=RULE,
        facecolor="#f8fafc",
        transform=ax.transAxes,
    )
    ax.add_patch(note)
    ax.text(
        0.065,
        0.173,
        "Interpretation",
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        color=INK,
        va="center",
    )
    ax.text(
        0.065,
        0.136,
        "The intervention swaps the whole denoiser at the listed σ values; it does not swap Fourier components.",
        transform=ax.transAxes,
        fontsize=11.2,
        color=INK,
        va="center",
    )
    ax.text(
        0.065,
        0.106,
        "The high candidate is two sampled levels, not a continuously validated interval.",
        transform=ax.transAxes,
        fontsize=10.5,
        color=MUTED,
        va="center",
    )
    ax.text(
        0.5,
        0.045,
        "Sampler indices (reproducibility only): low target i = 8; high target i = 9, 10.",
        transform=ax.transAxes,
        fontsize=10.5,
        color=MUTED,
        ha="center",
        va="center",
    )

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_DIRECTORY / "low_high_directional_swap_table.png",
        dpi=220,
        facecolor=PAPER,
    )
    figure.savefig(
        OUTPUT_DIRECTORY / "low_high_directional_swap_table.pdf",
        facecolor=PAPER,
    )
    plt.close(figure)


if __name__ == "__main__":
    render()
