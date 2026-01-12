"""Generate figures for the explanation pages."""

import numpy as np
import matplotlib.pyplot as plt


def circle_scanlines():
    fig, ax = plt.subplots(figsize=(4, 4))

    # Draw filled circle
    theta = np.linspace(0, 2 * np.pi, 100)
    r = 3
    cx, cy = 4, 4
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    ax.fill(x, y, color="#4a90d9", alpha=0.7)

    # Draw vertical scanlines
    for col in np.arange(0.25, 8, 0.5):
        ax.axvline(col, color="#888", linewidth=0.5, linestyle="-")

        # Mark boundary crossings
        # Circle equation: (x - cx)^2 + (y - cy)^2 = r^2
        # Solve for y: y = cy +/- sqrt(r^2 - (x - cx)^2)
        dx = col - cx
        if abs(dx) < r:
            dy = np.sqrt(r**2 - dx**2)
            ax.plot(col, cy + dy, "o", color="#e63946", markersize=4)
            ax.plot(col, cy - dy, "o", color="#e63946", markersize=4)

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        "../_static/circle_scanlines.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print("Saved circle_scanlines.png")


if __name__ == "__main__":
    circle_scanlines()
