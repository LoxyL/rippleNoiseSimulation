import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 18
rcParams['axes.labelsize'] = 15
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.rm'] = 'Times New Roman'
rcParams['mathtext.it'] = 'Times New Roman:italic'
rcParams['mathtext.bf'] = 'Times New Roman:bold'
rcParams['mathtext.tt'] = 'Times New Roman'

def generate_asymmetric_triangle(time_array: np.ndarray, period: float, duty: float) -> np.ndarray:
    """Generate an asymmetric triangle wave in [-1, 1].

    The wave rises linearly from -1 to +1 during the rising portion (length = duty*period),
    then falls linearly from +1 back to -1 during the remaining portion.
    """
    if not (0.0 < duty < 1.0):
        raise ValueError("duty must be between 0 and 1 (exclusive)")

    t_mod = np.mod(time_array, period)
    rise_mask = t_mod < duty * period

    y = np.empty_like(time_array, dtype=float)

    # Rising segment: -1 -> +1 over duty*period
    y[rise_mask] = -1.0 + (t_mod[rise_mask] / (duty * period)) * 2.0

    # Falling segment: +1 -> -1 over (1-duty)*period
    t_fall = t_mod[~rise_mask] - duty * period
    y[~rise_mask] = 1.0 - (t_fall / ((1.0 - duty) * period)) * 2.0

    return y


def plot_ripple_example():
    # Parameters roughly matching the provided figure
    period = 1.0
    duty = 0.4  # rising time as fraction of period
    t_start, t_end = 0.0, 3.0

    t = np.linspace(t_start, t_end, 1200)
    v = generate_asymmetric_triangle(t, period=period, duty=duty)

    fig, ax = plt.subplots(figsize=(12, 4.5))

    # Plot the triangle waveform
    ax.plot(t, v, color="#0b61ff", linewidth=2.5)

    # Reference lines
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.axhline(1.0, color="red", linestyle=(0, (5, 5)), linewidth=1.0)
    ax.axhline(-1.0, color="red", linestyle=(0, (5, 5)), linewidth=1.0)

    # Vertical guide lines at key instants
    guide_times = [0.4, 1.0, 1.4, 2.0, 2.4]
    for x in guide_times:
        ax.axvline(x, color="gray", linestyle=(0, (3, 7)), linewidth=1.0, alpha=0.7)

    # Labels for rising/falling edge
    ax.text(0.12, -1.1, "Rising\nEdge", ha="center", va="top", fontsize=11)
    ax.text(0.60, -1.1, "Falling\nEdge", ha="center", va="top", fontsize=11)

    # Time window annotations for DT and (1-D)T within a period
    y_anno = -1.22
    # DT arrow (from 1.0 to 1.0 + D*T = 1.4)
    ax.annotate("",
                xy=(1.4, y_anno), xytext=(1.0, y_anno),
                arrowprops=dict(arrowstyle="<->", color="black"))
    ax.text(1.2, y_anno - 0.08, "DT", ha="center", va="top")

    # (1-D)T arrow (from 1.4 to 2.0)
    ax.annotate("",
                xy=(2.0, y_anno), xytext=(1.4, y_anno),
                arrowprops=dict(arrowstyle="<->", color="black"))
    ax.text(1.7, y_anno - 0.08, "Period (T)  (1-D)T", ha="center", va="top")

    # Ripple amplitude annotation (vertical, red, near x=2.5)
    ax.annotate("",
                xy=(2.5, 0.95), xytext=(2.5, -1.05),
                arrowprops=dict(arrowstyle="<->", color="red", linewidth=1.5))
    ax.text(2.52, -0.05, r"$V_\mathrm{pp,ripple}$", color="red", rotation=0,
            ha="left", va="center", fontsize=12)

    # Axes cosmetics
    ax.set_xlim(t_start, t_end)
    ax.set_ylim(-1.55, 1.65)
    ax.set_xlabel("Time / $\mu$s")
    ax.set_ylabel("Voltage / V")
    ax.grid(True, which="both", axis="both", linestyle=(0, (2, 6)), alpha=0.4)
    ax.tick_params(direction='in')

    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    fig, _ = plot_ripple_example()
    # Save alongside the script for convenience
    fig.savefig(
        "ripple.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.show()


