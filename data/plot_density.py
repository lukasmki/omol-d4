from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt

HERE = Path(__file__).parent

DATASETS = [
    ("npt_volume_uma-s-1p2p1_omol_n4.csv", "UMA-s-1.2.1"),
    ("npt_volume_uma-s-1p2p1_omol_n4_atm.csv", "UMA-s-1.2.1(3B-D4)"),
]


def main():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for filename, label in DATASETS:
        df = pl.read_csv(HERE / filename)
        ax.plot(df["time_ps"], df["density_g_cm3"], lw=1.0, alpha=0.8, label=label)

    ax.axhline(1.0, color="black", linestyle="dashed", alpha=0.25)
    ax.axhline(1.1, color="black", linestyle="dashed", alpha=0.25)

    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Density (g/cm$^3$)")
    ax.set_title("NPT density vs. time")
    ax.legend()
    fig.tight_layout()

    out = HERE / "density_vs_time.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
