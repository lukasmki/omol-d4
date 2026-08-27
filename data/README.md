# Data

A 64-molecule (`n4`) omol water box run both ways: UMA alone and UMA + the D4
three-body term, 30 ps of NPT each. The CSVs are stage 2's volume traces and
the PNGs are stage 3's figures.

| File | What it is |
| --- | --- |
| `npt_volume_uma-s-1p2p1_omol_n4.csv` | UMA-s-1.2.1 alone |
| `npt_volume_uma-s-1p2p1_omol_n4_atm.csv` | the same run with the D4 three-body term |
| `density_vs_time.png`, `volume_vs_time.png`, `temperature_vs_time.png` | both traces, plotted together |

Regenerate the figures, or re-read the density, without touching a GPU:

```sh
cd ../scripts
python 3-plot.py --outdir ../data
python 2-npt.py --analyse --model uma-s-1p2p1 --task omol --n-side 4 --outdir ../data
```

The three `plot_*.py` scripts that used to live here are now
`omol_d4.plotting`, driven by `scripts/3-plot.py`.
