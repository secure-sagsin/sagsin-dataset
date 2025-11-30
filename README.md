# SAGSIN Dataset Assets

This repoistory packages the geographic layers, icons, and minimal plotting utilities used to recreate the tutorial scenarios from [secure-sagsin.github.io](https://secure-sagsin.github.io).
It focuses on:
- Natural Earth shapefiles (land, water, coastline, rivers)
- Ground/sea/air/LEO coordinate CSVs (Madagascar–Mozambique channel, Mexico, Western North America)
- A `tutorial.ipynb` notebook describing synthetic and real-data map compositions

## Repository layout

```
assets/
  icons/                      # SVG markers for plotting
  map/                        # Natural Earth shapefiles + CSV coordinates
sagsin/
  basestations.py             # Node and graph primitives
  environment.py              # DataManager + PlotManager for visualization
tutorial.ipynb                # End-to-end walkthrough
```

## 🔥 Quickstart

```python
import datetime as dt
from sagsin import environment as env

# Hook directly into the open dataset
haps_coords = env.load_haps_positions(
    str(map_dir / "haps_positions.csv"),
    dt.datetime(2020, 9, 28, 10, 0),
)
```

## Getting started

1. Create the Python environment (example using conda):
   ```bash
   conda env create -f environment.yaml
   conda activate sagsin
   ```
2. Open `tutorial.ipynb` in Jupyter Lab/Notebook and run the cells to visualize:
   - Synthetic topology (lat 25°-45°, lon 15°-45°)
   - Madagascar–Mozambique channel (lat -23°-12°, lon 32°-50°)
   - Western North America (lat 23°-38°, lon -120°-90°)
3. `sagsin.environment.PlotManager` already references `assets/icons` and `assets/map`, so no extra configuration is required.
