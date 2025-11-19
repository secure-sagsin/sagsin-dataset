# SAGSIN Dataset Assets

This repository packages the geographic layers, icons, and minimal plotting utilities used to recreate the tutorial scenarios from [secure-sagsin.github.io](https://secure-sagsin.github.io). It focuses on:
- Natural Earth shapefiles (land, water, coastline, rivers)
- Ground/sea/air/LEO coordinate CSVs (Madagascar–Mozambique channel, Mexico, Western North America)
- The trimmed `ssir` package with only the `basestations` and `environment` modules
- A `tutorial.ipynb` notebook describing synthetic and real-data map compositions

> **Note**: Plotting utilities are intentionally minimal and contain no routing/pathfinder code. For the full SSIR implementation, consult the main research repository or [secure-sagsin.github.io](https://secure-sagsin.github.io).

## Repository layout

```
assets/
  icons/                      # SVG markers for plotting
  map/                        # Natural Earth shapefiles + CSV coordinates
ssir/
  basestations.py             # Node and graph primitives
  environment.py              # DataManager + PlotManager for visualization
tutorial.ipynb                # End-to-end walkthrough
```

## Getting started

1. Create the Python environment (example using conda):
   ```bash
   conda env create -f environment.yaml
   conda activate ssir
   ```
2. Open `tutorial.ipynb` in Jupyter Lab/Notebook and run the cells to visualize:
   - Synthetic topology (lat 25°–45°, lon 15°–45°)
   - Madagascar–Mozambique channel (lat -23°–-12°, lon 32°–50°)
   - Western North America (lat 23°–38°, lon -120°–-90°)
3. `ssir.environment.PlotManager` already references `assets/icons` and `assets/map`, so no extra configuration is required.

For demos, screenshots, and overview material, visit [secure-sagsin.github.io](https://secure-sagsin.github.io).
