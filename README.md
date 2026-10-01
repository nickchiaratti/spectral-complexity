# Spectral Complexity Pipeline

The codebase provides a sensor-agnostic framework to calculate local spectral complexity from multi-temporal satellite imagery. The codebase processes Harmonized Landsat/Sentinel-2 (HLS) surface reflectance data alongside hyperspectral data from the Tanager-1 satellite in a common orthorectified grid. 

It evaluates local spectral complexity by extracting endmembers and calculating the localized volume of their resulting parallelotope via the Gram matrix determinant, and performs continuous change detection (CCD) through harmonic regression on the derived time-series metrics.

## Pipeline Orchestrator

**`HarmonizedSC_run_pipeline.py`**
The consolidated entry point. It starts the end-to-end execution of a 6-step pipeline, handling state persistence, and configuration routing.

## Pipeline Steps / Modules

**1. `HLS30_earthAccess_to_hdf5.py`**
Queries NASA's CMR STAC API for HLSS30 and HLSL30 tiles. Downloads the raw source data via `earthaccess` and structures it into a native, unprojected HDF-EOS5 archive. An active NASA EarthAccess account is required. 

**2. `Tanager_STAC_downloader.py`**
Queries the Planet STAC catalog for Tanager-1 Basic Swath Reflectance scenes matching the spatial constraints, downloading the accompanying hyperspectral HDF5 swaths.

**3. `HLST_constellation_to_hdf5.py`**
The harmonization engine. It identifies valid dates based on cloud/roi coverage and robustly streams both HLS and Tanager swaths into a single, unified `Albers Equal Area` master grid, saving everything into a consolidated `.h5` file.

**4. `HLST_SC_calculations.py`**
Iterates over the temporally stacked HDF5 frames to calculate the sliding-window spectral complexity (volumes and geometric heights), appending the generated metric cubes directly back into the source HDF5 file.

**5. `harmonized_CCD_main.py`**
Performs Continuous Change Detection (CCD) harmonic analysis on the spectral complexity series using Lasso/RLS sliding-window regression. Generates anomaly models and saves the CCD results (predicted series, RMSE, and change dates) into a dedicated `/CCD_Results` group inside the harmonized file.

**6. `harmonized_CCD_vis.py`**
Renders spatial anomaly overlays and interactive pixel-level time-series plots. Enables visualization of the harmonic model fits against the raw complexity data.

## Core Libraries

**`SpecComplexTorch.py` / `SpecComplex.py`**
The core mathematical library. Provides PyTorch-accelerated algorithms for endmember extraction via iterative Orthogonal Subspace Projection (MaxD) and calculates exact parallelotope volumes.

**`hdfeos_odl.py`**
Utility for generating strict HDF-EOS5 ODL metadata structures required for geospatial compatibility in tools like Panoply or GDAL.
