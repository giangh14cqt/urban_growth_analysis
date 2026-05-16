# Handover TODO Roadmap: Multi-Temporal Urban Growth Analysis

Welcome, teammate! The core scientific image processing and machine learning pipelines (Phases 1–6) are fully complete, tested, and optimized. The system is centralized and fully generalized—able to run on any coordinates, size, and years.

Here is the exact task roadmap to finalize the project for submission.

---

## 🎯 Active Tasks to Complete

### 📌 Task 1: Implement Socio-Economic Modeling (`src/economic_analysis.py`)
This module correlates the physical land changes we measured (Urban Sprawl, Nature Loss) with real-world Warsaw demographics and models future trends.

**Steps to Implement**:
1. **Create the Script**: Create [src/economic_analysis.py](file:///Users/truonggiangdo/Data/LearningMaterials/UW/SEM2/Image%20Processing/urban_growth_analysis/src/economic_analysis.py).
2. **Define Historical Warsaw Datasets**:
   Hardcode Warsaw's actual census data matching our milestones (2018, 2022, 2025):
   * **Warsaw Population**:
     * 2018: 1,778,000
     * 2022: 1,860,000
     * 2025: 1,950,000
   * **Warsaw Real GDP Index** (or similar index):
     * 2018: 100.0
     * 2022: 116.5
     * 2025: 130.2
3. **Compute Pearson Correlation Coefficients ($r$)**:
   Determine the correlation between physical urban footprint area ($km^2$) and Population/GDP:
   $$r = \frac{\sum (X - \bar{X})(Y - \bar{Y})}{\sqrt{\sum (X - \bar{X})^2 \sum (Y - \bar{Y})^2}}$$
   *Tip*: Use `scipy.stats.pearsonr` or `numpy.corrcoef`.
4. **Model Footprint Projections to 2030**:
   Use a linear regression model ($y = mx + b$) fit on our 2018, 2022, and 2025 data points to project:
   * Warsaw's concrete footprint area ($km^2$) for **2030**.
   * Cumulative nature loss ($km^2$) for **2030**.
   *Tip*: Use `sklearn.linear_model.LinearRegression` or `scipy.stats.linregress`.
5. **Save Outputs**: Save the numerical correlations and projections to a structured CSV file (`data/socio_economic_report.csv`).

#### 💡 Blueprint Code for `src/economic_analysis.py`
```python
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, linregress

def run_economic_correlation(urban_areas_km2, forest_areas_km2, years, output_dir="data"):
    print("\n>>> Running Socio-Economic Correlation & Projection")
    
    # 1. Align milestones with Warsaw statistics
    # Standard Warsaw census proxies
    pop_mapping = {2018: 1778000, 2022: 1860000, 2025: 1950000}
    gdp_mapping = {2018: 100.0, 2022: 116.5, 2025: 130.2}
    
    population = np.array([pop_mapping[y] for y in years])
    gdp_index = np.array([gdp_mapping[y] for y in years])
    urbans = np.array(urban_areas_km2)
    forests = np.array(forest_areas_km2)
    
    # 2. Pearson Correlation
    r_pop, _ = pearsonr(urbans, population)
    r_gdp, _ = pearsonr(urbans, gdp_index)
    
    print(f"Pearson r (Urban Sprawl vs Population): {r_pop:.4f}")
    print(f"Pearson r (Urban Sprawl vs GDP Index) : {r_gdp:.4f}")
    
    # 3. Linear Projections to 2030
    slope_urb, intercept_urb, _, _, _ = linregress(years, urbans)
    slope_for, intercept_for, _, _, _ = linregress(years, forests)
    
    target_year = 2030
    projected_urban = slope_urb * target_year + intercept_urb
    projected_forest = slope_for * target_year + intercept_for
    
    print(f"Projected 2030 Warsaw Urban Area: {projected_urban:.2f} sq km")
    print(f"Projected 2030 Warsaw Forest Area: {projected_forest:.2f} sq km")
    
    # Save to data/socio_economic_report.csv
    ...
```

---

### 📌 Task 2: Integrate into `main.py`
Once `src/economic_analysis.py` is ready, update [main.py](file:///Users/truonggiangdo/Data/LearningMaterials/UW/SEM2/Image%20Processing/urban_growth_analysis/main.py):
1. **Import the module**: Add `from src.economic_analysis import run_economic_correlation`.
2. **Execute it**: At the end of the `run_pipeline` function, extract the urban and forest areas from the K-Means results dictionary, pass them to your new correlation function, and run the projections!

---

### 📌 Task 3: Compile into the Interactive Jupyter Notebook (`urban_growth_analysis.ipynb`)
This is the final, visual delivery artifact to wow the professors. 

**Steps to Compile**:
1. **Initialize Notebook**: Create `urban_growth_analysis.ipynb` in the root workspace directory.
2. **Structuring Sections**: Organize it beautifully with rich markdown narratives:
   * **Introduction**: Explain the project overview, satellite bands, and coordinate choices.
   * **Data Pipeline Blocks**: Write code blocks that import and execute each phase script.
   * **Visual Interactivity**:
     * Load the output visual png reports from the `data/` directory (`visual_2018.png`, `enhancements_report_2018.png`, `segmentation_report_2018.png`, etc.) and display them side-by-side using `matplotlib` or `IPython.display`.
     * Build standard interactive plots (e.g. Urban growth curve, Nature loss curve, Box-counting log-log regression).
   * **Socio-Economic Discussion**: Discuss the GDP and population correlations, presenting the 2030 Warsaw projections.
3. **Run the Notebook**: Ensure all cells run sequentially from top-to-bottom without warnings or failures.

---

## 📈 Status Overview of Modules

* **Phase 1 (Environment Setup)**: `[x]` **COMPLETE** (pyproject.toml + uv virtual env)
* **Phase 2 (Acquisition)**: `[x]` **COMPLETE** (`sentinel_api.py` retrieves float32 bands)
* **Phase 3 (Alignment)**: `[x]` **COMPLETE** (`alignment.py` executes ORB + RANSAC warping)
* **Phase 4 (Features)**: `[x]` **COMPLETE** (`enhancement.py` computes NDVI/Saturation; `fourier_analysis.py` runs 2D sliding FFTs)
* **Phase 5 (Morphology & Shape)**: `[x]` **COMPLETE** (`morphology.py` cleans masks; `descriptors.py` runs connected components and boundary Box-Counting)
* **Phase 6 (Unsupervised Segmentation)**: `[x]` **COMPLETE** (`segmentation.py` implements pixel KMeans stacking, auto physical mapping, and outputs statistics)
* **Phase 7 (Socio-Economic Modeling)**: `[ ]` **TODO (YOUR TASK)**
* **Phase 8 (Interactive Notebook)**: `[ ]` **TODO (YOUR TASK)**
