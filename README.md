# IITG.AI-recruitment-task-2

# Bangladesh Electricity Demand Forecasting

**Project Goal**  
Predict the **next hour's electricity demand** (`demand_mw` at t+1) for Bangladesh's national power grid.  

**Final Score**: **3.5% MAPE** on the 2024 test set

---

### Datasets
- **PGCB_date_power_demand.csv** → Main hourly grid data (demand, generation, source breakdown, load shedding)
- **weather_data.csv** → Hourly weather data from Dhaka
- **economic_full_1.csv** → World Bank annual macroeconomic indicators

---

### Key Challenges & How I Handled Them

#### 1. PGCB Data Cleaning (The Hardest Part)
The raw PGCB data had many issues:
- Half-hourly timestamps (:30)
- Duplicate timestamps
- Missing hours
- Blackout rows (sum of sources = 0 but demand > 0)
- Outliers in demand, generation, and load shedding
- Sparse columns like solar and wind that behave very differently
- Mainly strong connections between columns

**How we cleaned it:**

- Parsed datetime and sorted everything chronologically.
- Removed exact duplicate timestamps by taking the mean.
- Handled :30 records smartly: averaged when :00 existed, otherwise used weighted interpolation (0.7 × previous + 0.3 × half-hour).
- Reindexed to a complete hourly timeline and forward-filled missing timestamps.
- **Blackout fixing**: Used physical relationship (`demand = generation + load_shedding`) and scaled source mix from the next valid period (bfill + scale).
- **Outlier detection & fixing**:
  - Used rolling MAD (Median Absolute Deviation) with short (24h) and long (7-day) windows.
  - Added jump detection using lag difference.
  - For source columns, we used **column-wise fixing** instead of global fixing because solar and wind are very sparse.
  - For each bad row (detected via physical imbalance), we identified the biggest culprit column using deviation from its own rolling median and corrected only that column to restore physical balance.
  - Load shedding outliers were zeroed out before using in calculations.

This step was extremely important because the physical relationship between sources, generation, and demand was the backbone of reliable cleaning.

#### 2. Economic Indicators Selection
The original economic file had 1500+ indicators. We carefully selected only the most relevant ones:

**Final chosen indicators:**
- Access to electricity (% of population)
- GDP (current US$)
- GDP growth (annual %)
- Manufacturing value added (% of GDP)
- Manufacturing value added growth (annual %)
- Total population
- Population growth (annual %)

**How I selected them:**
- Focused on **energy demand drivers**: GDP level & growth, manufacturing (power-intensive sector in BD), population, and electrification rate.
- Dropped redundant columns (e.g., GNI, per-capita versions when we already had total population).
- Kept both absolute level and growth rate for GDP and manufacturing because they capture different signals (long-term trend vs yearly changes).
- Converted from wide format (years as columns) to proper long format for easy merging.


---

### Feature Engineering

- Strong lag features on demand (1h, 24h, 168h)
- Rolling means (24h and 168h windows)
- Calendar features (hour, dayofweek, month, is_weekend)
- Wind direction cyclical features (sin/cos)
- Weather features merged from Dhaka station
- Economic features + two derived features (`gdp_per_capita` and `econ_growth`)
- Checking similar cols like temperature cols and drop them to decrease dimensionality

**Target Preparation**  
Since we need to predict **next hour’s** demand, we shifted the target column by -1 and dropped rows with NaN target.

---

### Modeling

- Model: **XGBoost**
- Training: 2015–2023 (train), 2024 (test) — strict chronological split
- Best parameters found: `n_estimators=500`, `max_depth=8`, `learning_rate=0.03`
- Final MAPE on 2024 test set: **3.5%**

---

### Important Decisions Summary

- All statistics, thresholds, and imputations were calculated **only on train** data.
- Heavy emphasis on physical consistency (generation + load shedding ≈ demand).
- Treated solar and wind differently because of their sparse nature.
- Carefully selected only high-signal economic indicators instead of using everything.
- Made sure there was **no data leakage** (especially target shift).

This was a long but very educational project. The final 3.5% MAPE.

---

**Made by Gowtham Sai Reddy**
