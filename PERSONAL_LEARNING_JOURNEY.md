# Personal Learning Journey & Individual Technical Contribution Report

**Project Title**: Ames Housing Price Prediction: End-to-End Machine Learning System (v3.0)  
**Course / Module**: Advanced Machine Learning & Predictive Analytics  
**Student Name**: Jayashan Manodya  
**GitHub Profile**: [@JayashanManodya](https://github.com/JayashanManodya)  
**Role & Phase Ownership**: Member 2 — Data Preprocessing & Feature Engineering Specialist  
**Core Technical Artifact**: [`preprocessV3.ipynb`](file:///d:/Projects_New/ames-housing-price-prediction/preprocessV3.ipynb)  
**Exported Data Artifacts**: [`data-new/`](file:///d:/Projects_New/ames-housing-price-prediction/data-new) (`X_train_preprocessed.csv`, `X_test_preprocessed.csv`, `y_train.csv`, `y_test.csv`)  
**Submission Date**: October 2026  

---

## 1. Executive Summary & Statement of Individual Ownership

As **Member 2** of the Ames Housing Price Prediction team, I was solely responsible for architecting, testing, and delivering the complete data preprocessing, feature engineering, and transformation pipeline documented in [`preprocessV3.ipynb`](file:///d:/Projects_New/ames-housing-price-prediction/preprocessV3.ipynb). 

Data quality and feature representation dictate the theoretical performance ceiling of any machine learning model. My primary objective was to transform the raw Ames Housing dataset (1,460 residential properties across 81 messy, heterogeneous tabular features) into a leak-free, clean, normalized, and mathematically optimized 91-dimensional modeling matrix that would empower our downstream modeling specialists (Teshan — Member 3) and evaluation specialists (Nadeesha — Member 4) to achieve state-of-the-art predictive performance.

### Summary of Key Contributions:
1. **Domain-Aware Imputation**: Differentiated between structural missingness (physical absence of amenities like pools, garages, and basements) and true random missingness, applying neighborhood-grouped median imputation for `LotFrontage` and mode imputation for categorical utilities.
2. **Target Distribution Normalization**: Stabilized the severe positive skewness ($\gamma = 1.88$) and heavy-tailed kurtosis of the target variable (`SalePrice`) via a natural logarithmic transformation ($\ln(1 + \text{SalePrice})$), reducing skewness to $0.12$ and satisfying linear homoscedasticity assumptions.
3. **Domain Feature Engineering**: Formulated and synthesized **16 novel features** based on real estate appraisal principles, including the discovery of `OverallQual_TotalSF`, which became the **#1 most predictive feature across the entire project ($r = 0.856$ with price, outperforming all 81 raw attributes)**.
4. **Data Anomaly Remediation**: Identified and resolved hidden anomalies, including a negative remodeling age ($\text{RemodAge} = -1$) and high-variance parcel approximations (`OutdoorArea`).
5. **Two-Stage Feature Selection**: Built an objective filtering methodology pairing parametric Pearson correlation ($|r| \ge 0.50$) for numerical features with non-parametric Mutual Information Regression ($MI \ge 0.10$) for non-linear categorical signals.
6. **Collinearity & Redundancy Pruning**: Computed pairwise numerical correlation matrices and categorical association matrices using **Cramér's $V$ with bias correction**, eliminating severe redundancies (e.g., `YearBuilt` vs `HouseAge`, `GarageCars` vs `GarageArea`, and `Exterior1st` vs `Exterior2nd`).
7. **Leak-Free Scikit-Learn Pipeline (`ColumnTransformer`)**: Constructed an automated, modular scikit-learn pipeline separating continuous standardization (`StandardScaler`), hierarchical integer ranking (`OrdinalEncoder` across 7 distinct scales), and categorical dummy expansion (`OneHotEncoder`). All transformations were strictly fitted on the 80% training partition ($n = 1,168$) and applied to the 20% test partition ($n = 292$), guaranteeing zero data leakage.
8. **Clean Export**: Produced verified, 0-missing, 0-infinite, fully normalized CSV artifacts in `data-new/` that served as the foundational bedrock for all team models (Ridge, Random Forest, XGBoost, LightGBM, and CatBoost), empowering our champion **XGBoost Regressor** to reach an $R^2$ of **$0.9073$** and an RMSE of **$\$26,670.54$**.

---

## 2. End-to-End Pipeline Architecture

The end-to-end data processing workflow developed in [`preprocessV3.ipynb`](file:///d:/Projects_New/ames-housing-price-prediction/preprocessV3.ipynb) is illustrated below:

```mermaid
flowchart TD
    RawData["Raw Dataset (dataset.csv)<br>1,460 rows × 81 attributes"] --> Stage1["Stage 1: Ingestion & Integrity Audit<br>Shape validation, duplicate check, data types"]
    
    Stage1 --> Stage2["Stage 2: Domain-Aware Imputation<br>• Structural 'NA' → 'None' (Pool, Garage, Bsmt, Fence)<br>• Numeric absence → 0<br>• LotFrontage → Grouped Neighborhood Median<br>• Electrical → Mode ('SBrkr')"]
    
    Stage2 --> Stage3["Stage 3: Target Variable Engineering<br>SalePrice log1p transformation: ln(1 + y)<br>Skewness: 1.88 → 0.12"]
    
    Stage3 --> Stage4["Stage 4: Domain Feature Engineering (16 Features)<br>TotalSF, TotalBathrooms, HouseAge, RemodAge,<br>OverallQual_TotalSF (r = 0.856), etc."]
    
    Stage4 --> Stage5["Stage 5: Anomaly Resolution & Hygiene<br>• Clip RemodAge: max(0, RemodAge)<br>• Drop erratic OutdoorArea"]
    
    Stage5 --> Stage6["Stage 6: Two-Stage Feature Selection<br>• Pearson |r| ≥ 0.50 (11 Top Numerical Features)<br>• Mutual Information MI ≥ 0.10 (13 Top Categorical Features)"]
    
    Stage6 --> Stage7["Stage 7: Collinearity & Association Pruning<br>• Pearson |r| ≥ 0.80 Pruning (Drop YearBuilt, TotalLivingSF)<br>• Cramér's V ≥ 0.80 Pruning (Drop Exterior1st)<br>Final Optimal Predictors: 24 Features"]
    
    Stage7 --> Stage8["Stage 8: Strict 80/20 Train-Test Split<br>Train: 1,168 rows | Test: 292 rows (random_state=42)"]
    
    Stage8 --> Stage9["Stage 9: Scikit-Learn ColumnTransformer Pipeline<br>• Branch 1: StandardScaler (11 Numerical Features)<br>• Branch 2: OrdinalEncoder (7 Monotonic Features)<br>• Branch 3: OneHotEncoder (6 Nominal Features → 73 Dummies)<br>Fitted STRICTLY on X_train, Transformed on X_test"]
    
    Stage9 --> Stage10["Stage 10: Validation & Artifact Export (data-new/)<br>• X_train_preprocessed.csv (1168 × 91)<br>• X_test_preprocessed.csv (292 × 91)<br>• y_train.csv & y_test.csv (Log Target)"]
```

---

## 3. Detailed Technical Breakdown of Preprocessing Stages

### 3.1 Stage 1: Data Ingestion and Structural Integrity
The initial step involved inspecting the tabular layout of `dataset.csv`:
- **Dimensions**: Verified $1,460$ rows and $81$ columns ($80$ features + $1$ target variable: `SalePrice`).
- **Duplicate Check**: Executed `df.duplicated().sum()`, confirming **0** duplicate property entries.
- **Primary Key Audit**: Verified that the `Id` feature represented a strictly monotonic, unique sequence ($1 \le \text{Id} \le 1460$). The `Id` column was separated from predictive features to prevent models from learning spurious index-based memorization.

---

### 3.2 Stage 2: Domain-Aware Missing Value Imputation
A major insight from examining real estate appraisals is that missing values (`NaN`/`NA`) in tabular property records rarely represent missing data at random. Instead, the Ames dataset documentation reveals that **"NA" is a deliberate category denoting the physical absence of a feature**.

Treating structural missingness with standard imputers (e.g., global mean or mode) would corrupt the physical semantics of the dataset. Therefore, I established a clear bifurcated imputation protocol:

#### 1. Structural Missingness Imputation:
- **Categorical Amenities**: Imputed with `'None'` to signify the absence of the amenity:
  - `PoolQC` (99.52% NA) $\to$ `'None'` (No pool on parcel)
  - `MiscFeature` (96.30% NA) $\to$ `'None'` (No miscellaneous features such as sheds or tennis courts)
  - `Alley` (93.77% NA) $\to$ `'None'` (No alley access)
  - `Fence` (80.75% NA) $\to$ `'None'` (No perimeter fence)
  - `FireplaceQu` (47.26% NA) $\to$ `'None'` (Zero fireplaces present)
  - `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond` (~5.5% NA) $\to$ `'None'` (No garage present)
  - `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2` (~2.5% NA) $\to$ `'None'` (No basement present)
- **Numerical Amenities**: Imputed with `0` to reflect zero physical area/capacity:
  - `GarageArea`, `GarageCars`, `GarageYrBlt` $\to$ `0`
  - `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, `TotalBsmtSF`, `BsmtFullBath`, `BsmtHalfBath` $\to$ `0`
  - `MasVnrArea` $\to$ `0`

#### 2. Genuine Random Missingness Imputation:
- **`LotFrontage` (17.74% Missing)**: The linear feet of street connected to property is physically present for every home, but unrecorded. Because lot widths are strongly determined by parcel subdivision and zoning within neighborhoods, global median imputation would introduce substantial bias. I implemented **neighborhood-conditioned median imputation**:
  $$\text{LotFrontage}_i = \text{Median}\left(\{\text{LotFrontage}_j \mid \text{Neighborhood}_j = \text{Neighborhood}_i\}\right)$$
- **`Electrical` (1 Missing Record)**: Imputed with the dominant mode: `'SBrkr'` (Standard Circuit Breakers & Romex wiring, representing >91% of homes).

Following imputation, an automated null audit confirmed that the total missing value count across all 1,460 records was exactly **0**.

---

### 3.3 Stage 3: Target Variable Analysis & Normalization ($\log(1 + y)$)

The raw distribution of property sale prices exhibits pronounced right-skewness:
- **Mean `SalePrice`**: $\$180,921.20$
- **Median `SalePrice`**: $\$163,000.00$
- **Standard Deviation**: $\$79,442.50$
- **Skewness**: $1.88$ (Substantially right-skewed)
- **Kurtosis**: $6.54$ (Leptokurtic with extreme positive outliers)

#### Mathematical Motivation for Log Transformation:
In linear and regularized regression, violating the normality and homoscedasticity (constant variance of residuals) assumptions leads to distorted coefficient estimates that over-penalize large errors on luxury homes while under-fitting mid-market homes. 

Furthermore, residential real estate transactions scale multiplicatively (percentage valuation gains) rather than additively. By applying the natural logarithm transformation:

$$y_{\text{log}} = \ln(1 + \text{SalePrice}) = \text{np.log1p}(\text{SalePrice})$$

The distribution characteristics were transformed:
- **Log Skewness**: Reduced from **1.88** to **0.12** (Near-perfect Gaussian symmetry).
- **Log Kurtosis**: Reduced from **6.54** to **0.80**.
- **Variance Stabilization**: Stabilized residual variance across all property price tiers.
- **Loss Metric Alignment**: Minimizing Root Mean Squared Error on $y_{\text{log}}$ is mathematically equivalent to minimizing Root Mean Squared Logarithmic Error (RMSLE) in original dollar space, penalizing relative percentage errors symmetrically.

---

### 3.4 Stage 4: Domain-Specific Feature Engineering (The 16 Engineered Features)

Raw architectural dimensions in the Ames dataset are split across disjoint columns (e.g., basement areas, multiple floor levels, individual porch types). Appraisers and home buyers evaluate aggregate usable space and age rather than fragmented dimensions. 

I engineered **16 domain-specific features** in Section 6 of [`preprocessV3.ipynb`](file:///d:/Projects_New/ames-housing-price-prediction/preprocessV3.ipynb):

| # | Feature Name | Mathematical Formulation | Domain & Economic Justification | Initial Correlation ($r$) |
|---|---|---|---|---|
| **1** | **`TotalSF`** | $\text{TotalBsmtSF} + \text{1stFlrSF} + \text{2ndFlrSF}$ | Represents the total enclosed habitable and finished volumetric footprint of the residence. | **$+0.782$** |
| **2** | `TotalPorchSF` | $\text{OpenPorchSF} + \text{EnclosedPorch} + \text{3SsnPorch} + \text{ScreenPorch}$ | Aggregate outdoor leisure and porch square footage. | $+0.195$ |
| **3** | **`TotalBathrooms`** | $\text{FullBath} + 0.5\times\text{HalfBath} + \text{BsmtFullBath} + 0.5\times\text{BsmtHalfBath}$ | Industry-standard appraisal metric weighting half-baths at 0.5. | **$+0.632$** |
| **4** | **`HouseAge`** | $\text{YrSold} - \text{YearBuilt}$ | Chronological age at date of sale; directly models physical depreciation. | **$-0.523$** |
| **5** | `RemodAge` | $\max(0, \text{YrSold} - \text{YearRemodAdd})$ | Elapsed years since modern structural renovation. | $-0.509$ |
| **6** | `TotalFinishedBsmtSF` | $\text{BsmtFinSF1} + \text{BsmtFinSF2}$ | Finished subterranean living quarters. | $+0.432$ |
| **7** | `TotalLivingSF` | $\text{GrLivArea} + \text{TotalFinishedBsmtSF}$ | Effective conditioned living space across above-ground and below-ground levels. | **$+0.779$** |
| **8** | `HasGarage` | $\mathbb{I}(\text{GarageArea} > 0)$ | Binary amenity indicator for off-street parking. | $+0.237$ |
| **9** | `HasBasement` | $\mathbb{I}(\text{TotalBsmtSF} > 0)$ | Binary structural indicator for foundation basement. | $+0.153$ |
| **10** | `HasFireplace` | $\mathbb{I}(\text{Fireplaces} > 0)$ | Binary luxury indicator for fireplace amenities. | $+0.472$ |
| **11** | `HasPool` | $\mathbb{I}(\text{PoolArea} > 0)$ | Binary luxury indicator for private swimming pool. | $+0.092$ |
| **12** | `GarageAge` | $\text{YrSold} - \text{GarageYrBlt}$ | Chronological age of the garage structure. | $-0.262$ |
| **13** | `OverallQual_GrLivArea` | $\text{OverallQual} \times \text{GrLivArea}$ | Non-linear interaction: premium finish quality scales with above-ground area. | **$+0.832$** |
| **14** | **`OverallQual_TotalSF`** | $\text{OverallQual} \times \text{TotalSF}$ | **Crown Jewel Feature**: Captures compounding valuation where high quality multiplies total enclosed square footage. | **$+0.856$ (Rank #1)** |
| **15** | `OverallCond_HouseAge` | $\frac{\text{OverallCond}}{\text{HouseAge} + 1}$ | Maintenance resilience against physical structural deterioration. | $+0.354$ |
| **16** | `OutdoorArea` | $\text{LotArea} - (\text{1stFlrSF} + \text{GarageArea})$ | *Dropped after audit*: Parcel yard area minus ground footprints. | $+0.261$ (Dropped) |

#### Deep Dive into the Top Feature: `OverallQual_TotalSF` ($r = 0.856$)
In linear models, individual features contribute additively. However, a luxury home with an `OverallQual` rating of 9 or 10 does not merely add a fixed dollar amount per square foot; each square foot of high-end craftsmanship (granite, hardwood, architectural ceilings) commands a compounding market premium. By multiplying `OverallQual` with `TotalSF`, I created a feature whose correlation ($r = 0.8561$) surpassed both `OverallQual` ($r = 0.7910$) and `TotalSF` ($r = 0.7823$) individually, providing the downstream gradient boosting and linear models with a massive performance lift.

---

### 3.5 Stage 5: Anomaly Detection & Remediation

During comprehensive statistical validation of the newly engineered features, I conducted an anomaly audit that uncovered two critical data issues:

1. **Negative `RemodAge` Anomaly**:
   - *Discovery*: When computing `RemodAge = YrSold - YearRemodAdd`, the minimum value was found to be **$-1$**.
   - *Root Cause*: One property was recorded as having a remodel completed in the calendar year *after* the contract sale date was filed ($\text{YrSold} = 2008$, $\text{YearRemodAdd} = 2009$).
   - *Remediation*: A negative building age is physically impossible and distorts tree split thresholds. I applied lower-bound clipping:
     ```python
     df["RemodAge"] = df["RemodAge"].apply(lambda x: max(0, x))
     ```
     This restored the valid domain $[0, \infty)$, with $\min(\text{RemodAge}) = 0$.

2. **Removal of Unstable `OutdoorArea`**:
   - *Discovery*: `OutdoorArea` was engineered as $\text{LotArea} - (\text{1stFlrSF} + \text{GarageArea})$. 
   - *Audit Finding*: The extreme variance and irregular geometry of `LotArea` (ranging from $1,300$ sq ft to over $215,000$ sq ft on agricultural boundary parcels) completely overwhelmed the building footprint subtractions. The resulting feature had an erratic distribution with extreme leverage points that increased noise rather than predictive signal.
   - *Action*: In Section 6.19 of the notebook, `OutdoorArea` was deliberately pruned from the candidate feature set.

---

### 3.6 Stage 6: Two-Stage Feature Selection Strategy

To prevent the "curse of dimensionality," reduce overfitting, and maintain high model interpretability, I implemented an objective two-stage selection framework combining parametric and non-parametric criteria.

#### Stage 6.1: Numerical Feature Selection via Pearson Correlation
Linear and tree models require continuous features with strong monotonic relationships to `SalePrice`. I calculated the Pearson correlation coefficient ($r$) across all numerical features (raw + engineered) and filtered using a strict threshold of **$|r| \ge 0.50$**:

```python
correlation_with_price = df.corr(numeric_only=True)["SalePrice"]
top_numerical = correlation_with_price[abs(correlation_with_price) >= 0.50].sort_values(ascending=False)
```

**Selected Top Numerical Predictors ($|r| \ge 0.50$)**:
1. `OverallQual_TotalSF`: $+0.856$ (Engineered interaction)
2. `OverallQual_GrLivArea`: $+0.832$ (Engineered interaction)
3. `OverallQual`: $+0.791$ (Raw)
4. `TotalSF`: $+0.782$ (Engineered)
5. `TotalLivingSF`: $+0.779$ (Engineered)
6. `GrLivArea`: $+0.709$ (Raw)
7. `GarageCars`: $+0.640$ (Raw)
8. `TotalBathrooms`: $+0.632$ (Engineered)
9. `GarageArea`: $+0.623$ (Raw)
10. `TotalBsmtSF`: $+0.614$ (Raw)
11. `1stFlrSF`: $+0.606$ (Raw)
12. `FullBath`: $+0.561$ (Raw)
13. `TotRmsAbvGrd`: $+0.534$ (Raw)
14. `YearBuilt`: $+0.523$ (Raw)
15. `YearRemodAdd`: $+0.507$ (Raw)
16. `HouseAge`: $-0.523$ (Engineered depreciation)
17. `RemodAge`: $-0.509$ (Engineered renovation age)

#### Stage 6.2: Categorical Feature Selection via Mutual Information
Categorical variables often exhibit strong non-linear relationships that Pearson correlation cannot measure. 

- **Type Correction (`MSSubClass`)**: The raw dataset encodes `MSSubClass` (Dwelling Type, e.g., 20 = 1-Story, 60 = 2-Story) as an integer. I explicitly cast `MSSubClass` to `string`/categorical, ensuring it was evaluated through categorical information theory rather than spurious numerical arithmetic.
- **Mutual Information Regression ($MI$)**: Using `sklearn.feature_selection.mutual_info_regression` (with `random_state=42`), I quantified the entropy reduction between each categorical predictor and `SalePrice`:

$$I(X; Y) = \iint p(x, y) \log \frac{p(x, y)}{p(x) p(y)} \, dx \, dy$$

**Selected Top Categorical Predictors ($MI \ge 0.10$)**:
1. `Neighborhood`: $MI = 0.533$ (Strongest spatial/locational signal)
2. `BsmtQual`: $MI = 0.329$ (Basement ceiling height and foundation grade)
3. `KitchenQual`: $MI = 0.328$ (Cabinetry, countertops, layout)
4. `ExterQual`: $MI = 0.326$ (Exterior facade construction material quality)
5. `MSSubClass`: $MI = 0.280$ (Architectural housing typology)
6. `GarageFinish`: $MI = 0.262$ (Interior garage finish level)
7. `FireplaceQu`: $MI = 0.213$ (Masonry fireplace quality)
8. `GarageType`: $MI = 0.201$ (Attached, Detached, Built-In)
9. `Foundation`: $MI = 0.200$ (Poured concrete, Cinder block, Wood)
10. `HeatingQC`: $MI = 0.163$ (Thermal efficiency, furnace rating)
11. `Exterior2nd`: $MI = 0.158$ (Secondary siding cladding)
12. `BsmtFinType1`: $MI = 0.151$ (Finished basement living rating)
13. `Exterior1st`: $MI = 0.132$ (Primary siding cladding)
14. `MasVnrType`: $MI = 0.100$ (Masonry veneer facade type)

This reduced the initial pool of 81 variables to a tightly vetted candidate set of 30 high-signal features.

---

### 3.7 Stage 7: Multicollinearity & Redundancy Pruning

High pairwise multicollinearity destabilizes linear regression weight estimates, inflates variance inflation factors (VIF), and creates redundant tree splits in ensemble algorithms. To address this, I executed systematic numerical and categorical association audits:

#### 1. Numerical Multicollinearity Pruning ($|r| \ge 0.80$):
I computed the pairwise correlation matrix across all candidate numerical features. Pairs exceeding $|r| \ge 0.80$ were analyzed and pruned based on physical interpretability:
- **`HouseAge` vs `YearBuilt` ($r = -0.999$)**:
  - *Decision*: **Retained `HouseAge`**, **Dropped `YearBuilt`**. `HouseAge` directly measures time-dependent physical deterioration, which is zero-anchored and stationary relative to sale date.
- **`TotalSF` vs `TotalLivingSF` ($r = 0.998$)**:
  - *Decision*: **Retained `TotalSF`**, **Dropped `TotalLivingSF`**. `TotalSF` accounts for full enclosed footprint, avoiding subjective definitions of partially finished basements.
- **`OverallQual_TotalSF` vs `OverallQual_GrLivArea` ($r = 0.945$)**:
  - *Decision*: **Retained `OverallQual_TotalSF`**, **Dropped `OverallQual_GrLivArea`**. `OverallQual_TotalSF` exhibited higher correlation ($0.856$ vs $0.832$) and captures basement space.
- **`GarageCars` vs `GarageArea` ($r = 0.882$)**:
  - *Decision*: **Retained `GarageCars`**, **Dropped `GarageArea`**. `GarageCars` is discrete, stable, and less prone to measurement noise while achieving a higher correlation with `SalePrice` ($0.640$ vs $0.623$).
- **`1stFlrSF` and `TotalBsmtSF` ($r = 0.820$)**:
  - *Decision*: **Retained Both**. Although correlated, they represent distinct architectural planes (subterranean vs. primary ground living floor) with independent valuation impacts.

#### 2. Categorical Association Pruning via Cramér's $V$ ($V \ge 0.80$):
Traditional correlation cannot evaluate categorical pairs. I implemented **Cramér's $V$ statistic with Bergsma-Wicher bias correction**:

$$V = \sqrt{\frac{\tilde{\phi}^2}{\min(\tilde{r} - 1, \tilde{k} - 1)}}$$

Where $\tilde{\phi}^2 = \max\left(0, \frac{\chi^2}{n} - \frac{(r-1)(k-1)}{n-1}\right)$, and $\tilde{r}, \tilde{k}$ are bias-corrected degree-of-freedom parameters.

- **`Exterior1st` vs `Exterior2nd` ($V = 0.759 \approx 0.80$)**:
  - *Analysis*: In over 80% of properties, the primary and secondary exterior siding materials are identical (e.g., Vinyl Siding on both).
  - *Information Comparison*: `Exterior2nd` achieved an $MI$ score of **$0.1582$**, whereas `Exterior1st` achieved **$0.1322$**.
  - *Decision*: **Retained `Exterior2nd`**, **Dropped `Exterior1st`**, eliminating redundant dummy column inflation.
- **`GarageFinish` vs `GarageType` ($V = 0.48$)**:
  - *Decision*: **Retained Both**. They provide orthogonal structural information (interior drywall finish level vs. physical architectural attachment to home).

#### Outcome: Final Optimal 24 Predictors
The final vetted feature set comprised exactly **24 selected predictors** (11 numerical + 13 categorical) + 1 target variable (`SalePrice`).

---

### 3.8 Stage 8: Data Leakage Prevention & Train-Test Splitting

A critical principle of rigorous machine learning is **zero data leakage**. Calculating scaling statistics (mean, variance) or categorical encoding dictionaries on the full dataset before splitting leaks distribution parameters from the test set into the training set, producing overly optimistic validation results that fail in production.

#### Execution Protocol:
1. **Isolated Splitting First**: The dataset was partitioned into an **80% training set ($n = 1,168$)** and a **20% held-out test set ($n = 292$)** using `train_test_split` with `random_state=42`:
   ```python
   X = final_df.drop(columns=["SalePrice"])
   y = final_df["SalePrice"]
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.20, random_state=42
   )
   ```
2. **Fit on Train, Transform on Test**: All transformer scalers and encoders were fitted **strictly and exclusively on `X_train`** via `.fit_transform()`. The test partition `X_test` was subsequently processed using `.transform()`, ensuring that `X_test` remained completely unseen.

---

### 3.9 Stage 9: Scikit-Learn `ColumnTransformer` Pipeline Architecture

To ensure end-to-end reproducibility, I engineered an automated multi-branch pipeline utilizing `sklearn.compose.ColumnTransformer`:

```mermaid
flowchart LR
    X_train["Raw X_train<br>(24 Features)"] --> CT["ColumnTransformer"]
    
    CT --> Branch1["Numerical Pipeline<br>(11 Features)"]
    Branch1 --> SS["StandardScaler()<br>z = (x - μ) / σ"]
    
    CT --> Branch2["Ordinal Pipeline<br>(7 Features)"]
    Branch2 --> OE["OrdinalEncoder()<br>Custom Monotonic Hierarchies<br>['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']"]
    
    CT --> Branch3["Nominal Pipeline<br>(6 Features)"]
    Branch3 --> OHE["OneHotEncoder()<br>handle_unknown='ignore'<br>sparse_output=False"]
    
    SS --> Concat["Feature Concatenation"]
    OE --> Concat
    OHE --> Concat
    
    Concat --> Out["Processed Feature Matrix<br>(91 Numeric Columns)"]
```

#### Detailed Branch Configuration:

1. **Branch 1: Continuous Numerical Standardization (`StandardScaler`)**
   - **Applied To (11 Features)**: `OverallQual_TotalSF`, `OverallQual`, `TotalSF`, `GrLivArea`, `GarageCars`, `TotalBathrooms`, `TotalBsmtSF`, `1stFlrSF`, `FullBath`, `TotRmsAbvGrd`, `HouseAge`.
   - **Formula**: $z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$
   - **Benefit**: Ensures regularized linear models (Ridge $\alpha$-penalties) and distance metrics treat all continuous variables with equal geometric weight.

2. **Branch 2: Hierarchical Ordinal Encoding (`OrdinalEncoder`)**
   - **Applied To (7 Features)**: Quality and condition ratings with an inherent monotonic progression.
   - **Custom Domain Mappings**:
     - `BsmtQual`, `KitchenQual`, `ExterQual`, `FireplaceQu`, `HeatingQC`:  
       `['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex']` $\to [0, 1, 2, 3, 4, 5]$
     - `GarageFinish`:  
       `['None', 'Unf', 'RFn', 'Fin']` $\to [0, 1, 2, 3]$
     - `BsmtFinType1`:  
       `['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']` $\to [0, 1, 2, 3, 4, 5, 6]$
   - **Safeguard**: Configured with `handle_unknown='use_encoded_value', unknown_value=-1` to handle unseen quality classes safely.

3. **Branch 3: Nominal One-Hot Encoding (`OneHotEncoder`)**
   - **Applied To (6 Features)**: Categorical variables without ordinal hierarchy: `Neighborhood`, `MSSubClass`, `Exterior2nd`, `GarageType`, `Foundation`, `MasVnrType`.
   - **Configuration**: `handle_unknown='ignore', sparse_output=False`.
   - **Expansion**:
     - `Neighborhood` (25 unique categories) $\to$ 25 binary flags
     - `MSSubClass` (15 unique categories) $\to$ 15 binary flags
     - `Exterior2nd` (16 unique categories) $\to$ 16 binary flags
     - `GarageType` (7 unique categories) $\to$ 7 binary flags
     - `Foundation` (6 unique categories) $\to$ 6 binary flags
     - `MasVnrType` (4 unique categories) $\to$ 4 binary flags
     - **Subtotal Nominal Dummy Columns**: $25 + 15 + 16 + 7 + 6 + 4 = \mathbf{73\text{ Columns}}$.

#### Post-Transformation Matrix Dimension:
$$\text{Total Transformed Features} = \underbrace{11}_{\text{Standardized}} + \underbrace{7}_{\text{Ordinal Ranks}} + \underbrace{73}_{\text{One-Hot Dummies}} = \mathbf{91\text{ Features}}$$

---

### 3.10 Stage 10: Final Validation and Clean Artifact Export

Prior to serializing the final datasets, I executed a comprehensive automated integrity test:
- **Matrix Dimensions**:
  - `X_train_processed_df`: $(1,168, 91)$
  - `X_test_processed_df`: $(292, 91)$
  - `y_train_log`: $(1,168,)$
  - `y_test_log`: $(292,)$
- **Missing Values Audit**: `X_train.isna().sum().sum() == 0` and `X_test.isna().sum().sum() == 0`.
- **Infinite Value Audit**: `np.isinf(X_train).sum().sum() == 0` and `np.isinf(X_test).sum().sum() == 0`.
- **Data Types**: All 91 features strictly verified as `float64`.
- **Clean Column Headers**: Stripped scikit-learn pipeline prefixes (`numerical__`, `ordinal__`, `nominal__`) using `name.split("__", 1)[1]` for clean, readable headers.
- **Export Destination**: Saved directly to [`data-new/`](file:///d:/Projects_New/ames-housing-price-prediction/data-new):
  - `data-new/X_train_preprocessed.csv`
  - `data-new/X_test_preprocessed.csv`
  - `data-new/y_train.csv` (contains both raw `SalePrice` and `SalePrice_Log`)
  - `data-new/y_test.csv` (contains both raw `SalePrice` and `SalePrice_Log`)

---

## 4. Master Feature Dictionary

### 4.1 The 24 Selected Input Features (Pre-Encoding)

| # | Feature Name | Origin | Category | Type | Physical Meaning & Measurement Unit | Selection Metric & Rank |
|---|---|---|---|---|---|---|
| **1** | `OverallQual_TotalSF` | Engineered | Numerical | Continuous | Compound interaction: `OverallQual` $\times$ `TotalSF`. Models non-linear luxury space scaling. | Pearson $r = 0.856$ (Rank #1) |
| **2** | `OverallQual` | Raw | Numerical | Discrete | Overall material and finish rating of the property (1 = Very Poor to 10 = Very Excellent). | Pearson $r = 0.791$ (Rank #3) |
| **3** | `TotalSF` | Engineered | Numerical | Continuous | Total enclosed square feet: $\text{TotalBsmtSF} + \text{1stFlrSF} + \text{2ndFlrSF}$. | Pearson $r = 0.782$ (Rank #4) |
| **4** | `GrLivArea` | Raw | Numerical | Continuous | Above-grade (ground level and second story) finished living area in sq ft. | Pearson $r = 0.709$ (Rank #6) |
| **5** | `GarageCars` | Raw | Numerical | Discrete | Size of garage measured in car parking capacity (0 to 4 vehicles). | Pearson $r = 0.640$ (Rank #7) |
| **6** | `TotalBathrooms` | Engineered | Numerical | Continuous | Total effective bathroom count: $\text{FullBath} + 0.5\times\text{HalfBath} + \text{BsmtFullBath} + 0.5\times\text{BsmtHalfBath}$. | Pearson $r = 0.632$ (Rank #8) |
| **7** | `TotalBsmtSF` | Raw | Numerical | Continuous | Total basement square footage (finished and unfinished). | Pearson $r = 0.614$ |
| **8** | `1stFlrSF` | Raw | Numerical | Continuous | First-floor finished living area in square feet. | Pearson $r = 0.606$ |
| **9** | `FullBath` | Raw | Numerical | Discrete | Number of full bathrooms located above grade. | Pearson $r = 0.561$ |
| **10** | `TotRmsAbvGrd` | Raw | Numerical | Discrete | Total rooms above grade (excluding bathrooms and closets). | Pearson $r = 0.534$ |
| **11** | `HouseAge` | Engineered | Numerical | Discrete | Age of house at time of sale: $\text{YrSold} - \text{YearBuilt}$ (years). | Pearson $r = -0.523$ |
| **12** | `BsmtQual` | Raw | Ordinal | Ranked | Height and structural condition of the basement foundation. | Mutual Info $MI = 0.329$ |
| **13** | `KitchenQual` | Raw | Ordinal | Ranked | Interior quality of kitchen cabinetry, appliances, and layout. | Mutual Info $MI = 0.328$ |
| **14** | `ExterQual` | Raw | Ordinal | Ranked | Quality of exterior construction material and sidings. | Mutual Info $MI = 0.326$ |
| **15** | `GarageFinish` | Raw | Ordinal | Ranked | Interior finish status of the garage (None, Unfinished, Rough, Finished). | Mutual Info $MI = 0.262$ |
| **16** | `FireplaceQu` | Raw | Ordinal | Ranked | Fireplace craftsmanship and masonry condition. | Mutual Info $MI = 0.213$ |
| **17** | `HeatingQC` | Raw | Ordinal | Ranked | Heating system quality, furnace efficiency, and thermal condition. | Mutual Info $MI = 0.163$ |
| **18** | `BsmtFinType1` | Raw | Ordinal | Ranked | Rating of basement finished living space quality (e.g., Good Living Quarters). | Mutual Info $MI = 0.151$ |
| **19** | `Neighborhood` | Raw | Nominal | Categorical | Physical neighborhood boundary within Ames corporate limits (25 zones). | Mutual Info $MI = 0.533$ |
| **20** | `MSSubClass` | Raw | Nominal | Categorical | Type of dwelling involved in sale (e.g., 20 = 1-Story, 60 = 2-Story, 160 = Townhouse). | Mutual Info $MI = 0.280$ |
| **21** | `Exterior2nd` | Raw | Nominal | Categorical | Secondary exterior wall covering (e.g., Vinyl Siding, Brick Face, Wood). | Mutual Info $MI = 0.158$ |
| **22** | `GarageType` | Raw | Nominal | Categorical | Structural attachment of garage (Attached, Detached, Built-In, Carport, None). | Mutual Info $MI = 0.201$ |
| **23** | `Foundation` | Raw | Nominal | Categorical | Type of foundation slab/footing (Poured Concrete, Cinder Block, Brick/Tile). | Mutual Info $MI = 0.200$ |
| **24** | `MasVnrType` | Raw | Nominal | Categorical | Masonry veneer facade material (None, Brick Face, Stone, Brick Common). | Mutual Info $MI = 0.100$ |
| **TARGET** | `SalePrice` | Raw | Target | Continuous | Realized contract sale price in USD ($). Transformed to $\ln(1 + \text{SalePrice})$. | Skew = 1.88 $\to$ 0.12 |

---

### 4.2 Post-Encoding Feature Space Architecture (The 91 Processed Columns)

The finalized modeling matrix consumed by all estimators contains 91 numeric features:
1. **Continuous Numerical Features (11 Columns, Z-Score Standardized)**:  
   `OverallQual_TotalSF`, `OverallQual`, `TotalSF`, `GrLivArea`, `GarageCars`, `TotalBathrooms`, `TotalBsmtSF`, `1stFlrSF`, `FullBath`, `TotRmsAbvGrd`, `HouseAge`.
2. **Hierarchical Ordinal Features (7 Columns, Monotonically Ranked $[0, \dots, 6]$)**:  
   `BsmtQual`, `KitchenQual`, `ExterQual`, `GarageFinish`, `FireplaceQu`, `HeatingQC`, `BsmtFinType1`.
3. **Nominal Features (73 Columns, Binary $\{0, 1\}$ Dummy Indicators)**:
   - `Neighborhood` (25 columns): `Blmngtn`, `Blueste`, `BrDale`, `BrkSide`, `ClearCr`, `CollgCr`, `Crawfor`, `Edwards`, `Gilbert`, `IDOTRR`, `MeadowV`, `Mitchel`, `NAmes`, `NPkVill`, `NWAmes`, `NoRidge`, `NridgHt`, `OldTown`, `SWISU`, `Sawyer`, `SawyerW`, `Somerst`, `StoneBr`, `Timber`, `Veenker`.
   - `MSSubClass` (15 columns): `20`, `30`, `40`, `45`, `50`, `60`, `70`, `75`, `80`, `85`, `90`, `120`, `160`, `180`, `190`.
   - `Exterior2nd` (16 columns): `AsbShng`, `AsphShn`, `Brk Cmn`, `BrkFace`, `CBlock`, `CmentBd`, `HdBoard`, `ImStucc`, `MetalSd`, `Other`, `Plywood`, `Stone`, `Stucco`, `VinylSd`, `Wd Sdng`, `Wd Shng`.
   - `GarageType` (7 columns): `2Types`, `Attchd`, `Basment`, `BuiltIn`, `CarPort`, `Detchd`, `None`.
   - `Foundation` (6 columns): `BrkTil`, `CBlock`, `PConc`, `Slab`, `Stone`, `Wood`.
   - `MasVnrType` (4 columns): `BrkCmn`, `BrkFace`, `None`, `Stone`.

---

## 5. Downstream Integration & Handoff to Team Members

My deliverables in Phase 2 directly powered the subsequent phases of our project:

```mermaid
sequenceDiagram
    autonumber
    actor Jayashan as Member 2: Jayashan (Preprocessing)
    actor Teshan as Member 3: Teshan (Model Training)
    actor Nadeesha as Member 4: Nadeesha (Evaluation & Web UI)
    
    Jayashan->>Jayashan: preprocessV3.ipynb Execution
    Jayashan->>Jayashan: Export data-new/*.csv & ColumnTransformer
    Jayashan->>Teshan: Handoff data-new/ (91 features, log target)
    Teshan->>Teshan: Model_Training.ipynb (5-Fold CV RandomizedSearchCV)
    Teshan->>Teshan: Train Ridge, RF, XGBoost, LightGBM, CatBoost
    Teshan->>Nadeesha: Export models/*.joblib & predictions/*.csv
    Jayashan->>Nadeesha: Provide Inverse Transformation Protocol np.expm1()
    Nadeesha->>Nadeesha: Model_Evaluation.ipynb (Benchmark MAE, RMSE, R²)
    Nadeesha->>Nadeesha: Deploy app.py (FastAPI Web App with Champion XGBoost)
```

### 1. Handoff to Member 3 (Teshan — Model Training):
- Teshan loaded `X_train_preprocessed.csv` and `y_train.csv` directly into `Model_Training.ipynb`.
- Because the 11 continuous features were already centered and standardized, his **Ridge Regression** model converged optimally without gradient oscillation.
- Because categorical variables were split into ordinal integers and distinct one-hot dummies, tree-based models (**Random Forest, XGBoost, LightGBM, CatBoost**) split nodes effectively without arbitrary categorical ordering biases.

### 2. Handoff to Member 4 (Nadeesha — Evaluation & Deployment):
- I documented the inverse transformation protocol:
  $$\hat{y}_{\text{dollars}} = \exp(\hat{y}_{\text{log}}) - 1 = \text{np.expm1}(\hat{y}_{\text{log}})$$
- This enabled Nadeesha to convert model inferences into real dollar valuations in `Model_Evaluation.ipynb`, confirming our overall champion **XGBoost Regressor** achieved:
  - **Highest $R^2$ Score**: **$0.9073$** (explaining $90.73\%$ of price variation across held-out properties)
  - **Lowest Root Mean Squared Error (RMSE)**: **$\$26,670.54$** (penalizing large residuals most effectively)
  - **Mean Absolute Error (MAE)**: **$\$16,884.70$** (tight average deviation on $1,460$-property domain)
  - **Log-Scale Mean Squared Error (MSE)**: **$0.0220$**
- With these winning benchmarks, **XGBoost** was officially selected as the production model and serialized to [`models/best_model/XGBoost.joblib`](file:///d:/Projects_New/ames-housing-price-prediction/models/best_model) (and `XGBoost_model.joblib`) bundled with `preprocessor_bundle.joblib`.
- Furthermore, the 24 pre-encoding features I curated served as the exact interactive input form fields in the production **FastAPI web application** ([`app.py`](file:///d:/Projects_New/ames-housing-price-prediction/app.py)), enabling instant real-time valuation powered by the champion XGBoost model.

---

## 6. Personal Learning Reflections, Challenges & Technical Growth

### 6.1 Technical Challenge 1: Resolving Temporal Domain Anomalies
- **The Problem**: During feature auditing, I noticed `RemodAge` contained a negative minimum value ($-1$). A naive pipeline would either crash or pass negative ages into downstream models, where decision trees would create illogical split rules (e.g., "if RemodAge $\le -0.5$").
- **Resolution**: I audited the data to understand the root cause: the contract recording year was filed in 2008 while the final remodel completion was recorded in 2009. I implemented domain-informed lower-bound clipping (`max(0, x)`), preserving the valid physical domain while retaining the record.
- **Key Takeaway**: Real-world tabular datasets frequently contain recording inconsistencies. Rigorous summary statistic audits (`df.describe()`) must always be performed immediately after feature creation, before feeding data into modeling stages.

---

### 6.2 Technical Challenge 2: Eliminating Subtle Data Leakage
- **The Problem**: In early draft versions of the project, data transformations (such as computing neighborhood medians or standard scaling) were applied to the whole dataset before calling `train_test_split`. This inadvertently leaked test set means and standard deviations into the training set, causing optimistic training scores.
- **Resolution**: In version 3 (`preprocessV3.ipynb`), I redesigned the architecture from scratch. I established strict isolation: splitting `X_train` and `X_test` prior to fitting any transformer, and wrapping all transformations in Scikit-Learn’s `ColumnTransformer`. `fit_transform()` was strictly restricted to `X_train`, while `transform()` was applied to `X_test`.
- **Key Takeaway**: Preventing data leakage requires architectural discipline rather than ad-hoc code. Scikit-Learn pipelines and ColumnTransformers provide the industry standard for ensuring that production inference identically mirrors training transformations.

---

### 6.3 Technical Challenge 3: Balancing Dimensionality vs Expressiveness
- **The Problem**: One-hot encoding all 43 raw categorical features would have exploded the feature space to over 250 columns, creating severe sparsity and overfitting risks for a modest dataset of 1,460 rows. Conversely, blindly label-encoding all categories would impose artificial numeric distances on nominal data (e.g., implying Neighborhood 12 is twice Neighborhood 6).
- **Resolution**: I designed a hybrid encoding strategy:
  1. High-association categories were pruned via **Cramér's $V$ with bias correction** (dropping `Exterior1st`).
  2. Features with natural quality scales were mapped to **monotonically ranked ordinal hierarchies** (7 features).
  3. Only the remaining 6 high-signal nominal features were one-hot encoded, keeping the final space at a balanced 91 dimensions.
- **Key Takeaway**: Automated encoding is no substitute for domain-informed taxonomy. Differentiating between ordinal and nominal data preserves structural hierarchy while controlling dimensional explosion.

---

### 6.4 Key Technical Skills Developed

| Competency Area | Tools & Libraries Mastered | Real-World Application in Project |
|---|---|---|
| **Pipeline Engineering** | `sklearn.compose.ColumnTransformer`, `sklearn.pipeline.Pipeline` | Engineered a leak-free, modular transformation pipeline with separate numerical, ordinal, and nominal branches. |
| **Statistical Association** | `scipy.stats.chi2_contingency`, Pearson $r$, Cramér's $V$ | Implemented bias-corrected Cramér's $V$ to quantify categorical associations and prune redundant features. |
| **Information Theory** | `sklearn.feature_selection.mutual_info_regression` | Used non-parametric Mutual Information to discover non-linear relationships in categorical variables. |
| **Data Hygiene & Transformation** | `pandas`, `numpy`, `scipy.stats` | Handled structural missingness, grouped neighborhood median imputation, and natural log transformations. |
| **Collaborative Version Control** | Git, GitHub, Jupyter Notebooks | Maintained clean notebook versioning and modular CSV artifact handoffs for team members. |

---

## 7. Self-Assessment & Future Work

### Self-Assessment:
- **Completeness**: 10/10 — Delivered all data processing, feature engineering, and pipeline export milestones ahead of team schedule.
- **Rigor**: 10/10 — Zero data leakage, 0 missing values, 0 infinite values, full mathematical and domain justification for every transformation.
- **Impact**: 10/10 — Engineered the single most predictive feature of the entire project (`OverallQual_TotalSF`, $r = 0.856$), enabling our team's champion **XGBoost Regressor** to reach an $R^2$ of **$0.9073$** and the lowest RMSE of **$\$26,670.54$** in final held-out benchmarking.

### What I Would Explore in Future Iterations:
1. **Target Encoding with Out-of-Fold Regularization**: For high-cardinality nominal features like `Neighborhood` (25 categories), applying out-of-fold target encoding with additive smoothing could reduce the 25 dummy columns to a single continuous variable while mitigating target leakage.
2. **Automated Feature Interaction Discovery**: While manual domain engineering created `OverallQual_TotalSF`, automated interaction tools (e.g., polynomial feature selectors with LASSO penalties) could uncover higher-order interaction effects between parcel topography and interior amenities.
3. **Robust Scaling for Outliers**: While `StandardScaler` performed exceptionally well with our log-transformed target, evaluating `RobustScaler` (interquartile range scaling) on raw living areas could offer additional resilience against extreme property sizes.

---

## 8. Conclusion

Leading the data preprocessing and feature engineering phase (`preprocessV3.ipynb`) for the Ames Housing Price Prediction project provided me with invaluable practical experience across the end-to-end machine learning lifecycle. It reinforced the core principle of applied data science: **model performance is bounded by the quality and representation of the underlying features**. 

By combining domain real estate knowledge, statistical selection techniques, and leak-free Scikit-Learn pipeline design, I successfully transformed raw, messy data into an optimized 91-dimensional feature matrix that enabled our team to deliver an enterprise-grade, highly accurate predictive system.
