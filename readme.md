1. Load train.csv
        ↓
2. Separate features and target
   X = features
   y = SalePrice
        ↓
3. Train–Test Split
   X_train, X_test, y_train, y_test
        ↓
4. Data Understanding / EDA
   - data types
   - missing values
   - distributions
   - duplicates
   - outliers
        ↓
5. Missing Value Handling
        ↓
6. Outlier Handling
        ↓
7. Feature Engineering
   - create new features
   - transform features
   - handle skewness where appropriate
        ↓
8. Categorical Encoding
   - One-Hot Encoding
   - Ordinal Encoding where appropriate
        ↓
9. Feature Selection
   - correlation analysis
   - remove redundant/unnecessary features
   - feature importance / selection method
        ↓
10. Feature Scaling
   - StandardScaler / RobustScaler etc.
        ↓
11. Model Training
   model.fit(X_train_processed, y_train)
        ↓
12. Model Evaluation
   X_test_processed + y_test
        ↓
13. Finalize the model
        ↓
14. Apply final pipeline to official test.csv
        ↓
15. Predict SalePrice
        ↓
16. Create submission using sample_submission.csv