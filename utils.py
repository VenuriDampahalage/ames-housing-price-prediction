import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def setup_directories():
    for folder in ['models', 'predictions', 'visualizations']:
        os.makedirs(folder, exist_ok=True)

def save_and_evaluate(model_name, best_model, X_test, y_test, y_pred, feature_names):
    setup_directories()
    
    # 1. Save Model & Predictions for Nadeesha
    joblib.dump(best_model, f'models/{model_name}_model.joblib')
    pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).to_csv(f'predictions/{model_name}_preds.csv', index=False)
    print(f"[{model_name}] Saved model and predictions.")

    # 2. Actual vs Predicted Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='#2ab7ca')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f'{model_name}: Actual vs Predicted Prices')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.tight_layout()
    plt.savefig(f'visualizations/{model_name}_actual_vs_pred.png')
    plt.close()

    # 3. Residual Distribution Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='#fe4a90')
    plt.title(f'{model_name}: Residual Distribution')
    plt.xlabel('Residuals (Actual - Predicted)')
    plt.tight_layout()
    plt.savefig(f'visualizations/{model_name}_residuals.png')
    plt.close()

    # 4. Feature Importance Plot
    importances = None
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = np.abs(best_model.coef_) # Use absolute weights for linear models
        
    if importances is not None:
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')
        plt.title(f'{model_name}: Top 10 Feature Importances')
        plt.tight_layout()
        plt.savefig(f'visualizations/{model_name}_feature_importance.png')
        plt.close()
        
    print(f"[{model_name}] Visualizations saved to /visualizations folder.\n")