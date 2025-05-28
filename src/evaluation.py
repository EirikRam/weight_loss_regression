import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test, y_pred, feature_names):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MSE: {mse:.2f}")
    print(f"Test R²: {r2:.2f}")

    # Save loss curve and weight chart
    weights, bias = model.layers[0].get_weights()

    os.makedirs('outputs/plots', exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, weights.flatten(), color='skyblue')
    plt.axvline(0, color='gray', linestyle='--')
    plt.title('Feature Weights Learned by Linear Regression Model')
    plt.xlabel('Weight Value')
    plt.tight_layout()
    plt.savefig('outputs/plots/feature_weights.png')
    plt.close()

    # Save metrics
    with open('outputs/metrics.txt', 'w') as f:
        f.write(f"Test MSE: {mse:.2f}\n")
        f.write(f"Test R²: {r2:.2f}\n")

    return mse, r2
