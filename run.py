from src.data_preprocessing import load_and_preprocess_data
from src.model import build_model
from src.evaluation import evaluate_model
import joblib
import matplotlib.pyplot as plt
import os


def main():
    # Load data
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess_data()

    # Build model
    model = build_model(input_dim=X_train.shape[1])

    # Train
    history = model.fit(X_train, y_train, epochs=100, verbose=1)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    evaluate_model(model, X_test, y_test, y_pred, feature_names)

    # Save the trained model for future use
    model.save('models/calorie_model.h5')

    # Save the scaler
    joblib.dump(scaler, 'models/scaler.pkl')

    # Show feature weights for inspection
    weights, bias = model.layers[0].get_weights()
    print("\n🎯 Model Weights:")
    for name, weight_val in zip(feature_names, weights.flatten()):
        print(f"{name}: {weight_val:.4f}")
    print(f"Bias: {bias[0]:.4f}")

    # Create the feature of importance plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, weights.flatten())
    plt.xlabel("Weight Value")
    plt.title("Feature Importance (Linear Model Weights)")
    plt.grid(True)
    plt.tight_layout()

    # Save to outputs
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/feature_importance.png")
    plt.close()

if __name__ == '__main__':
    main()
