import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(path='data/processed/cleaned_data.csv'):
    df = pd.read_csv(path)

    # Separate features and target
    X = df.drop(columns=['Calories']).values
    y = df[['Calories']].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    feature_names = df.drop(columns=['Calories']).columns.tolist()

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler
