import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# Load dataset
df = pd.read_csv('fifa_players.csv')

# Simulate winning_rate (replace with real data if available)
np.random.seed(42)
df['winning_rate'] = (
    df['overall_rating'] * 0.3 +
    df['potential'] * 0.2 +
    df['stamina'] * 0.1 +
    df['composure'] * 0.1 +
    df['vision'] * 0.05 +
    df['dribbling'] * 0.05 +
    df['short_passing'] * 0.05 +
    df['reactions'] * 0.05 +
    df['acceleration'] * 0.05 +
    df['ball_control'] * 0.05
) / 100

df['winning_rate'] += np.random.normal(0, 0.03, size=len(df))
df['winning_rate'] = df['winning_rate'].clip(0, 1)

# Define features
features = [
    'overall_rating', 'potential', 'stamina', 'composure', 'vision',
    'dribbling', 'short_passing', 'ball_control', 'acceleration',
    'sprint_speed', 'reactions', 'shot_power', 'agility', 'strength'
]

# Drop rows with missing values in selected columns
df = df.dropna(subset=features)

# Features and target
X = df[features]
y = df['winning_rate']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model
model = XGBRegressor(n_estimators=250, learning_rate=0.08, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# Save model if good
if r2 >= 0.8:
    joblib.dump(model, 'winning_model.pkl')
    joblib.dump(features, 'features.pkl')
    print("✅ Model saved successfully.")
else:
    print("❌ R² score < 0.8. Try tuning or adding more data.")
