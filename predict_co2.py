import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('data/co2_emissions.csv')

# Prepare features
data = data[['Year', 'Total', 'Gas Fuel', 'Liquid Fuel', 'Solid Fuel']]
data = data.rename(columns={'Total': 'CO2_Emissions'})

X = data[['Gas Fuel', 'Liquid Fuel', 'Solid Fuel']]
y = data['CO2_Emissions']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("R² Score:", r2)

# Visualization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual CO₂ Emissions")
plt.ylabel("Predicted CO₂ Emissions")
plt.title("Actual vs Predicted CO₂ Emissions")
plt.show()

# Save results
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results.to_csv('outputs/model_predictions.csv', index=False)

with open('outputs/evaluation_metrics.txt', 'w') as f:
    f.write(f"MAE: {mae}\nR²: {r2}")
