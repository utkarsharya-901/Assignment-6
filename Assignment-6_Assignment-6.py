import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# Task 1: Data Collection and Understanding
# ==========================================

# 1. Fetch weather data using the Open-Meteo API
url = "https://api.open-meteo.com/v1/forecast?latitude=28&longitude=77&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m&forecast_days=7"
print("Fetching data from Open-Meteo API...")
response = requests.get(url).json()

# 2. Convert the JSON response into a Pandas DataFrame
df = pd.DataFrame(response['hourly'])

# Create a new column named Weather_Class (Warm >= 25°C, Cool < 25°C)
df['Weather_Class'] = df['temperature_2m'].apply(lambda x: 'Warm' if x >= 25 else 'Cool')

# 3. Display the first five records
print("\n--- First 5 Records ---")
print(df.head())

# ==========================================
# Task 2: Data Preprocessing
# ==========================================

# Check for and drop missing values
df = df.dropna()

# Remove unnecessary columns (e.g., 'time')
X = df[['temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'wind_speed_10m']]

# Encode the target variable (Cool = 0, Warm = 1)
y = df['Weather_Class'].map({'Cool': 0, 'Warm': 1})

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# Task 3: Model Development
# ==========================================

print("\nTraining SVM Model...")
# Build an SVM Classifier using Kernel = RBF
svm_model = SVC(kernel='rbf')

# Train the model and predict the weather class for the test dataset
svm_model.fit(X_train_scaled, y_train)
y_pred = svm_model.predict(X_test_scaled)

# ==========================================
# Task 4: Model Evaluation
# ==========================================

print("\n--- Model Evaluation ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# Task 4 Observations
# ==========================================
print("\n--- Observations ---")
print("1. The SVM classifier with an RBF kernel achieves near-perfect accuracy because the target variable is directly derived from one of the input features (Temperature threshold of 25°C).")
print("2. The confusion matrix displays minimal to no false positives or false negatives, indicating high precision and recall scores across both the 'Warm' and 'Cool' classes.")
print("3. Standardizing the features ensured that variables with larger numerical ranges, like surface pressure, did not disproportionately influence the decision boundary constructed by the RBF kernel.")
