# Weather Condition Classification using SVM

## Objective
To classify weather conditions into 'Cool' or 'Warm' categories based on meteorological observations collected from the Open-Meteo API using a Support Vector Machine (SVM) classification model.

## API Documentation Link
[Open-Meteo API](https://open-meteo.com/)

## Libraries Used
- `requests`: For fetching data from the API.
- `pandas`: For data manipulation and DataFrame construction.
- `scikit-learn`: For data preprocessing (StandardScaler, train_test_split), model building (SVC), and evaluation metrics.

## Methodology
1. **Data Collection:** Fetched 7-day hourly weather data (Temperature, Relative Humidity, Surface Pressure, and Wind Speed) using the Open-Meteo API.
2. **Data Preprocessing:** Converted JSON responses into a Pandas DataFrame. Handled missing values, created a target variable `Weather_Class` (Warm >= 25°C, Cool < 25°C), encoded the targets, and split the data into 80% training and 20% testing sets. Features were standardized using `StandardScaler`.
3. **Model Development:** Trained an SVM classifier using the Radial Basis Function (RBF) kernel.
4. **Model Evaluation:** Evaluated the model using Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix.

## Results
- **Accuracy:** ~1.0000
- **Precision:** ~1.0000
- **Recall:** ~1.0000
- **F1-Score:** ~1.0000
*The confusion matrix showed highly accurate predictions with negligible misclassifications because the target variable is deterministically created from the temperature feature.*

## Conclusion
This project successfully classified weather as Warm or Cool using Open-Meteo data. Feature scaling was crucial for the SVM model to perform fairly without being biased by features with large numeric scales (like surface pressure). While SVM with an RBF kernel is excellent at capturing complex boundaries, its computational cost on massive datasets remains a standard limitation.
