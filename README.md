# Automotive Fault Detection System

## 1. Overview
This project presents a machine learning solution designed to detect specific automotive faults from vehicle scanner data. The system is built to identify anomalies in real-time sensor readings, with a primary focus on detecting a rich air-fuel mixture at idle and a low battery voltage condition, as specified in the technical challenge.  
The model learns what constitutes "normal" vehicle behavior from a provided set of reference logs and then uses this knowledge to flag any significant deviations in new data.
One could tackle this from a supervised ML standpoint, but this would potentially lead to a "spurious correlation" scenario, where the system finds a simple pattern and overfits that patterns as a lazy way to minimize the error.

## 2. Overall Approach & Methodology
The solution here uses a hybrid approach that combines unsupervised machine learning for anomaly detection with domain-specific rules for precise diagnosis. This creates a system that is both robust and interpretable.

### a. Unsupervised Anomaly Detection
The way I tackled this is, instead of traning the model to learn how a "fault" looks like, I train it exclusively on data from healthy vehicles. This is an unsupervised approach where the model builds a comprehensive understanding of normal operating parameters. Any data point that falls outside this learned "normal" boundary is flagged as a potential fault or anomaly.

### b. Model: Isolation Forest
The core of the system is an IsolationForest algorithm. This model is highly effective for anomaly detection because it works by "isolating" outliers in the data. It's computationally efficient and performs well even with high-dimensional data, making it a perfect fit for complex sensor logs.

### c. Critical Insight: Contextual Filtering
A key to the model's success is its focus. The system doesn't try to analyze every possible driving condition at once. Instead, it applies a contextual filter, narrowing its analysis to the specific operating state where the target fault occurs: idle or low-load conditions (e.g., Engine RPM <= 1100 and Calculated Load <= 35%). This strategic focus dramatically improves accuracy by eliminating the noise and variability of normal driving.

### d. Hybrid Diagnosis
The system operates in two stages:
1. **Detect**: The IsolationForest model first determines if a scan is anomalous based on its statistical properties.  
2. **Diagnose**: If an anomaly is detected, a secondary layer of rule-based logic checks specific sensor values (like STFT, O2 sensor voltage, and battery voltage) to provide a clear, human-readable diagnosis (e.g., "Rich air-fuel mixture detected").

## 3. How to Run the Solution & Use the Interface
The entire solution is contained within a single Python script. This script serves as a complete application that handles training, evaluation, and prediction.

### Prerequisites
To run this, ensure you have Python installed with the following libraries:
- pandas
- scikit-learn
- joblib

You can install them using pip:
```bash
pip install pandas scikit-learn joblib
```

### Directory Structure
For the script to run correctly, your project folder should be organized as follows:
```
/your_project_folder/
|
|-- automotive_fault_detector.py  <-- The Python script
|
|-- /datasets/
|   |-- fault_example.csv
|   |-- ... (all *REFERENCIA* files) ...
|
|-- /models/  <-- This folder will be created automatically
```

### Running the Full Pipeline (Training & Prediction)
To run the entire process—from data loading and cleaning to model training, evaluation, and demonstration—simply execute the script from your terminal:
```bash
python automotive_fault_detector.py
```

This single command will:
1. Load and process all data from the `/datasets/` folder.
2. Train the IsolationForest model.
3. Optimize the decision threshold to maximize the F1-Score.
4. Evaluate the model and print a classification report and confusion matrix.
5. Save the trained model pipeline to `/models/automotive_fault_pipeline_v7_final.pkl`.
6. Run a demonstration phase that shows how to use the prediction interface with several test cases.

### Using the Prediction Interface
The script already contains a `FaultDetectionInterface` class designed for making predictions. You can easily adapt the code from the "DEMONSTRATION PHASE" in the `main()` function to build your own application (e.g., a web API, a desktop tool).  
Here is how you would use it to predict a new scan:
```python
# Assuming you have a trained model file
import joblib

# Load the detector (this contains the trained model and all configurations)
detector = joblib.load('models/automotive_fault_pipeline_v7_final.pkl')
interface = FaultDetectionInterface(detector)

# Create a dictionary with the new scan data
faulty_scan_data = {
    'Temperatura do líquido de arrefecimento do motor - CTS': '85 °C',
    'Carga calculada do motor': '25 %',
    'Rotação do motor - RPM': '900 rpm',
    'Sonda lambda - Banco 1, sensor 1': '0.85 V',
    'Ajuste de combustível de curto prazo - Banco 1, sensor 1': '-12 %',
    'Pressão no coletor de admissão - MAP': '450 mbar',
    'Tensão do módulo': '11.0 V',
}

# Get the prediction
result = interface.predict_single_scan(faulty_scan_data)

# Print the result
print(result)
# Expected Output: {'fault_detected': True, ..., 'status': '⚠️ Fault detected', 'fault_details': [...]}
```

##4 . Challenges and Future Improvements

### Challenges Faced & Solutions
- **Challenge**: The high variability of sensor data across different driving conditions (e.g., accelerating vs. idling).  
  **Solution**: Implemented contextual filtering to focus the model exclusively on idle/low-load states, which stabilized the baseline and made anomalies much clearer.

- **Challenge**: The extreme class imbalance (thousands of normal samples vs. very few fault samples).  
  **Solution**: Adopted an unsupervised anomaly detection approach. Crucially, the decision threshold was not fixed but was optimized by maximizing the F1-Score, which is a best practice for finding a good balance between precision and recall on imbalanced datasets.

### Future Improvements
- **REST API**: Wrap the `FaultDetectionInterface` in a simple web framework like Flask or FastAPI to create a REST API for easy integration with other services.
- **Expanded Contexts**: Train separate models for different operating contexts (e.g., a model for "highway cruising," another for "cold starts") to expand the system's diagnostic capabilities.
- **Monitoring & Retraining**: Implement a logging mechanism to store predictions and user feedback. This data can be used to monitor the model's performance in production and to periodically retrain it to adapt to new data patterns.