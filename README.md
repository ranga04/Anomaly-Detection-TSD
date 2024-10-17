# Anomaly-Detection-TSD

Anomaly Detection in Time Series Data Using LSTM with Attention

Overview

This project focuses on detecting anomalies in time series data using Long Short-Term Memory (LSTM) networks with an attention mechanism. The project includes several key elements such as model hyperparameter tuning, advanced evaluation metrics, and anomaly visualization. The objective is to improve anomaly detection by enhancing the model’s ability to focus on relevant parts of the time series through attention mechanisms.

Key Features:
LSTM Model: Uses an LSTM network to capture temporal dependencies in the time series data.
Attention Mechanism: An attention layer is added to improve the model's ability to focus on critical parts of the sequence.
Hyperparameter Tuning: A comprehensive search for optimal hyperparameters, including number of LSTM units, dropout rate, optimizer type, and batch size.
Evaluation Metrics: Includes MSE, precision, recall, F1-score, and ROC-AUC for a detailed performance analysis.
Anomaly Visualization: Anomalies detected by the model are visualized in a plot comparing true values and predicted values.
Installation
This project is implemented using Python and requires the following libraries:

bash
Copy code
pip install numpy pandas matplotlib tensorflow scikit-learn
Requirements:
Python 3.x
TensorFlow 2.x
Scikit-learn
Matplotlib
Data
The project uses synthetic time-series data, with added noise and random anomalies. You can replace this with your own time-series dataset to replicate or extend the work.

Running the Model
To run the model, open the notebook Untitled6.ipynb and follow these steps:

Data Preprocessing: Normalize the data and create time-series sequences.
Model Training: Train the LSTM model with the attention mechanism using the best hyperparameters found via tuning.
Anomaly Detection: Use the trained model to detect anomalies, then visualize the results.
Evaluation: Evaluate the model's performance using MSE, precision, recall, F1-score, and ROC-AUC.
Results
Best Model: The best model achieved a Mean Squared Error (MSE) of 0.5902 with precision of 0.8167 and recall of 0.4455.
Anomaly Detection: The model successfully detected large anomalies in the data and showed areas for improvement in detecting smaller variations.
Future Work
Threshold Optimization: Fine-tuning the threshold for anomaly detection could improve the balance between precision and recall.
Advanced Models: Further experimentation with Variational Autoencoders (VAEs) or transformer-based architectures could enhance performance on more complex datasets.
Real-World Data: Applying the model to real-world datasets such as financial data, IoT sensor data, or healthcare data could extend its applicability.
Contributing
Feel free to fork this repository and submit pull requests if you have any suggestions or improvements!

License
This project is licensed under the MIT License.
