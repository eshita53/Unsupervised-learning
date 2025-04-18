# Anomaly Detection
In this assignment, we studied data from pump sensors to build a model that can detect abnormal behavior. These anomaly detection models help spot anything that doesn't look normal in the data. This kind of analysis is important because it helps find problems in the system early and prevents future issues.

## Dataset

We have used pump sensor data from [Kaggle](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data). The data includes readings from 52 sensors, recorded every minute from April 1st, 2018 to August 31st, 2018. The pump data also includes machine statuses that show at any minute whether the pump was working normally, was broken, or was in the process of recovering from a breakdown. We analyzed this information and proposed an anomaly detection model.

## File and Directories

### Python Files
- **anomaly.py** - Helper class to train and test anomaly detection models, along utility function to plot data and visualize the results. 

### Notebooks
- **best-model.ipynb** - Notebook for analysis and finding the best anomaly detection model 
- **plot-analysis.ipynb** - This notebook is used for plotting and analyzing the results of anomaly detection. The main goal of the analysis is to identify unreliable sensors and compare the performance of different anomaly detection algorithms.

