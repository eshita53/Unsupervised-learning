from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc


class Anomaly:

    def __init__(self, df) -> None:

        self.df = df
        self.df['timestamp'] = pd.to_datetime(df['timestamp'])
        self.df = self.df.set_index('timestamp')
        self.dataframe_to_nd_array()
        
    def dataframe_to_nd_array(self):
        
        self.df = self.drop_nan_columns()
        m, n = self.df.shape
        X = self.df.iloc[:,:n-1] # ignore machine status columns
        X = X.fillna(X.mean()) # we can use last point as missing values
        self.X = X 
        print(self.X)
        

    def anomaly_detection_with_train_test(self, model, X_train, X_test, local_outlier = False):

        """Anomaly detection for train test"""        
        
        scaler=StandardScaler()
        scaler.fit(X_train)
        transformed_train_X = scaler.transform(X_train)
        transformed_test_X = scaler.transform(X_test)
        
        if local_outlier:
            y_pred = model.fit_predict(transformed_test_X)
        else:
            y_pred = model.fit(transformed_train_X).predict(transformed_test_X)

        return y_pred
    
    
    
    
    def anomaly_detection(self, model, local_outlier = False):

        """Anomaly detection for complete dataframe"""

        # dataframe = self.drop_nan_columns()

        # m, n = dataframe.shape
        # X = dataframe.iloc[:,:n-1] # ignore machine status columns
        # X.fillna(X.mean()) # we can use last point as missing values
        

        scaler=StandardScaler()
        X = scaler.fit_transform(self.X)

        if local_outlier:
            y_pred = model.fit_predict(X)
        else:
            y_pred = model.fit(X).predict(X)

        return y_pred

    def percantage_missing_values(self):

        """Find missing values for each column"""

        return (self.df.isna().mean() * 100).to_dict()


    def split_train_test(self, size=0.2):

        """Split dataframe into train-test splits"""

        # dataframe = self.drop_nan_columns()
        # m, n = dataframe.shape
        # X = dataframe.iloc[:,:n-1] 
        # X = X.fillna(X.mean()) 

        scaler=StandardScaler()
        X = scaler.fit_transform(self.X)
        # y = dataframe['machine_status'].apply(lambda x: 1 if x=="NORMAL" else -1)
        y = self.return_target()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)

        return X_train, X_test, y_train, y_test

    def find_best_params(self, model, X, y, param_grid, cv=3):

        """find best parameters of a model """
        
        def custom_scorer(model, X, y):

            y_pred = model.predict(X)

            return f1_score(y, y_pred)

        grid_search = GridSearchCV(model, param_grid, scoring=custom_scorer, cv=cv, n_jobs=-1)
        grid_search.fit(X, y)

        return grid_search.best_params_


    def drop_nan_columns(self, nan_threshold = 34):
        """ Drop columns if percentage of nan values of a colums is more than the threshold values"""
        cols = [k for k,v in self.percantage_missing_values().items() if v>=nan_threshold]
        self.removed_cols = cols
        return self.df.drop(columns=cols)

    def scale(self):

        """ Scale columns before ploting"""

        scaler=StandardScaler()
        dataframe = self.drop_nan_columns()
        sensors_list = [cols for cols in dataframe.columns if cols.startswith("sensor")]
        scaled_df = pd.DataFrame(scaler.fit_transform(dataframe[sensors_list]), columns=sensors_list, index=dataframe.index)
        scaled_df['machine_status'] = dataframe['machine_status']

        return scaled_df

    
    def plot_sensor(self, column = None, preds=None):

        """ Plot sensors"""
        
        scaler=StandardScaler()
        plot = plt.figure(figsize=(8,4))
        scaled_df = self.scale()

        if column is None:
            sensor = 'sensor_04'
            columns = scaled_df.columns
            columns = columns[:-1]
        elif isinstance(column, list):
            columns = column
            sensor = columns[-1]
        elif isinstance(column, str):
            columns = [column]
            sensor = column

        for col in columns:
            if col in scaled_df.columns:
                plot = plt.plot(scaled_df[col], color='grey')
            
        if preds is not None:
            scaled_df['algorithm'] = preds
            anomalies = scaled_df[scaled_df['algorithm']==-1]
            plot = plt.plot(anomalies[sensor], linestyle='none', marker='X', color='blue', markersize=3, label = 'anomalies')   

        broken_rows = scaled_df[scaled_df['machine_status']=='BROKEN']
        recovery_rows = scaled_df[scaled_df['machine_status']=='RECOVERING']

        plot = plt.plot(recovery_rows[sensor], linestyle='none', marker='o', color='yellow', markersize=5, label='recovering')
        plot = plt.plot(broken_rows[sensor], linestyle='none', marker='X', color='red', markersize=7, label='broken')

        if isinstance(column, str):
            plot = plt.title(column)
            
        plot = plt.legend()
        plt.show()

    def return_target(self):
        target = self.df['machine_status'].apply(lambda x: 1 if x=="NORMAL" else -1)
        return target
        
    def evaluation(self, y_pred, target = None):

        """Evaluate models and print reports"""

        if target is None:
            target = self.df['machine_status'].apply(lambda x: 1 if x=="NORMAL" else -1)
        
        print(classification_report(target, y_pred,labels=[-1,1], target_names=['Anomaly', 'Normal']))
        matrix =  confusion_matrix(target, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=['Anomaly', 'Normal'])
        disp.plot()

    def return_tpr_fpr_auc(self,model, X_train, y_test, X_test = None):
        """when X_test is none. That means no split has been done. X_train and y_test are the unspli
        ted data"""
       
        if X_test is not None:
            model.fit(X_train)
            y_pred = model.predict(X_test)
        else:
            model.fit(X_train)
            y_pred = model.predict(X_train)
        
        if str(model).startswith('LocalOutlierFactor'):
            if X_test is not None:
                y_pred = model.fit_predict(X_test)
            else:
                y_pred = model.fit_predict(X_train)
            y_scores = - model.negative_outlier_factor_
        else:
            if X_test is not None:
                y_scores = model.decision_function(X_test)

            else:
                y_scores = model.decision_function(X_train)
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr,roc_auc
    
    def plot_auc(self,model_name, fpr, tpr,roc_auc):
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'AUC Curve {model_name}')
        plt.legend(loc="lower right")
        plt.show()


    def before_after_auc_plot(self,model_name, fpr1, tpr1,roc_auc1,fpr2, tpr2,roc_auc2 ):
    
        plt.figure(figsize=(8, 6))
        plt.plot(fpr1, tpr1, color='darkorange', 
                    label=f'Before cleaning the dataset(AUC = {roc_auc1:.2f})')
        plt.plot(fpr2, tpr2, color='blue', 
                    label=f'After cleaning the dataset(AUC = {roc_auc2:.2f})')
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"Before vs. After Data Cleaning: AUC Comparison for {model_name}")
        plt.legend(loc="lower right")
        plt.show()

