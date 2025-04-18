import numpy as np
import pandas as pd
from sklearn.cluster import Birch
import os
from tqdm import tqdm

class BIRCHClustering:
    """Handles BIRCH clustering operations"""
    def __init__(self, threshold=0.2, branching_factor=100, n_clusters=200):
        self.model = Birch(
            threshold=threshold,
            branching_factor=branching_factor,
            n_clusters=n_clusters,
            compute_labels=True
        )
    
    def fit_predict_batches(self, data, batch_size=10000):
        """Fit and predict labels in batches with progress tracking"""
        all_labels = np.empty(len(data), dtype=int)
        
        for i in tqdm(range(0, len(data), batch_size), desc="Clustering"):
            batch = data[i:i + batch_size]
            self.model.partial_fit(batch)
            batch_labels = self.model.predict(batch)
            all_labels[i:i + batch_size] = batch_labels
            
        return all_labels
    
    def get_centroids(self):
        """Return cluster centroids"""
        return self.model.subcluster_centers_





