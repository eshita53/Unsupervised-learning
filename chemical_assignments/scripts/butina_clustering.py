
import rdkit
from rdkit.Chem import DataStructs
from rdkit.ML.Cluster import Butina
import numpy as np
class ButinaClustering():
    
    def __init__(self, finger_print_data, similarity_threshold=.2):
        self.finger_print_data = finger_print_data
        self.similarity_threshold=similarity_threshold # distThresh: elements within this range of each other are considered to be neighbors
        self.clusters_ = None
        
    @staticmethod
    def tanimoto_distance_matrix(fp_list):
        """Calculate distance matrix for fingerprint list"""
        dissimilarity_matrix = []
        # we are skipping the first and last items in the list
        # because we don't need to compare them against themselves        
        for i in range(1, len(fp_list)):
            similarities = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
            # Since we need a distance matrix, calculate 1-x for every element in similarity matrix
            dissimilarity_matrix.extend([1 - x for x in similarities])
        return dissimilarity_matrix

    def clustering(self):
        print(self.finger_print_data)
        distance_matrix = ButinaClustering.tanimoto_distance_matrix(self.finger_print_data)
        cluster = Butina.ClusterData(data= distance_matrix, nPts = len(self.finger_print_data),distThresh= self.similarity_threshold, isDistData=True)
        self.clusters_ = sorted(cluster, key=len, reverse=True)
        
        num_clust1 = sum(1 for c in self.clusters_ if len(c) ==1)
        num_clust5= sum(1 for c in self.clusters_ if len(c) >5)
        num_clust2 = sum(1 for c in self.clusters_ if len(c) ==2)
        num_clust3 = sum(1 for c in self.clusters_ if len(c) ==3)
        num_clust4 = sum(1 for c in self.clusters_ if len(c) ==4)
        num_clust6 = sum(1 for c in self.clusters_ if len(c) >25)
        num_clust7 = sum(1 for c in self.clusters_ if len(c) >100)
        print(f'Number of clusters {len(self.clusters_)}')
        print(f'cluster with only 1 compounds  {num_clust1}')
        print(f'cluster with   2 compounds  {num_clust2}')
        print(f'cluster with  3 compounds  {num_clust3}')
        print(f'cluster with   4 compounds  {num_clust4}')
        print(f'cluster with  more than 5 compounds  {num_clust5}')
        print(f'cluster with more than 25 compounds  {num_clust6}')
        print(f'cluster with more than 25 compounds  {num_clust7}')
        print(self.clusters_)
        return self
    
    def get_cluster_labels(self):
        labels = np.zeros(len(self.finger_print_data), dtype=int) -1
        for cluster_id, cluster in enumerate(self.clusters_):
            # print(cluster)
            for index in cluster:
                # print(cluster_id)
                labels[index] = cluster_id
        # print(labels)
        return labels