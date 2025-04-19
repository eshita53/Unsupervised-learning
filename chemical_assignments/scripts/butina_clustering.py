
import rdkit
from rdkit.Chem import DataStructs
from rdkit.ML.Cluster import Butina
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


class ButinaClustering():

    def __init__(self, finger_print_data, similarity_threshold=.2):
        self.finger_print_data = finger_print_data
        # distThresh: elements within this range of each other are considered to be neighbors
        self.similarity_threshold = similarity_threshold
        self.clusters_ = None
        self.labels_ = None
        self.cluster_stats_ = None
        self.intra_similarities_ = None

    @staticmethod
    def tanimoto_distance_matrix(fp_list):
        """Calculate distance matrix for fingerprint list"""
        dissimilarity_matrix = []
        # we are skipping the first and last items in the list
        # because we don't need to compare them against themselves
        for i in range(1, len(fp_list)):
            similarities = DataStructs.BulkTanimotoSimilarity(
                fp_list[i], fp_list[:i])
            # Since we need a distance matrix, calculate 1-x for every element in similarity matrix
            dissimilarity_matrix.extend([1 - x for x in similarities])
        return dissimilarity_matrix

    def clustering(self):
        """Perform Butina clustering and calculate cluster statistics"""

        distance_matrix = ButinaClustering.tanimoto_distance_matrix(
            self.finger_print_data)
        clusters = Butina.ClusterData(data=distance_matrix,
                                      nPts=len(self.finger_print_data),
                                      distThresh=self.similarity_threshold,
                                      isDistData=True)
        self.clusters_ = sorted(clusters, key=len, reverse=True)
        self.labels_ = self._get_cluster_labels()
        self._calculate_cluster_stats()
        self._calculate_intra_similarities()
        return self

    def _get_cluster_labels(self):
        """Generate cluster labels for each data point"""
        labels = np.zeros(len(self.finger_print_data), dtype=int) - 1
        for cluster_id, cluster in enumerate(self.clusters_):
            for index in cluster:
                labels[index] = cluster_id
        return labels

    def _calculate_cluster_stats(self):
        """Calculate comprehensive statistics about the clusters"""
        cluster_sizes = [len(c) for c in self.clusters_]

        self.cluster_stats_ = {
            'total_clusters': len(self.clusters_),
            'total_compounds': len(self.finger_print_data),
            'singleton_clusters': sum(1 for c in self.clusters_ if len(c) == 1),
            'doubleton_clusters': sum(1 for c in self.clusters_ if len(c) == 2),
            'tripleton_clusters': sum(1 for c in self.clusters_ if len(c) == 3),
            'quadrupleton_clusters': sum(1 for c in self.clusters_ if len(c) == 4),
            'clusters_gt5': sum(1 for c in self.clusters_ if len(c) > 5),
            'clusters_gt25': sum(1 for c in self.clusters_ if len(c) > 25),
            'clusters_gt100': sum(1 for c in self.clusters_ if len(c) > 100),
            'average_cluster_size': np.mean(cluster_sizes),
            'median_cluster_size': np.median(cluster_sizes),
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'size_distribution': cluster_sizes
        }

    def _calculate_intra_similarities(self, min_clusters=2):
        """Calculate intra-cluster Tanimoto similarities"""
        intra_similarities_ = []

        for cluster in self.clusters_:
            if len(cluster) < min_clusters:
                continue  # Skip clusters with defined mninimum cluster number

            fps = [self.finger_print_data[i] for i in cluster]
            similarities = []

            for fp1, fp2 in combinations(fps, 2):
                similarities.append(DataStructs.TanimotoSimilarity(fp1, fp2))

            intra_similarities_.append(similarities)
        
        return intra_similarities_

    def get_similarity_stats(self):
        """Calculate statistics about intra-cluster similarities"""

        if not self.intra_similarities_:
            self.intra_similarities_ = self._calculate_intra_similarities()

        all_similarities = [
            sim for sims in self.intra_similarities_ for sim in sims]

        return {
            'average_similarity': np.mean(all_similarities) if all_similarities else 0,
            'median_similarity': np.median(all_similarities) if all_similarities else 0,
            'min_similarity': np.min(all_similarities) if all_similarities else 0,
            'max_similarity': np.max(all_similarities) if all_similarities else 0,
            'std_deviation': np.std(all_similarities) if all_similarities else 0,
            'similarity_range': np.ptp(all_similarities) if all_similarities else 0,
            'num_similarity_pairs': len(all_similarities),
            'percent_above_threshold': {
                '0.7': np.mean(np.array(all_similarities) >= 0.7) * 100 if all_similarities else 0,
                '0.8': np.mean(np.array(all_similarities) >= 0.8) * 100 if all_similarities else 0,
                '0.9': np.mean(np.array(all_similarities) >= 0.9) * 100 if all_similarities else 0
            }
        }
    
    def plot_intra_cluster_similarity(self, min_clusters, output_file=None):
        """Visualize intra-cluster similarity distribution"""
        intra_similarities_ = self._calculate_intra_similarities(min_clusters=min_clusters)

        fig, ax = plt.subplots(figsize=(15, 5))
        cluster_index = list(range(len(intra_similarities_)))
        ax.set_xlabel("Cluster index")
        ax.set_ylabel("Similarity",)
        ax.set_xticks(cluster_index)
        ax.set_xticklabels(cluster_index)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_yticks(np.arange(0.1, 1.0, 0.1))
        ax.set_title("Intra-cluster Tanimoto similarity", fontsize=13)
        r = ax.violinplot(intra_similarities_, cluster_index,
                          showmeans=True, showmedians=True, showextrema=False)

        # Setting colors
        r["cmeans"].set_color("red")
        for part_name, part in r.items():
            if part_name == 'cmedians':
                part.set_color('blue')

        # Adding legend
        mean_line = Line2D([0], [0], color='red', lw=2, label='Mean')
        median_line = Line2D([0], [0], color='blue', lw=2, label='Median')
        ax.legend(handles=[mean_line, median_line], loc='upper right')

        if output_file:  # if output file is not given, show the plot
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
