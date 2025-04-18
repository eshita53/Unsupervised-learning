import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


class UMAPVisualizer:

    """Enhanced UMAP visualizer with dual labeling (clusters or compound types)"""

    def __init__(self, n_neighbors=30, min_dist=0.1, n_components=2, metric='jaccard'):
        """
        Initialize UMAP visualizer with flexible visualization options
        """
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.metric = metric
        self.reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric
        )

    def run_umap(self, fingerprints_matrix, labels=None, types=None, sample_size=None):
        """
        Run UMAP with support for both cluster labels and compound types
        """
        if sample_size and len(fingerprints_matrix) > sample_size:
            if labels is not None:
                sampled_data, sampled_labels, sampled_types = self._stratified_sample(
                    fingerprints_matrix, labels, types, sample_size
                )
            else:
                idx = np.random.choice(
                    len(fingerprints_matrix), sample_size, replace=False)
                sampled_data = fingerprints_matrix[idx]
                sampled_labels = None
                sampled_types = types[idx] if types is not None else None
        else:
            sampled_data = fingerprints_matrix
            sampled_labels = labels
            sampled_types = types

        embedding = self.reducer.fit_transform(sampled_data)
        umap_df = pd.DataFrame(embedding, columns=[
                               f'UMAP{i+1}' for i in range(self.n_components)])

        if sampled_labels is not None:
            umap_df['Cluster'] = sampled_labels
        if sampled_types is not None:
            umap_df['Type'] = sampled_types

        return umap_df

    def show_plot(self, umap_df, color_by='cluster', output_dir=None, title_suffix=""):
        """
        Visualize UMAP results with flexible coloring options
        """
        title = f"UMAP Projection (n_neighbors={self.n_neighbors}, min_dist={self.min_dist})"
        if title_suffix:
            title += f" - {title_suffix}"

        if self.n_components == 2:
            plt.figure(figsize=(12, 10))
            if color_by == 'type' and 'Type' in umap_df:
                palette = {'natural': 'green', 'synthetic': 'blue'}
                sns.scatterplot(
                    x='UMAP1', y='UMAP2',
                    hue='Type',
                    palette=palette,
                    data=umap_df,
                    s=60,
                    alpha=0.7
                )
                plt.title(f"{title} - Colored by Compound Type")
            else:
                sns.scatterplot(
                    x='UMAP1', y='UMAP2',
                    hue='Cluster' if 'Cluster' in umap_df else None,
                    palette='tab10',
                    data=umap_df,
                    s=60,
                    alpha=0.7
                )
                plt.title(f"{title} - Colored by Cluster")

        elif self.n_components == 3:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')

            if color_by == 'type' and 'Type' in umap_df:
                colors = umap_df['Type'].map(
                    {'natural': 'green', 'synthetic': 'blue'})
                ax.scatter(
                    umap_df['UMAP1'], umap_df['UMAP2'], umap_df['UMAP3'],
                    c=colors,
                    s=60
                )
            else:
                ax.scatter(
                    umap_df['UMAP1'], umap_df['UMAP2'], umap_df['UMAP3'],
                    c=umap_df['Cluster'] if 'Cluster' in umap_df else 'b',
                    cmap='Spectral',
                    s=60
                )
            ax.set_xlabel('UMAP1')
            ax.set_ylabel('UMAP2')
            ax.set_zlabel('UMAP3')

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"umap_by_{color_by}.png"
            plt.savefig(os.path.join(output_dir, filename),
                        dpi=300, bbox_inches='tight')

        plt.show()
        plt.close()

    def _stratified_sample(self, data, labels, types, sample_size):
        """Stratified sampling that preserves both cluster and type distributions"""
        unique_labels = np.unique(labels)
        sampled_indices = []

        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            sample_size_per_label = max(
                1, int(sample_size * len(label_indices) / len(labels)))
            sampled_indices.extend(
                np.random.choice(label_indices,
                                 size=min(sample_size_per_label,
                                          len(label_indices)),
                                 replace=False)
            )

        return data[sampled_indices], labels[sampled_indices], types[sampled_indices]


