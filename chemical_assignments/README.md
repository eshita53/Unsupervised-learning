# Identifying Biological Substitutes for Synthetic Compounds Using Clustering
Synthetic product is harmful for the health and environment. People are looking forward to substitute the synthetic product with the natural product. In this assignment, we analyzed two clustering algorithms to  
identify the natural compounds structurally similar to synthetic compounds for potential substitution.

## Files and Directories

### Configuration Files  
- **config.yaml** – Specifies raw dataset paths, and directories required for the project.  

### Python Files  

- **scripts/analyzer_mixed_cluster.py**: Helper functions for visualizing and analyzing mixed clusters
- **scripts/birch_cluster_class.py**: Class for handling BIRCH clustering
- **scripts/birch.py**: Script for full clustering workflow with BIRCH including data loading, processing and clustering
- **scripts/butina_clustering.py**: Class for handling Butina clustering
- **scripts/data_processing.py**: Data and Feature Processing script
- **scripts/main.py**: 
- **umap_visualizer.py**: Script for UMAP visualization

### Notebooks

- **data.ipynb**: Full clustering workflow with Butina including data loading, processing and clustering

### Other Directories
- **logs/** – Keeps track of workflow execution and errors.
- **cluster_results/** - Save all figures and results for clustering

## Installation Instructions
**Clone the Repository and Set Up Dependencies**: Follow [these instructions](https://github.com/eshita53/Unsupervised-learning/blob/main/README.md#installation) to clone the repository and set up the dependencies.
