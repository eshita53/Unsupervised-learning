import numpy as np
import pandas as pd
import os
import yaml
from scripts.data_processing import Preprocessing
from scripts.birch_cluster_class import BIRCHClustering
from scripts.umap_visualizer import UMAPVisualizer
from scripts.analyzing_mixed_cluster import visualize_mixed_clusters

def get_config(file_name):
    """This is the configuration file"""
    with open(file_name, 'r', encoding='UTF-8') as stream:
        config = yaml.safe_load(stream)
    return config

def analyze_with_groupby(combined_df):
    """Using pandas' built-in groupby operations for grouping the mixed clusters
    Used group as our dataset is big """
    cluster_stats = (
        combined_df
        .groupby(['cluster_label', 'compound_type'])
        .size()
        .unstack(fill_value=0) # Pivots the table to show compound types as columns for easy comparison
    )
    
    mixed_clusters = cluster_stats[cluster_stats.gt(0).all(axis=1)]
    return mixed_clusters.sort_values(by=['synthetic', 'natural'], ascending=False)


def main():
    config = get_config('../config.yaml')  
    
    # Load and preprocess data
    natural_df = pd.read_csv(config['natural'], sep='\t', nrows=100000)
    synthetic_df = pd.read_csv(config['synthetic'], sep='\t')
    
    nat_process = Preprocessing(natural_df)
    nat_feat_df = nat_process.get_features('morgan')
    syn_process = Preprocessing(synthetic_df)
    syn_feat_df = syn_process.get_features('morgan')
    
    comb_fet = pd.concat([syn_feat_df, nat_feat_df], ignore_index=True, axis=0)
    comb_fet = comb_fet.rename(columns={'name': 'compound_name'})
    # both synthetic and natural fingerprint list
    fingerprint_list = comb_fet.iloc[:,4:1024].values 
    comb_fet = comb_fet.drop(columns=['mol'])
    
    # Cluster with BIRCH
    birch = BIRCHClustering()
    comb_fet['cluster_label'] = birch.fit_predict_batches(fingerprint_list)
    comb_fet['compound_type'] = ['synthetic']*len(syn_feat_df) + ['natural']*len(nat_feat_df)
        
    # Saving results in case we require it in future for further analysis
    output_dir = '../cluster_results/Birch_cluster'
    os.makedirs(output_dir, exist_ok=True)
    comb_fet.to_parquet(f'{output_dir}/clustered_compounds.parquet', compression='gzip')
    np.save(f'{output_dir}/cluster_centroids.npy', birch.get_centroids())

    
    # Visualize with UMAP
    visualizer = UMAPVisualizer(n_neighbors=30, min_dist=0.1)
    
    umap_df = visualizer.run_umap(fingerprint_list, labels=comb_fet['cluster_label'].values, types=comb_fet['compound_type'].values)
    visualizer.show_plot(umap_df, output_dir=output_dir, color_by='cluster')
    visualizer.show_plot(umap_df, output_dir=output_dir, color_by='type')
    
    
    # mixed cluster analysis
    mixed_clusters = analyze_with_groupby(comb_fet)
    visualize_mixed_clusters(
        combined_df=comb_fet,
        mixed_stats=mixed_clusters,
        n_samples=50000,
        output_dir="../cluster_results"
    )
    
    print(f"Found {len(mixed_clusters)} mixed clusters")


if __name__ == '__main__':
    main()


