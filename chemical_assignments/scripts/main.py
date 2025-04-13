from data_processing import Preprocessing
from butina_clustering import ButinaClustering
from umap_visualization import run_umap, save_umap_plot
from analyzing_mixed_cluster import analyze_mixed_clusters
from rdkit.Chem import AllChem
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 

def preprocessin_ongo(df, fingetprint_type):
    p_df = Preprocessing(df)
    p_df.remove_invalid_smile()
    p_df.convert_smile_to_mol()
    p_df.finger_print_add(fingetprint_type)
    fp_df = p_df.get_fp_csmile_df() 
    chem_prop_df = p_df.get_mole_prop_df()
    return p_df, fp_df, chem_prop_df


def get_config(file_name):
    '''This is the configuaration file'''
    with open(file_name, 'r', encoding='UTF-8') as stream:
        config = yaml.safe_load(stream)
    return config

def main():

    config = get_config('config.yaml')
    synthetic_df = pd.read_csv(config['synthetic'], delimiter = '\t')
    natural_df = pd.read_csv(config['natural'], delimiter = '\t')
    umap_image =config['umap_plot']
    
    os.makedirs(umap_image, exist_ok=True)

    syn_processed_df, syn_fp_df, syn_chem_prop_df = preprocessin_ongo(synthetic_df,'morgan')
    nat_processed_df, nat_fp_df, nat_chem_prop_df = preprocessin_ongo(synthetic_df,'morgan')
    
    umap_file = f"{umap_image}umap_plot.png"
    
    # For analysis we need to combine all the fingerprints of natural and synthetic
    all_fps = np.vstack([ nat_fp_df.iloc[:,3:], syn_fp_df.iloc[:,3:]])
    
    all_fps = [AllChem.DataStructs.CreateFromBitString(''.join(map(str, row))) for row in all_fps]

    butina = ButinaClustering(all_fps)
    butina.clustering()
    print(len(butina.clusters_))
    clusters_label = butina.get_cluster_labels()
    umap_df, n_neighbors, min_dist, n_components = run_umap(all_fps, clusters_label, 50, .2, 2)
    
    save_umap_plot(umap_df, n_neighbors, min_dist, n_components, umap_file)

    visualize_df = pd.DataFrame({
        'umap_1': umap_df.iloc[:,0],
        'umap_2': umap_df.iloc[:,1],
        'cluster_label': umap_df.iloc[:,2],
        'compound_type': ['synthetic'] * len(syn_fp_df) + ['natural'] * len(nat_fp_df)  
    })
    
    print(visualize_df)
    
    syn_fp_df['clusters_label'] = clusters_label[len(nat_fp_df):]
    nat_fp_df['clusters_label'] =  clusters_label[:len(nat_fp_df)]
    synthetics = list(set(syn_fp_df['clusters_label']))
    natural = list(set(nat_fp_df['clusters_label']))
    
    
    results= analyze_mixed_clusters(syn_fp_df,nat_fp_df, 'clusters_label')

    mixed_clusters = list(results['cluster'].values)  # List of cluster IDs with both types
    print(set(mixed_clusters))

    plt.figure(figsize=(12, 10))

    for i, cluster_id in enumerate(mixed_clusters):
        # synthetic compounds in this cluster
        synthetic_mask = (visualize_df['compound_type'] == 'synthetic') & (visualize_df['cluster_label'] == cluster_id) 
        # natural compounds in this cluster
        natural_mask = (visualize_df['compound_type'] == 'natural') & (visualize_df['cluster_label'] == cluster_id)
        
        # Plot with the same color but different markers
        plt.scatter(visualize_df.loc[synthetic_mask, 'umap_1'], visualize_df.loc[synthetic_mask, 'umap_2'], 
                marker='o', label=f'Synthetic (Cluster {cluster_id})', alpha=0.7)
        plt.scatter(visualize_df.loc[natural_mask, 'umap_1'], visualize_df.loc[natural_mask, 'umap_2'], 
                marker='x', label=f'Natural (Cluster {cluster_id})', alpha=0.7)

    # Plotting non-mixed clusters in gray
    in_mixed_clusters = visualize_df['cluster_label'].isin(mixed_clusters)
    non_mixed_mask = ~in_mixed_clusters
    
    # non_mixed_mask = ~visualize_df['cluster_label'].isin(mixed_clusters)
    plt.scatter(visualize_df.loc[non_mixed_mask, 'umap_1'], visualize_df.loc[non_mixed_mask, 'umap_2'], color='gray', alpha=0.3, label='Non-mixed clusters')
    plt.title('Mixed Clusters Visualization (umap)')
    plt.xlabel('Umap Component 1')
    plt.ylabel('Umap Component 2')
    plt.legend()
    plt.savefig('mixed_clusters_tsne.png')
    plt.show()

if __name__ == "__main__":
    main()