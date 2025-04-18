import umap.umap_ as umap
from rdkit.Chem import Draw
from rdkit import Chem
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def visualize_mixed_clusters(combined_df, mixed_stats, n_samples=50000, output_dir="results"):
    """
    Visualize mixed clusters from groupby analysis using UMAP
    """
    # Visualizing top 10 mixed clusters
    top_clusters = mixed_stats.head(10).index

    sample_df = (
        combined_df[combined_df['cluster_label'].isin(top_clusters)]
        .groupby(['cluster_label', 'compound_type'])
        .apply(lambda x: x.sample(min(len(x), n_samples//10)))
        .reset_index(drop=True)
    )

    # Run UMAP only on fingerprint columns
    # used the jaccard metric because data is binary
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='jaccard')
    embedding = reducer.fit_transform(sample_df.iloc[:, 4:1024])

    plt.figure(figsize=(14, 10))
    scatter = sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=sample_df['cluster_label'].astype(str),
        style=sample_df['compound_type'],
        palette='tab20',
        s=30,
        alpha=0.7
    )

    plt.title(f"Top {len(top_clusters)} Mixed Clusters (n={len(sample_df)})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/mixed_clusters_umap.png",
                bbox_inches='tight', dpi=300)
    plt.close()

    # Return coordinates for further analysis
    result_df = pd.DataFrame({
        'UMAP1': embedding[:, 0],
        'UMAP2': embedding[:, 1],
        'cluster': sample_df['cluster_label'],
        'type': sample_df['compound_type'],
        **{col: sample_df[col] for col in ['compound_name', 'canonical_smiles']}
    })
    result_df.to_parquet(f"{output_dir}/umap_coordinates.parquet")
    return result_df


def draw_cluster_mol(butina_clusters, mixed_clusters, combined_df, output_dir=None):
    
    butina_mixed_cluster = []
    for cluster in set(mixed_clusters['cluster'].values.tolist()):
        butina_mixed_cluster.append(butina_clusters[cluster])
        
    compounds = combined_df['canonical_smiles'].apply(Chem.MolFromSmiles)
    compound_name = combined_df['compound_name']
    nat_df = combined_df[combined_df['compound_type'] == 'natural']
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    for cluster_id, cluster in enumerate(butina_mixed_cluster):
        molecules = [compounds[i] for i in cluster]
        legends = []
        for i in cluster:
            name = compound_name[i]
            if i < len(nat_df):  # Natural
                legends.append(f"Natural: {name}")
            else:  # Synthetic
                legends.append(f"Synthetic: {name}")
        img = Draw.MolsToGridImage(molecules, legends=legends, returnPNG=False)
        filename=f"cluster_{cluster_id}.png"
        img.save(os.path.join(output_dir, filename))
        
def plot_property_heatmap(syn_fp_df, nat_fp_df, cluster_id, syn_chem_df, nat_chem_df, properties):
    synthetic_cluster =  syn_fp_df[syn_fp_df['clusters_label'] == cluster_id]
    natural_cluster =  nat_fp_df[nat_fp_df['clusters_label'] == cluster_id]
    
    synthetic_cluster[properties] = syn_chem_df[syn_chem_df['compound_name'].isin(synthetic_cluster['compound_name'])][properties].values
    natural_cluster[properties] = nat_chem_df[nat_chem_df['compound_name'].isin(natural_cluster['compound_name'])][properties].values
    data = []
    for prop in properties:
        syn_avg = synthetic_cluster[prop].mean()
        nat_avg = natural_cluster[prop].mean()
        data.append([syn_avg,nat_avg])
        
    heatmap_df = pd.DataFrame(data, index=properties, columns=['Synthetic', 'Natural'])

    plt.tight_layout()
    plt.show()
    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_df, annot=True, cmap='coolwarm', center=0)
    plt.title(f'Property Comparison for Cluster {cluster_id}')
    plt.tight_layout()
    plt.savefig(f'cluster_{cluster_id}_properties.png', dpi=300)
    plt.show()
    t_stat, p_value = stats.ttest_ind(natural_cluster['logP'], synthetic_cluster['logP'])
    print(p_value)