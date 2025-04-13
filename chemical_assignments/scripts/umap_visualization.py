
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def run_umap(fingerprints_matrix, labels, n_neighbors, min_dist, n_components):
    
    """This function is used to run the umap and calll the plot funtion to show the visualisation"""
    
    umap_model = umap.UMAP( n_neighbors=n_neighbors, min_dist= min_dist, n_components=n_components, metric='jaccard') # used the jaccard metric because data is binary
    
    umap_embedding = umap_model.fit_transform(fingerprints_matrix)
    
    umap_df = pd.DataFrame(umap_embedding, columns=[
                           f'UMAP{n}' for n in range(1, int(n_components)+1)])
    umap_df['Label'] = labels
    
    return umap_df, n_neighbors, min_dist, n_components
    # show_umap_plot(umap_df, n_neighbors, min_dist, n_components)


def save_umap_plot(umap_df, n_neighbors, min_dist, n_components, outputfile):
    """ This function is used to visualize the cluster using umap with different parameters and show the plot"""
    
    if n_components == 2:
        plt.figure(figsize=(12, 10))
        sns.scatterplot(x='UMAP1', y='UMAP2', hue='Label',
                        data=umap_df, palette='tab10', s=60, alpha=0.7)
    elif n_components == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(umap_df['UMAP1'], umap_df['UMAP2'],
                   umap_df['UMAP3'], c=umap_df['Label'], cmap='Spectral', s=60)
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_zlabel('UMAP3')

    # Save the plot as an image
    plt.title(f"UMAP Visualization with Butina clustering algorithm (n_neighbors={n_neighbors}, min_dist={min_dist}, n_components={n_components})")
    plt.savefig(outputfile)
    plt.close()