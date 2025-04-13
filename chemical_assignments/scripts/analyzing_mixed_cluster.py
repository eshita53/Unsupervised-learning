import pandas as pd
def analyze_mixed_clusters(syn_fp_df, nat_fp_df, cluster_col):
    synthetics = set(syn_fp_df[cluster_col])
    natural = set(nat_fp_df[cluster_col])
    # print(len(synthetics))
    # print(len(natural))
    mixed_clusters = synthetics.intersection(natural)
    results = []
    for cluster in mixed_clusters:
        synthetic_in_cluster = syn_fp_df[syn_fp_df[cluster_col] == cluster]
        natural_in_cluster = nat_fp_df[nat_fp_df[cluster_col] == cluster]
        
        results.append({
            'cluster': cluster,
            'synthetic_count': len(synthetic_in_cluster),
            'natural_count': len(natural_in_cluster),
            'synthetic_compounds': synthetic_in_cluster['compound_name'].tolist(),
            'natural_compounds': natural_in_cluster['compound_name'].tolist()
        })
    
    return pd.DataFrame(results)