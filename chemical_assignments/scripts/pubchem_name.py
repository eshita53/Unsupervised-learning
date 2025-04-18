import requests
import time
import pandas as pd

def get_cids_from_smiles_batch(smiles_batch):
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/cids/JSON"
    data = '\n'.join(smiles_batch)
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(url, data=f"smiles={data}", headers=headers)

    if response.status_code == 200:
        try:
            return response.json()["IdentifierList"]["CID"]
        except KeyError:
            return []
    if response.status_code != 200:
        print("Batch failed:", smiles_batch[:5])
    else:
        return []


def get_names_from_cids(cid_list):
    names = []
    for i in range(0, len(cid_list), 100):
        batch = cid_list[i:i+100]
        cid_str = ",".join(map(str, batch))
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/property/IUPACName/JSON"
        r = requests.get(url)
        if r.status_code == 200:
            try:
                data = r.json()['PropertyTable']['Properties']
                names.extend([(entry['CID'], entry.get('IUPACName', 'N/A')) for entry in data])
            except:
                pass
        time.sleep(1)
    return names


def batch_lookup(smiles_list, batch_size=5000):
    cid_results = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        cids = get_cids_from_smiles_batch(batch)
        cid_results.extend(cids)
        print(f"Processed batch {i // batch_size + 1}, got {len(cids)} CIDs")
        time.sleep(2)  # avoid throttling
    return cid_results
def main():
    # df = pd.read_csv('natural_com_fingerprint.csv')
    smiles_list = ['CO[C@@]1(NC(=O)C(C(=O)[O-])C2=CC=C(O)C=C2)C(=O)N2C(C(=O)[O-])=C(CSC3=NN=NN3C)CO[C@@H]21.[Na+].[Na+]',
  'CCCCCC/C=C\\CCCCCCCCCC(=O)N[C@H]1[C@H](OC[C@H]2O[C@H](OP(=O)([O-])[O-])[C@H](NC(=O)CC(=O)CCCCCCCCCCC)[C@@H](OCCCCCCCCCC)[C@@H]2O)O[C@H](COC)[C@@H](OP(=O)([O-])[O-])[C@@H]1OCC[C@@H](CCCCCCC)OC.[Na+].[Na+].[Na+].[Na+]',
  'O=C([O-])C1=C(CSC2=NN=NN2CS(=O)(=O)O)CS[C@@H]2[C@H](NC(=O)[C@H](O)C3=CC=CC=C3)C(=O)N12.[Na+]',
  'CC1(C)C(=CC=CC=CC=CC2=[N+](CCCCS(=O)(=O)O)C3=CC=C4C=CC=CC4=C3C2(C)C)N(CCCCS(=O)(=O)O)C2=CC=C3C=CC=CC3=C21']
    
    # Filter rows without names
    # smiles_list = df[df['name'].isna()]['canonical_smiles'].tolist()

    batch_size = 2
    smiles_to_cid = []

    # Step 1: Map SMILES to CID in batches
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        cids = get_cids_from_smiles_batch(batch)

        # Note: Order is not guaranteed, but assume 1:1 match for now (you can improve this by checking PubChem response structure)
        for j, cid in enumerate(cids):
            smiles_to_cid.append((batch[j], cid))

        print(f"Processed CID batch {i // batch_size + 1}")
        time.sleep(2)

    # Convert to DataFrame
    cid_df = pd.DataFrame(smiles_to_cid, columns=['canonical_smiles', 'cid'])
    print(cid_df)

    # Step 2: Get Names from CIDs
    cid_list = cid_df['cid'].dropna().unique().tolist()
    cid_name_pairs = get_names_from_cids(cid_list)

    # Step 3: Map names back to CIDs
    name_df = pd.DataFrame(cid_name_pairs, columns=['cid', 'name'])
    print(name_df)

    # Step 4: Merge with CID DataFrame
    final_df = pd.merge(cid_df, name_df, on='cid', how='left')

    # # Step 5: Merge back with original DataFrame
    # df_updated = pd.merge(df, final_df, on='canonical_smiles', how='left', suffixes=('', '_new'))

    # # Fill missing 'name' with newly found names
    # df_updated['name'] = df_updated['name'].combine_first(df_updated['name_new'])
    # df_updated.drop(columns=['name_new', 'cid'], inplace=True)

    # # Save to new CSV
    # df_updated.to_csv('natural_com_with_names.csv', index=False)
    # print("Saved updated file with names.")

if __name__ == "__main__":
    main()

    
    

    