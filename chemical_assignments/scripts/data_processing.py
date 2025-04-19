from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import RDKFingerprint
import pandas as pd
import numpy as np

class Preprocessing:
    
    FINGERPRINT_COL = "FingerPrint"
    VALID_COL = "Valid"
    MOL_COL = "mol"
    SMILE_COL = 'canonical_smiles'

    def __init__(self, df):
        self.df = df
    def check_compound_name(self):
        if 'name' in self.df.columns:
            self.df = self.df.rename(columns={'name': 'compound_name'})
        self.df['compound_name'] = self.df['compound_name'].fillna('Not-Specified')
    def remove_invalid_smile(self):
        self.df[self.VALID_COL] = self.df[self.SMILE_COL].apply(self.detect_invalid_smile)
        self.df = self.df[self.df[self.VALID_COL] == True]
        return self
    
    def finger_print_add(self, fingerprint_type):
        fingerprint_type = fingerprint_type.casefold()
        if fingerprint_type == 'morgan':
            self.morgan_finger_print_add()
        elif fingerprint_type == 'rdkit':
            self.rdkit_fingerprint_add()
        else:
            raise ValueError("Only 'morgan' and 'rdkit' fingerprints are supported now.")

    def morgan_finger_print_add(self):
        self.df[self.FINGERPRINT_COL] = self.df[self.MOL_COL].apply(self.generate_Morgran_fingerprint)
        
    def rdkit_fingerprint_add(self):
        self.df[self.FINGERPRINT_COL] = self.df[self.MOL_COL].apply(self.generate_rdkit_fingerprint)

    def get_fp_csmile_df(self):
        fp_lengths = self.df[self.FINGERPRINT_COL].apply(len).unique()
        if len(fp_lengths) > 1:
            raise ValueError(f"Inconsistent fingerprint lengths: detected: {fp_lengths}")

        fp_mat = pd.DataFrame(self.df[self.FINGERPRINT_COL].tolist(), index=self.df.index)
        fp_df = pd.concat([self.df.drop(columns=[self.FINGERPRINT_COL, self.VALID_COL]), fp_mat], axis=1)
        return fp_df

    def get_mole_prop_df(self):
        properties = self.df[self.SMILE_COL].apply(self.calculate_properties)
        properties_df = pd.DataFrame(properties.tolist())
        chem_prop_df = pd.concat([self.df.drop(columns=[self.FINGERPRINT_COL, self.VALID_COL]), properties_df], axis=1)
        return chem_prop_df

    def convert_smile_to_mol(self):
        self.df[self.MOL_COL] = self.df[self.SMILE_COL].apply(Chem.MolFromSmiles)
    
    @ staticmethod
    def calculate_properties(smiles):
        """
        Referenced: Fenna Fennastra's code
        Args:
            smiles (string): Canonical smiles

        Returns:
            dictionary:  Properties of a molecules like weight and everything
        """
        
        mol = Chem.MolFromSmiles(smiles)
        properties = {
        #Basic Physicochemical Properties
        "Molecular Weight": Descriptors.MolWt(mol),
        # Lipophilicity and Solubility
        "logP": Descriptors.MolLogP(mol),
        # Polarity
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        # Hydrogen Bonding
        "H-Bond Donors": Descriptors.NumHDonors(mol),
        "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
        }
        return properties
    
    @staticmethod
    def generate_Morgran_fingerprint(mol, radius=2, nBits =1024):
        return AllChem.GetMorganFingerprintAsBitVect(mol,radius=radius, nBits=nBits)
    
    @staticmethod
    def generate_rdkit_fingerprint(mol, minPath=1, maxPath =7, fpsize =1024):
        return RDKFingerprint(mol, minPath=minPath, maxPath =maxPath,  fpSize =fpsize)
        
    @staticmethod
    def detect_invalid_smile(smiles):
        """referenced from Fenna's code"""
        mol = Chem.MolFromSmiles(smiles)
        try: 
            if mol:
                return True
            else:
                return False
        except:
            return False
    
    def get_features(self, fingerprint_type):
        """ Generates a complete feature matrix combining molecular 
        fingerprints and chemical properties and returns it
        """
        self.check_compound_name()
        self.remove_invalid_smile()
        self.convert_smile_to_mol()
        self.finger_print_add(fingerprint_type)
        fp_df = self.get_fp_csmile_df() 
        chem_prop_df = self.get_mole_prop_df()
        
        fp_mat = pd.concat([fp_df.iloc[:,:4],pd.DataFrame(np.array(fp_df[0].tolist()))],axis=1)
        
        return pd.concat([fp_mat.reset_index(drop=True),chem_prop_df.iloc[:,4:]],axis =1)