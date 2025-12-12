import yaml
from pyteomics import mgf
from rdkit import Chem
import pubchempy as pcp
from pyteomics import mgf
from .dataset import load_cand_data_mzi
from .utils import DatasetBuilder, collate_spectra_data, Spectra_data
import torch
import numpy as np
from .train_contr import train_contr
from torch.utils.data import DataLoader

def init_datadir(dataset_builder):
# Initialize dataset_builder object with blank values because we will not need it
    dataset_builder.data_dict = []
    dataset_builder.mol_dict = []
    dataset_builder.pair_dict = []
    dataset_builder.split_dict = []
    dataset_builder.fp_dict = []
    dataset_builder.in_to_id_dict = []
    dataset_builder.in_to_id_dict_wneg = []
    dataset_builder.data_dir = './data/NPLIB1/'

def get_from_pubchem(in_smiles):
    m = Chem.MolFromSmiles(in_smiles)
    form = Chem.rdMolDescriptors.CalcMolFormula(m)
    smiles = []
    try:
        compounds = pcp.get_compounds(form, 'formula',record_format='json')
    except:
        return smiles
    for compound in compounds:
        smiles.append(compound.canonical_smiles)
    return smiles
    
def norm_mzi(mz, inten): #normalize intensoty to 999 because that is how JESTR has been trained. Drop mz < 1000 Da
    mz = mz.reshape(-1, 1)
    inten = inten/max(inten)
    inten = inten * 999
    inten = inten.reshape(-1, 1)
    mzi = np.hstack((inten, mz))
    idx_mz = mzi[:,1] <= 1000
    mzi = mzi[idx_mz]
    return mzi


def load_variables():
    with open('params.yaml') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    dir_path = ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ms_intensity_threshold = 0.0
    dataset_builder = DatasetBuilder(params['exp'])
    init_datadir(dataset_builder)
    molgraph_dict = {}
    data_path = dir_path + dataset_builder.data_dir
    dataset_builder.molgraph_dict = molgraph_dict
    logfile = './trial_server.log'
    output = open(logfile, 'a')
    #populate molgraph_dict with training molecules
    spec_id_from_mgf = 'CCMSLIB00000855758' 
    target_smiles = 'CC1=CC2C=C(C)C(C)C3C(CC(C)C)NC(=O)C23C(=O)CCC2OC(C)(C)OC2CC1'
    mgf_file = data_path + 'small_data.mgf' 
    candidate_dict = {'COc1c2c(c(C)c3c1C(=O)OC3)OC1(C)CC3(O)OCC4(C)OC5(OC)CCC(C)(C1C2)C43O5': ['CC(C)C12OC1C1OC13C1(C)CCC4=C(COC4=O)C1CC1OC13C2OC(=O)CC(C)(C)C(=O)O', 'COC(=O)C12C3C4OC(=O)C3(C)CC3=C(C)C5=C(CC(OC1(O)C(C)O4)C32C)C(C)(C)OC(=O)C5', 'COC(=O)C12C3C4OC(=O)C3(C)CC13CC(OC2(O)C(C)O4)C(C)=C1C(=C3C)CC(=O)OC1(C)C', 'COC(=O)C12C(=O)C(C)(O)C(=O)C(C)(CC3C(C)(O)C4=CC(=O)OC(C)(C)C4=CCC31C)C21CO1', 'CC1(C)OC(=O)C=CC2(C)C1CC(O)C1(C)C2CCC2(C)C(C3=CC(=O)OC3O)OC(=O)C3OC321', 'CCCC(=O)c1c(O)c(Cc2c(O)c(C(C)=O)c(O)c(C)c2OC)c(O)c2c1OC(C)(C)C(O)C2', 'CC1(C)OC(=O)C=CC2(C)C1CC(O)C1(C)C2CCC2(C)C(C3=CC(O)OC3=O)OC(=O)C3OC321', 'COc1c(C)c(O)c(C(C)=O)c(O)c1Cc1c(O)c2c(c(C(=O)C(C)C)c1O)OC(C)(C)C(O)C2', 'COC(=O)C12C(=O)C(C)OC3OC(=O)C(C)(CC4C(C)(O)C5=CC(=O)OC(C)(C)C5=CCC41C)C32', 'COC(=O)C12C3C4OC(=O)C3(C)CC3C(C)C5=CC(=O)OC(C)(C)C5=CC(OC1(O)C(C)O4)C32C', 'COc1cc2c(c(OC)c1OC)-c1c(cc3c(c1O)OCO3)CC(C)C(C)(O)C2OC(=O)C(C)C', 'C=C1CC2(O)C3(C)CCC(=O)C(C)(C)C3=C(O)C(=O)C2(C)C2C(=O)OC(C)(C(=O)OC)C(=O)C12C', 'CCCCCc1c(Oc2cc(OC)cc3c2C(=O)OC3(CCCC)OC)c(O)cc(O)c1C(=O)O', 'CCCCCc1c(Oc2cc3c(cc2OC)C(CCCC)(OC)OC3=O)c(O)cc(O)c1C(=O)O', 'C=C1CC2C3(C)C(=C(O)C(=O)C2(C)C2(C(=O)OC)C(=O)OC(C)C(=O)C12C)C(C)(C)C(=O)CC3O', 'COC(=O)C1C=C2C(=C(C)C3CC4(C)C(=O)OC5OC(C)C(=O)C(C13C)C54O)CC(=O)OC2(C)C', 'COC(=O)C1(C)c2ccoc2CC2C1C(OC(C)=O)C(OC(C)=O)C13OC1(C)C(C)CC(=O)C23C', 'COC(=O)C12C(=O)C(C)OC3OC(=O)C(C)(CC4C(C)C5=CC(=O)OC(C)(C)C5=CC(O)C41C)C32', 'CC(=O)OC1C2=C(C)C3(C=CC(=O)OC3(C)C)CCC2(C)C2C(=O)OC(C)C3(O)C(=O)OC1(C)C23', 'COc1cc(C2c3cc4c(cc3C(O)C(CO)C2C(=O)OC(C)(C)C)OCO4)cc(OC)c1OC', 'C=C1CC2C3(C)C(=C(O)C(=O)C2(C)C2(C(=O)OC)C(=O)C(C)OC(=O)C12C)C(C)(C)C(=O)CC3O', 'COc1c2c(c(OC)c3c(OC)cccc13)C1(OC(C)CC3OC(C)(C)OC31)OC(CC(=O)O)C2', 'CC1(C)OC(=O)CC(O)C2(C)C1CC(=O)C1(C)C2C(O)CC2(C)C(c3ccoc3)OC(=O)C3OC321', 'C=C1C(=O)C23C4OC5OC2(OC(C)=O)C(OC(C)=O)C2C(C)(C)CCC(OC(C)=O)C52C3CCC14', 'COC1=CC(C)C2CC3OC(=O)CC4C5(C)CC6(O)OC(C7COC(=O)C7)(CC34C6C2(C)C1=O)O5', 'CC1(C)OC(=O)CC(O)C2(C)C1C(=O)C(O)C1(C)C2CCC2(C)C(c3ccoc3)OC(=O)C3OC321', 'COc1cc2c(c(OC)c1OC)-c1c(cc3c(c1OC)OCO3)C(OC(C)=O)C(C)C(C)C2OC', 'COc1cc2c(c(OC(C)=O)c1)C(=O)OC(C)CC=CC1OC(C)(C)OC1C(OC(C)=O)CC=C2', 'COC(=O)C1OC12C1C(=O)CC3C(C)(C)OC4CC(=O)OCC43C1CCC2(C)C(O)c1ccoc1', 'COC(=O)C(O)C1C(C)C(=O)C=C2OC3C(C21C)C(C)(O)C12OC1CC(c1ccoc1)C2(C)C3O', 'CC1CC2C3CCC4=C(O)C(O)=CC(=O)C4(C)C34OC4CC2(C)C1(CC(=O)O)C(=O)CCC(=O)O', 'CC(C)(O)C1CC(=O)C2(C)C(CCC3(C)C(c4ccoc4)OC(=O)C4OC432)C1(C=CC(=O)O)CO', 'CC1(C)OCC2(C(O)CC(=O)O)C1CC(=O)C1(C)C2CCC2(C)C(c3ccoc3)OC(=O)C3OC321', 'CC1(C)OC(CC(=O)O)C2(CO)C1CC(=O)C1(C)C2CCC2(C)C(c3ccoc3)OC(=O)C3OC321', 'CC1(C)OC2CC(=O)OCC23C1CC(=O)C1(C)C3CCC(C)(C(O)c2ccoc2)C12OC2C(=O)O', 'CC1(C)OCC2(C=CC(=O)O)C1C(O)C(O)C1(C)C2CCC2(C)C(c3ccoc3)OC(=O)C3OC321', 'CCCCOC(=O)C1C(c2cc(OC)c(OC)c(OC)c2)c2cc3c(cc2C(O)C1CO)OCO3', 'CCOC(=O)C(CCc1ccc2c(c1)OCO2)(Cc1cc(OC)c(OC)c(OC)c1)C(=O)OCC', 'CC(C)(O)C1CC(=O)C2(C)C(CCC3(C)C(c4ccoc4)OC(=O)C4OC432)C12COC(=O)CC2O', 'COc1cc(C2c3c(cc(OC)c(OC)c3OC)C(OC(C)=O)C3COCC23)cc(OC)c1OC', 'CCCCCc1c(Oc2cc(OC)cc(C(=O)CCCC)c2C(=O)OC)c(O)cc(O)c1C(=O)O', 'COC12C=CC3(C)OC(=O)C(O)C3(O)C1(C)C1C(CC3(C)C(c4ccoc4)CC4OC43C1(C)O)O2', 'COc1cc(CC(COC(C)=O)C(COC(C)=O)Cc2ccc3c(c2)OCO3)cc(OC)c1OC', 'CCCC(=O)c1c(O)c(C)c(O)c(Cc2c(O)c(C(=O)C(C)C)c(O)c(C(=O)C(C)C)c2O)c1O', 'COc1cc(OC2OC(CO)C(O)C(O)C2O)c2c(O)c(C(C)=O)c3c(c2c1)CCC1CCCC31', 'C=C(C)C(O)Cc1c(O)c(Cc2c(O)c(C(C)=O)c(O)c(C)c2OC)c(O)c(C(=O)CCC)c1O', 'COC(OC)(C(=O)c1ccccc1OCCCCC(=O)O)c1ccccc1OCCCCC(=O)O', 'CCOc1c(OC)ccc(-c2oc3cc(OC)c(OCC)c(OCC)c3c(=O)c2OC)c1OCC', 'C=C(C)C(O)Cc1c(O)c(Cc2c(O)c(C(C)=O)c(O)c(C)c2OC)c(O)c(C(=O)C(C)C)c1O', 'COc1cc(CC2COCC2Cc2cc(OC)c(OC(C)=O)c(OC)c2)cc(OC)c1OC(C)=O', 'CC1(C)C(O)C2(O)CC34OC2(O)C2(COC(=O)CC12)C3CCC1(C)C(c2ccoc2)OC(=O)CC14', 'CCCCCc1c(Oc2cc(O)cc(CCC(=O)CC)c2C(=O)OC)c(OC)cc(O)c1C(=O)O', 'CCOC(=O)C1C(COCOC)Cc2cc3c(cc2C1c1cc(OC)c(OC)c(OC)c1)OCO3', 'CC1(C)C(O)C2(O)C=C3C4CC(=O)OC(c5ccoc5)C4(C)CCC3C(C)(C2=O)C1C(O)C(=O)O', 'CCOC(=O)C(=O)C(CCC(C)=O)(Cc1cc(OC)c2cccc(OC)c2c1OC)C(=O)OCC', 'CCOC(=O)c1c(O)c2ccc(OC3CC(C)(C)CC(O)=C3OC3CCCCO3)c(C)c2oc1=O', 'COCC=Cc1cc(OC)c2c(c1)C(COC1OC(CO)C(O)C(O)C1O)C(c1ccccc1)O2', 'CC(=O)OCC(=O)C1(OC(C)=O)C(C)CC2C3CCC4C(=O)C(=O)CC(=O)C4(C)C34OC4CC21C', 'C=CC(=O)OCCOCCOc1cc(OCC)c(OCCOCCOC(=O)C=C)c2ccccc12', 'CC12OC3CC(O1)C1(CO)C4C(O)C(=O)C5(C)C(c6cccc(=O)o6)CCC5(O)C4CCC1(C3)O2', 'CCOC(=O)OCC1OC(c2ccc(CC)c(Cc3ccc4c(c3)OCCO4)c2)C(O)C(O)C1O', 'CCOC(CC)Oc1ccc(Oc2cc(=O)c3c(O)c(OC)c(OC(CC)OCC)cc3o2)cc1', 'CCc1ccc(Cc2ccccc2OC2C(O)OC(COC(C)=O)(COC(C)=O)C(O)C2O)cc1', 'CC12C(=O)OCC3(C(O)CC1O)C2CC(O)C1(C)C3C(=O)C(O)C2(C)C(c3ccoc3)CC3OC321', 'CC12OC3CC(O1)C1(CO)C4C(O)C(=O)C5(C)C(c6ccc(=O)oc6)CCC5(O)C4CCC1(C3)O2', 'COC(=O)OC1C=C2C(=CC(=O)OC2(C)C)C(C)C2CC3(C)C(=O)OC4OC(C)C(=O)C(C43)C12C', 'CC(=O)OCOC(COC(C)=O)C(OC(C)=O)C(CCc1ccc2ccccc2c1)COC(C)=O', 'CCOC(=O)COc1ccc(CCc2cc(OCC(=O)OCC)cc(OCC(=O)OCC)c2)cc1', 'C=Cc1cc(COC(C)=O)cc(C2(O)OC(C(C)O)C(O)C(O)C2O)c1Cc1ccc(OC)cc1', 'COC(=O)CC1C2(C)C3=C(C)C(C4=CC(=O)OC4O)CC3OC2C(O)C2C(C)(O)C=CC(=O)C21C', 'COC(=O)CC1C2(C)C3=C(C)C(C4=COC(=O)C4O)CC3OC2C(O)C2C(C)(O)C=CC(=O)C21C', 'C=C1C2CC3(C)C(c4ccoc4)OC(=O)CC13OC1C(OC(C)=O)C(CO)C(C)(C)OC(C=O)C21', 'C=C(CCC12OC(C(C)=O)C(O)(C=O)C(C=O)(CC1O)O2)C(OC(C)=O)C(C)Cc1ccccc1', 'CC(=O)OC1CC2(O)C(CC(O)C3(C)C(c4ccc(=O)oc4)CCC23O)C2(C)C1=CC(O)C1OC12', 'C=CCc1cc(OC)c(OC(C)C(OC(C)=O)c2cc(OC)c(OC)c(OC(C)=O)c2)c(OC)c1', 'CCOc1cc(OC)c(Cc2ccc(OC3CCOC3)cc2)cc1C1OC(CO)C(=O)C(O)C1O', 'CCC(=O)OC1COC(OC(=O)CCC2=CCc3ccccc32)C(OC(=O)CC)C1OC(=O)CC', 'COc1cc(CC(COC(C)=O)Oc2ccc(CCCOC(C)=O)cc2OC)ccc1OC(C)=O', 'CC1=C2OC(=O)CCC(=O)OC(=C3CCC4C(C(O)CC5(C)C4CCC5(O)C(=O)CO)C13C)C2=O', 'COc1ccc2c(c1OC)C13CCCC(C2)C1(O)CC=C(OC(=O)CCC(=O)OC(C)C(=O)O)C3', 'CC1(C)Oc2cc3c(cc2CC1O)CC(c1cc(O)cc(OC2OC(CO)C(O)C(O)C2O)c1)C3', 'O=C(O)c1ccccc1C(=O)OC1C(C2COC3(CCCCC3)O2)OC2OC3(CCCCC3)OC21', 'CC#CC1(C)C(COC(Cc2ccccc2)(C(C)=O)C(=O)OCC)OC(OC(C)=O)C1OC(C)=O', 'COc1ccc(C=Cc2cc(OC)c(OC)c(OC)c2)cc1OC(=O)COCC(=O)OC(C)(C)C', 'C=C1C(=O)C23C4CC5C(C)(C)CCC(OC(C)=O)C5(C(=O)O4)C2C(OC(C)=O)CC1C3OC(C)=O', 'CC1=C2CC(=O)OC(c3ccoc3)C2(C)CCC1C1(CO)C(CC(=O)[O-])OC(C)(C)C1CC(=O)[O-]', 'COc1ccc(C(=C(CC(=O)O)C(=O)OC(C)(C)C)c2cc(OC)c(OC)c(OC)c2)cc1OC', 'COc1cc(CCCO)ccc1OC(CO)C(O)c1ccc(OC2OC3OC3=CC2C)c(OC)c1', 'CC(=O)OC1C(=O)c2c(c(CC=C(C)C)c3cc(O)cc(O)c3c2O)CC1(O)CC(O)CC(C)O', 'CCCCOC(OCCCC)(C(=O)c1ccc(OCC(=O)O)cc1)c1ccc(OCC(=O)O)cc1', 'CCc1ccc(Cc2ccc(COC(C)=O)cc2OC2OC(COC(C)=O)C(O)C(O)C2O)cc1', 'COC(=O)C(=Cc1ccc2c(c1OC)OCCOCCOCCOCCO2)c1ccc(OC)cc1', 'COC(=O)C(C)OC1OC(COC(C)=O)C(OCc2ccccc2)C(O)C1OCc1ccccc1', 'COc1cc2c(cc1OC)C(c1ccc3c(c1)OCO3)C(COC1OCC(O)CC1O)C(CO)C2', 'CC1=C2CCC3C(C(O)CC4(C)C3CCC4(O)C(=O)CO)C2(C)C2=C(OC(=O)CCC(=O)O2)C1=O', 'COc1ccc(C2c3cc(OC)c(OC)cc3CC(COC(C)=O)C2(O)COC(C)=O)cc1OC', 'CC(=O)OC1C=CC(C)=CC2OC(=O)C3(C)OC23C(OC(C)=O)C2C(C)=CCC(OC(C)=O)C12C', 'COc1ccc(COCC2=C(C3OC(CO)C(O)C(O)C3O)C(Cc3ccc(O)cc3)CO2)cc1', 'CC1(C)OCC(C(O)OC2OC3COC(c4ccccc4)OC3C(OCc3ccccc3)C2O)O1', 'CC12CCC3C(=C1CC(=O)OC2c1ccoc1)CC1C(=O)C3(C)C(C(O)C(=O)OO)C(C)(C)C1O', 'COc1cc(C2CC(=O)OC(CC(CCc3cc(OC)c(O)c(OC)c3)OC(C)=O)C2)ccc1O', 'COc1cc(OC)c(CC#Cc2cccc(COC3C(O)C(CO)OC(OC)C3O)c2)c(OC)c1', 'C=Cc1cc(COC(C)=O)cc(OC2OC(C(C)O)C(O)C(O)C2O)c1Cc1ccc(OC)cc1', 'COc1cc(OC)c(CC#Cc2ccc(COC3C(O)C(CO)OC(OC)C3O)cc2)c(OC)c1', 'COc1cc(OC2OC(CO)C(O)C(O)C2O)c2cc1CCC(=O)CCCCc1ccc(O)c-2c1', 'COc1ccc(C2OCC(C(OC(C)=O)c3ccc(OC)c(OC)c3)C2COC(C)=O)cc1OC', 'CC(=O)OC1CC(O)C23C(=O)OC4CC5(C)C(c6ccc(=O)oc6)CCC5(O)C(CCC2(O)C1)C43', 'CCOC(=O)C(=Cc1ccc(OC)c(OC)c1)C(C(=O)OCC)C(O)c1ccc(OC)c(OC)c1', 'C=C(C)C(=O)OCCOCCOc1cc(O)c(OCCOCCOC(=O)C(=C)C)c2ccccc12', 'COc1ccc2c3c1OC1CC(O)(CC=C1OC(=O)CC(OC(C)=O)C(=O)O)C(C2)C(C)CCC3', 'COc1ccc(C2c3cc(OC)c(OC)cc3CC(O)(COC(C)=O)C2COC(C)=O)cc1OC', 'C=CCOC(=O)C1=CC(c2coc3ccccc3c2=O)C(CCOCCOCCO)C(OCC)O1', 'COC(=O)CC1C2(C)C3=C(C)C(=O)CC3OC2C(OC(C)=O)C2C(C)(C(=O)OC)C=CC(=O)C21C', 'CC(=O)OCC1OC(Oc2cc(OC3CCOC3)ccc2Cc2ccc(C)cc2)C(O)C(O)C1O', 'CC(=O)OC1C2=C(C)C(=O)CC(O)(C(OC(C)=O)C3C(C=O)=CC=CC3(C)C1OC(C)=O)C2(C)C', 'CC(=O)Cc1ccc2c(c1)OCCOCCOc1ccc(COC(C)=O)cc1OCCOCCO2', 'C=CCC(C(=O)OC)(C(=O)OC)C1C=CC(C(Cc2ccccc2)(C(=O)OCC)C(=O)OCC)O1', 'C=C1C(=O)C23C(OC(C)=O)CC4C(C)(C=O)CCCC45COC(OC(C)=O)(CC1C2OC(C)=O)C53', 'CC1CC(=O)OC2=C(OC1=O)C1(C)C(=CC2=O)CCC2C1C(O)CC1(C)C2CCC1(O)C(=O)CO', 'COC(OC)(C(=O)c1ccc(OCCCCC(=O)O)cc1)c1ccc(OCCCCC(=O)O)cc1', 'Cc1cc(C2OC(CO)CC(O)C2=O)ccc1-c1ccc(C2OC(CO)C(O)C(O)C2O)c(C)c1', 'C=CC(=O)OCCOC(C)OCCOc1ccc(C(OC)(OC)C(=O)c2ccc(OC)cc2)cc1', 'COc1ccc(CCCC2(O)C=CC(=O)C2=CC=CC(OC(C)=O)C(O)COC(C)=O)cc1OC', 'C=C(COC(C)=O)C(=O)OC1C2C(=C)C(=O)OC(C(C)C3C=CC(=O)C31C)C2OC(=O)CC(C)C', 'C=C(COC(C)=O)C(=O)OC1C2C(=C)C(=O)OC2C(OC(=O)CC(C)C)C(C)C2C=CC(=O)C21C', 'CCC1OC2C3CC(C(C)C3C3C(=O)OC(=O)C3C3C(C)C4CC3C(OC=O)C4O)C2C(=O)C1=O', 'CCC1=C(O)C(=O)C2C3CC(C2O1)C(C1C(=O)OC(=O)C1C1C(C)C2CC1C(OC=O)C2O)C3C', 'C=C1C(=O)C23C(OC(C)=O)CC4C5(C)CCCC4(C(=O)OC5)C2C(OC(C)=O)CC1C3OC(C)=O', 'CC(C1=CC(CO)C(O)C(CO)=C1)(c1cc(CO)c(O)c(CO)c1)c1cc(CO)c(O)c(CO)c1', 'Cc1cc(-c2cc(C)c(OCOCC3CO3)c(OCOCC3CO3)c2)ccc1OCOCC1CO1', 'CC(=O)OCC1(OC(C)=O)CC23CC1CCC2C1(C)CCC2(OC(C)=O)OC(=O)C=C2C1CC3=O', 'C=CCC1CC(C(=O)OC)(C(=O)OC)C(C(=O)OC)(C(=O)OC)CC1C=CCOc1ccccc1', 'COCOc1ccc(C=CC(=O)c2ccc(OCOC)c(CC=C(C)CO)c2O)c(OCOC)c1', 'CC(=O)OC1C2=C(C)C(=O)CC(C(OC(C)=O)C3C(C=O)=CCC(=O)C3(C)C1OC(C)=O)C2(C)C', 'CCCC(=O)OCC1OC(Oc2cc(C=Cc3ccc(OC)cc3)cc(OC)c2)C(O)C(O)C1O', 'COC(=CC=C1C(=O)OC2(CCCCC2C)OC1=O)CC=C1C(=O)OC2(CCCCC2C)OC1=O', 'CCCC(=O)OCC1OC(Oc2ccc(C=Cc3cc(OC)cc(OC)c3)cc2)C(O)C(O)C1O', 'C=C1CC23CC1(OC(=O)C(=O)OC)CCC2C12CC(=O)C(C)(CC(OC(C)=O)C1)C2C3C(=O)OC', 'CC(=O)OCC1=CCCC2C3(CC(c4ccoc4)OC3=O)C(C)CC(OC(C)=O)C12COC(C)=O', 'CC1=CC2OC3CC4OC(=O)C=CC=CC(=O)OCCC(C)C(O)C(=O)OCC2(CC1)C4C31CO1', 'CC(C=CC=C(C)C(=O)O)=CC=CC=C(C)C=CC=C(C)C(=O)OC12OC(CO)C(O)C(O)C1O2', 'COC(C=CC1=C(O)OC2(CCCCC2C)OC1=O)=CC=C1C(=O)OC2(CCCCC2C)OC1=O', 'COC(=O)C1CC(OC(C)=O)C(=O)C2C1(C)CCC1C(=O)OC(C(=O)CCc3ccco3)CC12C', 'C=CC1(C)CC2=CC(=O)C3C4(C)C(=O)OC(C4OC(C)=O)C(OC(C)=O)C3(C)C2CC1OC(C)=O', 'CCC(=O)OCCC(O)C(O)CCOC1=CC2OC(=O)C(c3ccc(OCCO)cc3)=CC2C=C1', 'COC(=O)CC1C2C3=C(C)C(c4ccoc4)CC3OC2C(OC(C)=O)C2C(C=O)C(O)CC(O)C12', 'CC(=O)OC(C(=O)O)C(OC(C)=O)C(=O)Oc1ccc2c(c1)CCC1C2CCC2(C)C(O)CCC12', 'C=C1C(=O)OC2C(OC(=O)C(C)C)C(C)C3C=CC(=O)C3(C)C(OC(=O)C(=CC)COC(C)=O)C12', 'C=C1C2OC(=O)C=C(C)C2CC2C(C(OC(C)=O)C(C=C(C)C)OC(C)=O)=COC(OC(C)=O)C12', 'CC12CCC3c4ccc(O)cc4CCC3C1CC(OC(=O)CCC(=O)O)C2OC(=O)CCC(=O)O', 'CC(=O)C1C2COC(=O)CCCC(=O)OCCCc3ccc(cc3)OC(O2)C(C(C)=O)C1C(C)=O', 'CCOC(=O)OC1(C(=O)COC(=O)OC)CCC2C3CCC4=CC(=O)C=CC4(C)C3C(=O)CC21C', 'CCOC(=O)OCC(=O)C1(OC(=O)OC)CCC2C3CCC4=CC(=O)C=CC4(C)C3C(=O)CC21C'],
                    'CC1=CC2C=C(C)C(C)C3C(CC(C)C)NC(=O)C23C(=O)CCC2OC(C)(C)OC2CC1': ['CCC(C)C(=O)OC(=N)C1CCC=C2C=CC(C)C(C)(CCC3CC(C)CC(=O)O3)C21C', 'CC1=CC(CC2(C)C(C)=CC(OCC(C=O)=NCC(C)O)=C(C)C2C)=CC(C(C)C)C1O', 'CC1=C2C(O)C3C(CC(=O)C4CC(O)CCC43C)C2CCC12OC1CC(C)CNC1C2C', 'CC1=C2C(=O)C3C(CCC4CC(O)CC(O)C43C)C2CCC12OC1CC(C)CNC1C2C', 'CC(C)=CCCC1(C)C=CC(OC(=O)C=C(C)C)C(C)(O)C=CC(=O)C(CN(C)C)=CC1', 'CC(C)=CC1CC(C)C2(CCC3(C)CC4C5=C(CCC32)C(=O)N(CCO)C5CC4(C)O)O1', 'C=C1C(=CC=C2CCCC3(C)C(C(C)ON(CC(C)C)C(C)=O)=CCC23)CC(O)CC1O', 'CCC=C(C=CC(C)C(=O)N(CCC)CC(C)(C)OC1(C(C)=O)CC1)OC1=CCCC=C1', 'CC(C)=CCON=C(C)C1CCC2(O)C3=CC(=O)C4(C)CC(O)CCC4(C)C3CCC12C', 'C=C1C(=CC=C2CCCC3(C)C(C(C)OCC(=O)N(C)C(C)C)=CCC23)CC(O)CC1O', 'C=C1C(=CC=C2CCCC3(C)C(C(C)ON(C(C)=O)C(C)(C)C)=CCC23)CC(O)CC1O', 'CCC(=O)N(C1CC1)C(C)C1CCC2(O)C3=CC(=O)C4CC(O)CCC4(C)C3CCC12C', 'C=C1C(=CC=C2CCCC3(C)C(CC)=CCC23)CC(O)(OCC(=O)N(C)C(C)C)CC1O', 'C=C1C(=CC=C2CCCC3(C)C2CCC3C(C)CC2CC(C)(O)C(=O)N2)CC(O)CC1O', 'C=C1C(=CC=C2CCCC3(C)C(C(C)OCC(=O)NC(C)(C)C)=CCC23)CC(O)CC1O', 'C=C1C(=CC=C2CCCC3(C)C2CCC3C(C)C(O)C2CCN(C)C2=O)CC(O)CC1O', 'C=C1C(=CC=C2CCCC3(C)C(CC)=CCC23)CC(O)(ON(CC(C)C)C(C)=O)CC1O', 'COC1C(CC(C)(C)C)CC2C3CC4=CC=C(OCCO)C5OC1C2(CN3CC1CC1)C45', 'COCOC1CC2=CC=C3C4CCC(C(C)CC#N)C4(C)CCC3C2(C)C(OCOC)C1', 'C=C1C(=CC=C2CCCC3(C)C(CC)=CCC23)CC(O)(ON(C(C)=O)C(C)(C)C)CC1O', 'C=C1C(=CC=C2CCCC3(C)C(CC)=CCC23)CC(O)(OCC(=O)NC(C)(C)C)CC1O', 'CC1=CCCC(C)C(O)C(C)=CC(C)C(O)C(C)CC(C)Cc2cc(O)cc(c2)NC1=O', 'COC(=O)C1(C)CCCC2(C)C1CCC13C=C(C(C)C)C(CC21)C1C(O)CCC(=O)NC13', 'CCC(CC(=O)C1(C2=CC=C(C(=O)N3CCCCC3)C(C)C2C)OCCCO1)CC1CC1', 'COc1[nH]c(CC=C(C)CC=CC(C)=CC(C)C(O)C(C)=CC(C)C)c(C)c(=O)c1OC', 'CCOC1=C(C(=O)N2CCC3(CC2)CC(OCC)CC(C2=CCCCC2)O3)C=CCC1C', 'C=NC(CCC(=O)OC)C1CCC2C3C(=O)C(=CC)C4CC(O)CCC4(C)C3CCC12C', 'CC(C)N(C(=O)C1CCC2C3C(O)C=C4C=C(C(=O)O)CCC4(C)C3CCC12C)C(C)C', 'CCOCCOOC=CCNC(=O)C=C(C)C=CC=C(C)C=CC1=C(C)CCCC1(C)C', 'CCC=CC(=CCC1=CC(C)=CC=CCCC1)CCCC(=O)ON(C)COCCCC=O', 'C=CCON=CCOCC(C)C1CCC2C(=CC=C3CC(O)CC(O)C3=C)CCCC21C', 'C=C1C2CC3C(CC(=O)C4CC(O)CCC43C)C2(O)CCC12OC1CC(C)CNC1C2C', 'CCCC1OC1C1CC23CCC1(OC)C1OCC4=C(C=CCOC)CC2N(C)CC(C)C413', 'CCC1C=C2CC(O)CCC2(C)C2CCC3(C)C(C4CNCCO4)=C(CC(=O)O)CC3C12', 'CCC=C(C)C(OC)C(C)C=C(C)C=CCC(C)=CCc1[nH]c(OC)c(OC)c(=O)c1C', 'CCCCOCCCCC1C(C)C(C#N)CC1C1(C=O)CC2C=C(C(C)C)C1(C(=O)O)C2', 'COC(=O)C1(C)CCCC2(C)C1CCC13C=C(C(C)C)C(CC21)C1C(O)CCC(=NO)C13', 'CC(=CC(C(c1ccccc1)C1CC(C)CCC1C(C)C)N(C=O)OC(C)(C)C)C(=O)O', 'CNC(=O)C1=C2C(CCC3C4(C)CCCC23COC4)C2(C)CCC(C)C(C(=O)O)C2(C)C1', 'CCCCCC(C)C(C)C1=CC(O)(C(C)=O)C(c2cnccc2C(C)C)C(O)(C(C)=O)C1', 'COC(=O)C1=CC2=CCC3C(CCC4(C)C(C(=O)N(C)C(C)C(C)O)CCC34)C2(C)CC1', 'Cc1cc(O)cc(C(=O)C2C(C)C(NC(=O)OC(C)(C)C)CC3C(C)(C)CCCC23C)c1', 'C=C1C2(O)CC3C(CC(=O)C4CC(O)CCC43C)C2CCC12OC1CC(C)CNC1C2C', 'CC=C(C)C(OC)C(C)C=C(C)C=C(C)CC(C)=CCc1[nH]c(OC)c(OC)c(=O)c1C', 'CC1CCC2N(C1)CC1C(C(O)CC3(O)C1CC1C3CCC3=CC(=O)CCC31C)C2(C)O', 'COCOC1CC(C)(Cc2ccncc2)C(=O)C(C)C23CCC(OC)C2C1(C)C(C)CC3', 'CC1CCN(CC(=O)C2CCC3(O)C4=CC(=O)C5CC(O)CCC5(C)C4CCC23C)CC1', 'COC1CCC(N(C(=O)C2CCC(C)CC2)C2=C(C(=O)O)CC(C3CCCCC3)=C2)CC1', 'CC1(C=C2NC3CCCCC3C2=O)C(CC(=O)O)CCC2C1CCC1(C)C2CCC1(C)O', 'CC=C=C(C)C=CCOC(=O)C(C)=CC(C)(C)C(OC(=O)CCCN(C)C)C(C)=C=CC', 'CC(C)C(NC(=O)C1CCC2C3CC=C4C=C(C(=O)O)CCC4(C)C3CCC12C)C(C)O', 'CC(=CC(C)C(O)C1=NC(C)(C)CO1)C1OC(C(C)COCc2ccccc2)CCC1C', 'CC(C)N(C(=O)C1CCC2C3CC=C4C=C(OC(=O)O)CCC4(C)C3CCC12C)C(C)C', 'COC(=O)C1=CC2=CCC3C(CCC4(C)C(C(=O)NC(C)(C)C(C)O)CCC34)C2(C)CC1', 'CC=CC(C)(OCC1=NC(C)(C)CO1)C1OC(C(C)COCc2ccccc2)CCC1C', 'CC12CCC(N3CCCC3)CC1CCC1C2CC(O)C2(C)C(C3=CC(=O)OC3)CCC12O', 'C=C(CN1CCOCC1)C(=O)C1CCC2C3CCC4CC(O)CCC4(C)C3C(=O)CC12C', 'CC1C(=O)N(C2CCC(C(=O)OC3(C)CCC45CCCC4C(C)(C)C3C5)CC2)C(=O)C1C', 'CCOC(=O)C#CC1(C)CC(C)C2C3COC4(CNC)CC(=O)CCC4(C)C3CCC21C', 'CC(=O)C1CCC2C3CCC4=CC(=O)CC(NC(CC(C)C)C(=O)O)C4(C)C3CCC12C', 'CC(=O)C1CCC2C3CCC4=CC(=O)CCC4(C)C3C(NCC(=O)OC(C)(C)C)CC12C', 'CCOC1=C(C)CC(=O)N(CCC(O)c2ccc(C(C)(C)CC)cc2C(C)(C)CC)C1=O', 'C=C(C)C1CCC(C)=CC1c1c(O)cc(CCCCC)c(CN(C)C(=O)OC(C)C)c1O', 'CCCC12CC(C(C)(O)C(C)(C)CC)C(OC)C3Oc4c(O)ccc5c4C31CCNC2C5', 'CCOC(=O)C(OC(C)(C)C)c1c(C)c(C)c2c(c1C)CC(=O)N(CC1CCCCC1)C2', 'CC#CCCCC=CC1CCCC1CC=CCCC(OC1CCCCO1)C(=O)NC(C)=O', 'CC(=O)OC1CCC2(C)C(=CCC3C2CCC2(C)C3CCC2C(C)(O)C2CC(C)=NO2)C1', 'CCCCCCCCn1c(=O)c(OCC=C(C)C)c(OC)c2ccc(OCCCC)cc21', 'CC(CCCC(O)c1cocn1)C1CCC2C(=CC=C3CC(O)CC(O)C3)CCCC21C', 'CC(C)N(C(=O)C1CCC2C3CCC4=CC(OC(=O)O)=CCC4(C)C3CCC12C)C(C)C', 'CCOC1=C(C)CC(=O)N(CCCOc2ccc(C(C)(C)CC)cc2C(C)(C)CC)C1=O', 'O=C1CCC2(OCC3CCCCC3)C3CC4CCC(O)C5OC1C2(CCN3CC1CC1)C45', 'CC1CCC(C(C)C)C(OC(=O)c2cncc(C(=O)OC3CC(C)CCC3C(C)C)c2)C1', 'CC1C=CC(OCC(O)CCC2=C(CCCCCCC(=O)NCC3CC3)C(=O)CC2)=CC1', 'COCCOCCOC(C)C(C)(C)NC(=O)C(C1=CCCC=C1)(c1ccccc1)C(C)C', 'CC(=CC(Cc1ccccc1)N(C=O)OC(C)(C)C)C(=O)OC1CC(C)CCC1C(C)C', 'C=C(C)C1CCC(C)=CC1c1c(O)cc(CCCCC)c(CN(C)C(=O)OCCC)c1O', 'CCN(CC)C(=O)c1cc(OC)c2c(c1)OC(C)(CCC=C(C)CCC=C(C)C)C(O)C2', 'CC(CCCC(C)(N)C(=O)O)C1C(=O)CC2C3CC=C4CC(=O)CCC4(C)C3CCC21C', 'CCCCCCCCC(=O)Oc1ccc(C2C(CO)NC(=O)C2C(C)C)c2c1CCCC2', 'CCCN(CCC)Cc1cc2c(o1)CC1C3C2C(=O)OC3C(O)C2C(C)(C)CCCC12C', 'CCC(C)(C)C(C)(O)C1CC2(CC)C3Cc4ccc(O)c5c4C2(CCN3)C(O5)C1(C)OC', 'CCC1(OC)C(C(C)(O)C(C)(C)C)CC2(C)C3Cc4ccc(OC)c5c4C2(CCN3)C1O5', 'CCCCCC(O)CN(C)C1C=CC(OCc2ccccc2)C1CC=CCCCC(=O)O', 'C=C(C)C1CCC(CO)=CC1c1c(O)cc(C(C)(C)CCCCN2CCOCC2)cc1O', 'CCCCCC(C)C(C)c1cc(O)c2c(c1)OC(C)(C)C1=C2CN(C(=O)CCCO)CC1', 'COC(=O)C1(C)CCCC2(C)C1CCC13CC4(C(C)C)OC5CCC(=NO)C1C5C4CC23', 'CC1CCC(C(C)C)C(OC(=O)c2ccc(C(=O)OC3CC(C)CCC3C(C)C)nc2)C1', 'COC1C2Oc3c(O)ccc(C)c3C23CCN(CC2CC2)CC3CC1C(C)(O)C(C)(C)C', 'CCCCCC(O)C=CC1C(O)CC2Oc3c(CCCC(=O)N(CC)CC)cccc3C21', 'CCCCCCCCCC1C2C=CC(CC2)C1C(=O)Nc1c(OC)cc(OC)cc1OC', 'CC1CCC(C(C)C)C(OC(=O)c2cccc(C(=O)OC3CC(C)CCC3C(C)C)n2)C1', 'CCOC(=O)C=CCCCON=C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C', 'CC(C(=O)O)=C(C(Cc1ccccc1)NC(=O)OC(C)(C)C)C1CC(C)CCC1C(C)C', 'COCOc1c(N(C)c2cc(C)cc(C(C)(C)C)c2OCOC)cc(C)cc1C(C)(C)C', 'CCCCCCC(CC(=O)c1cc2c(cc1C(C)C)ON(CCCCCC)C=C2)C(=O)O', 'CC12CCC(N3CCC(O)C3)CC1CCC1C2CCC2(C)C(C3=CC(=O)OC3)CCC12O', 'CC(=O)OCC1CC(C)CCC1(C)C1CCC2(C)C(CCC2(O)c2ccccn2)C1CO', 'CCCCCC(C)C(C)c1cc(O)c2c(c1)OC(C)(C)C1=C2C(CC)N(CC(=O)O)CC1', 'COCOc1c(C)cc(C(C)(C)C)cc1N(C)c1cc(C(C)(C)C)cc(C)c1OCOC', 'CCN(C(=O)OC(C)OC(=O)C(C)(C)CC(C)(C)C)C1C2CCC(C2)C1c1ccccc1', 'CC(C(=O)O)c1cc2c(cc1OC(=O)N(C(C)C)C(C)C)C1(C)CCCC(C)(C)C1CC2', 'CCCCCCC(C)(C)c1cc(O)c2c(c1)OC(C)(C)C1CC=C(C(=O)NCCO)CC21', 'CCCC(C)(O)C1(C)CC2(CC)C3Cc4ccc(O)c5c4C2(CCN3C)C(O5)C1(C)OC', 'CCCCCCCCCCCCC=CC1OCC2C1OC(=O)N2Cc1ccc(OC)cc1', 'CC(=Cc1ccccc1)C(O)C(=CC(C)(C)C)C(=O)C(C)(C)OC(=O)N(C(C)C)C(C)C', 'CCOc1c(C)cc(=O)n(CCC(O)c2ccc(C(C)(C)CC)cc2C(C)(C)CC)c1O', 'COC(=O)C1(C)CCCC2(C)C1CCC13CC4C(CC21)C1C(CCC(=NO)C13)OC4(C)C', 'CC(=O)OC1CCC2(C)C(=CCC3C2CCC2(C)C3CC(N3CCCC3)C2OC(C)=O)C1', 'CC12CCC(=O)NC1CCC1C2CCC2(C)C1CCC2(O)C#CCCOC1CCCCO1', 'COC(O)c1[nH]c(C(C)(C)C)cc1CC(C)C(C)C(=O)CC(C)C(C)c1ccc(O)cc1', 'COc1ccc(C(=O)N2C(=O)CCCC(O)CCCC2C2CCCCCCCCC2)cc1', 'CCN(CC)C(=O)CCCC=CCC1C(O)CC(O)C1C=CC(O)CCc1ccccc1', 'CCCCCCCCCCCCCCCCOc1nc2ccc(CC(C)=O)cc2c(=O)o1', 'COC1=CC(CC=O)C(CCCN(C)CCCc2ccccc2)(C(C)C)C(OC)=C1OC', 'CCCCCC(O)C=CC1C(O)CC2Oc3c(CCCC(=O)NC(C)(C)C)cccc3C21', 'COC(=O)CCCCCCN1C(=O)CCC1C=CC(O)C(C)CCCCc1ccccc1', 'CC(=CC(Cc1ccccc1)NC(=O)OC(C)(C)C)C(=O)OC1CC(C)CCC1C(C)C', 'COCOc1c(C(N)c2cc(C)cc(C(C)(C)C)c2OCOC)cc(C)cc1C(C)(C)C', '[C-]#[N+]C1CC(O)C(c2ccc(C(C)(C)C(O)CCCC)cc2)C1CCCCCCC(=O)O', 'CCCCCCCCCCCC=CCC1OCC2C1OC(=O)N2Cc1ccc(OC)cc1', 'CCCCCCC(C)(C)c1cc(OC(C)=O)c2c(c1)N1CCCC1CC2CC(=O)OCC', 'CN(O)C(=O)C(C)(C)CCC=CCC1C2CCC(O2)C1CCCCOCc1ccccc1', 'COCOc1c(N(c2cccc(C(C)(C)C)c2OCOC)C(C)C)cccc1C(C)(C)C', 'C=C(C=C(N)c1cc(O)c(OCCCCCCC)cc1CC(C)CCO)C(=CC)C(C)=O', 'COCOc1c(C)cc(C(C)(C)C)cc1C(N)c1cc(C(C)(C)C)cc(C)c1OCOC', 'CC(CCCCCc1ccccc1)C(O)C=CC1CCC(=O)N1CCCCCCC(=O)O', 'CCCCOC(C)OC1(C#N)CCC2C3CCC4=CC5(CCC4C3CCC21C)OCCO5', 'CCCCCCC1CC1C(O)C(C(=O)N1C(=O)OC(C)(C)C1Cc1ccccc1)C(C)C', 'COc1ccc2c3c1OC1C(OC(C)(C)CCOC(C)(C)C)CCC4C(C2)N(C)CCC341', 'COCOc1ccc(C(C)(C)C)cc1N(c1cc(C(C)(C)C)ccc1OCOC)C(C)C', 'C=Cc1ccc(C(COC(=O)OC2CCCCC2)ON2C(C)(C)CC(C)CC2(C)C)cc1', 'CC(C)=CCCC(C)=CC(=O)C(OC(=O)N(C(C)C)C(C)C)C(C)OCc1ccccc1', 'CC(C)c1cc2c(cc1OCC(O)CN1CCOCC1)C1(C)CCCC(C)(C)C1CC2=O', 'C=C(C=NC(=C)c1ccc(OCC(O)COCCCCCCCC)cc1O)C(C)=CCC', 'CCCCCC(O)C=CC1C(O)CC2Oc3c(CCCCN4CCOCC4)cccc3C21', 'CCCCCCCCCCC#CC#CCCCCCCCC(C(=O)O)N1C(=O)CCC1=O', 'CCCCCC=CCC=CCCCCCCCC(=O)CC(O)COC(=O)c1ccncc1', 'C=C(CC(C)(C)C)N1C(c2ccc(CC)c(OCCCOC)c2)CCC1C1CCC(=O)O1', 'CC(=O)CCCCCCCCCCC1CCC(OC(=O)C=Cc2ccc(O)cc2)C(C)N1', 'CCCCCCCCCCC#Cc1cccc(C(O)(CCO)N(C(C)=O)C2CCCO2)c1', 'CNc1ccc2c(c1)C(OCCC1CCCCC1)C(OC(=O)C1CCCCC1)C(C)(C)O2', 'O=C1CC=CCCCCC2CCCN(C2)C(=O)C(=O)C2CCCCC2C(=O)CCCCC1', 'COC(=O)CCCC=CCN1C(=O)CCC1CCC(O)C(C)CCCCc1ccccc1', 'C=CC=C(C)C(=O)Nc1cc(OC)c(OC)c(CCCCCCCCCCC=C)c1OC', 'CCc1ccccc1C(NOC(C)Oc1cccc(CC(C)C)c1CC(C)C)(OC)OC', 'COC(=O)N1C(CC=CCCCCCCCC(C)=O)CCC(OCc2ccccc2)C1C', 'CC(C)(C)C(CC(C)(C)C(C)(C)C#N)c1ccc(C2CC3C(O)CC(O)OC3CO2)cc1', 'C=CCCCCCCCCOC(=O)c1cncc(C(=O)OCCCCCCCCC=C)c1', 'CCCCCC(C)C(C)c1cc(O)c2c(c1)OC(C)(C)C1=C2CN(CC(=O)OCC)CC1', 'COc1ccc(C2CN(C(=O)CC3CCCCC3)CC2(C)C(C)O)cc1OC1CCCC1', 'CCCCCCCCCC=CCCCNc1cccc(C(=O)C(C(C)=O)C(=O)OCC)c1', 'CCNC(=O)CCCC=CCC1C(O)CC(COC)C1C=CC(O)CCc1ccccc1', 'CCOc1c(C)cc(=O)n(CCCOc2ccc(C(C)(C)CC)cc2C(C)(C)CC)c1O', 'CCNC(=O)CCCC=CCC1C(OC)CC(CO)C1C=CC(O)CCc1ccccc1', 'CC(=O)CNC(=O)CCC(C)C1CCC2C3C(=O)CC4CC(=O)CCC4(C)C3CCC12C', 'CCCOC1=Cc2oc3c(c2CC1)C(=O)C(C)=C(CC(C)OCCN(CC)CC)C3(C)C', 'CC(C)CCCC1(C)Oc2ccccc2C2OC3CCN(C(=O)OC(C)(C)C)CC3CC21', 'CCCN(c1cccc(C(C)(C)C)c1OCOC)c1cccc(C(C)(C)C)c1OCOC', 'CCCN(c1cc(C(C)(C)C)ccc1OCOC)c1cc(C(C)(C)C)ccc1OCOC', 'CC(N)C(=O)OC1CCC2(C)C(CCC3C2CCC2(C)C(C4=COCC=C4)CCC32O)C1', 'O=C1CC=CCCCCC2CCCC(C2)C(=O)C(=O)N2CCCCC2C(=O)CCCCC1', 'CC(C)(OC(=O)Cc1ccccc1CC(=O)OC(C)(C)C1CCNCC1)C1CCCCC1', 'CC(CCCCCc1ccccc1)C(O)CCC1CCC(=O)N1CC=CCCCC(=O)O', 'CC12CCC(N3CCOCC3)CC1CCC1C2CCC2(C)C(C3=CC(=O)OC3)CCC12O', 'CCCCCCCCCCCCCCCCCCN1C(=O)c2ccc(C(=O)O)cc2C1=O', 'CCCCC(O)C(C)(C)c1ccc(C2C(O)CC(C#N)C2CCCCCCC(=O)O)cc1', 'CN(C(=O)CC1CCCCC1)C1CCC2(CCO)CC(c3ccccc3)CC(C)(O2)C1O', 'O=C(O)CCCCCCCCC#CC#CCCCCCCCCCCN1C(=O)CCC1=O', 'COC(=O)CC(NC(=O)C1(C)CCCC2(C)C(CCc3ccoc3)=C(C)CCC12)C(C)C', 'C=C(CC(O)(C(=O)O)C(N)CCC1CCC1)c1cc(CCC2CC2)oc1C=CCCCC', 'CCCCCCCCOc1ccc(C=C2N=C(C)OC2=O)cc1OCCCCCCCC', 'COc1cc(C(O)C(C)(C)C)ccc1CN(C)Cc1ccc(C(O)C(C)(C)C)cc1OC', 'CCNC(=O)CCCC=CCC1C(O)CC(O)C1C=CCCC(O)CCc1ccccc1', 'CCCC(C)(O)C12CC3(CC)C4Cc5ccc(OC)c(O)c5C3(CCN4C)C1C2(C)OC', 'CC(C)CCCC1(C)Oc2ccccc2C2OCC3(CCCN3C(=O)OC(C)(C)C)CC21', 'CCCC(O)(CNCCC1C2CCC(O2)C1CC=CCCCC(=O)O)Cc1ccccc1', 'COc1c2c(cc3c1C(CC(=O)CC1CCCCCCCCCCC1)N(C)CC3)OCO2', 'CCCCCC(O)C=CC1C(O)CC2Oc3c(CCCC(=O)NCCCC)cccc3C21', 'CCCCCC(=O)C1Cc2ccc(cc2)OCCCC(C(=O)CC)C(CC(C)C)C(=O)N1', 'CC(=O)CC(CC(C)C)C(=O)NC(CC(C)C)C(=O)CC(Cc1ccc(C)cc1)C(C)=O', 'CCC(CC)(c1ccc(OCC(O)C(C)(C)C)c(C(C)C)c1)c1ccc(C(=O)N(C)C)o1', 'C=C(c1ccc(C(CC)(CC)c2ccc(OCC(O)C(C)(C)C)c(CC)c2)o1)N(C)OC', 'CCC(C)(C)C1CCC2(OC2(CC)C(=O)Nc2cccc(C(=O)O)c2)C(C(C)(C)CC)C1', 'CCCCCc1cc(O)c(C2C=C(C)CCC2)c(O)c1CN(C)C(=O)OC1CCCCC1', 'CC(C)CNC(=O)CCCC=CCC1C(O)CC(O)C1C=CC(O)CCc1ccccc1', 'CC1CN(CCC(C(=O)CCCC(=O)O)C2CCCCC2)CCC1(C)c1cccc(O)c1', 'CCCCCCOc1ccc(C=CC(=O)OC2CCC(NC(=O)C(C)(C)CC)CC2)cc1', 'C=C1C(=O)C23CCC4C(C)(C(=O)OCCN5CCCCC5)CCCC4(C)C2CCC1(O)C3', 'CCNC(=O)CCCCCC=CCC1C(O)CC(O)C1C=CC(O)CCc1ccccc1', 'CCCCCCCCC=CCCCCCCCC(=O)NC(=O)C=Cc1ccc(O)c(O)c1', 'CCCCOc1ccc(C(=O)C2=C(N3CCOCC3)CCCCCC2)c(OCCCC)c1', 'C=C(CC(O)(C(=O)O)C(N)CCCC1CC1)c1cc(CCC2CC2)oc1C=CCCCC', 'CCCc1cc(C(CC)(CC)c2ccc(C(=O)N(C)C)o2)ccc1OCC(O)C(C)(C)C', 'CCC=CCC=CCC=CCCCCCCCC(=O)Nc1c(OC)cc(OC)cc1OC', 'CC(NC(=O)CCC1CCCCC1)C(O)c1ccc(OC(=O)CCC2CCCCC2)cc1', 'CN(CCO)C1CCC2(C)C(CCC3C2CCC2(C)C(c4ccc(=O)oc4)CCC32O)C1', 'CC=CCCC(=CCC)CCCN(CCc1ccc(OC)c(OC2CCOC2)c1)C(C)=O', 'CCCCC(C)(O)CC=CC1C(O)CC(=O)C1CCCCCCC(=O)Nc1ccccc1', 'CCCCCC(C)C=CC1C(O)CC(=O)C1CCCCCCC(=O)NOc1ccccc1', 'Cc1c(C)c2c(c(C)c1OCCCCC(=O)O)CCC(C)(CCCCCCCCC#N)O2', 'CCC(C)(C)COc1cccc(OCCOc2cccc(OCC(C)(C)CN(C)C)c2)c1', 'CC(C)(C)OC(=O)CCC=CCOC1C(OCc2ccccc2)CCC1N1CCCCC1', 'C=CCCCCCCCC(C=O)ON1C(C)(C)CC(OC(=O)c2ccccc2)CC1(C)C', 'CCCCCCCCCCC1(C(=O)Nc2c(OC)cc(OC)cc2OC)CC2C=CC1C2', 'CC(=O)OC1(c2ccccc2)CCC2C(CN)C(C3(C)CCC(O)CC3CO)CCC21C', 'CCCCCCCCCCCCCCCC#CC(=O)Oc1cnccc1C(=O)OCCC', 'C=C(C)CCCCCCC(=O)C(C)C(CN1CCCC1)C(O)c1ccc2c(c1)OCCO2', 'CCCCCCCCCCCCCC=C1OCC2C1OC(=O)N2Cc1ccc(OC)cc1', 'CCCCCCCOc1ccc(C=CC(=O)OC2CCC(NC(=O)C(C)CC)CC2)cc1', 'CCCCCCCCC(=O)C1C2CN(C(=O)OC(C)(C)C)CC2CC1(O)c1ccccc1', 'CCOC(=O)CC(=O)C(CCN1CCC(C)(c2cccc(O)c2)C(C)C1)C1CCCCC1', 'COc1cc2c(cc1OC)C1CC(COC(=O)CC3CCCC3)C(CC(C)C)CN1CC2', 'CCCCCCCCCCCCCC=CC(O)C1COC(=O)N1C(=O)Cc1ccccc1', 'C=CCC(CC)C1(C)OOC2(CCCC(c3ccc(OCCN4CCCCC4)cc3)C2)O1', 'CCCCCC(O)C=CC1C(O)CC(=O)C1CCCCCCC(=O)NCc1ccccc1', 'CCC(CC)(c1ccc(OCCNO)c(C)c1)c1ccc(OCC(O)C(C)(C)C)c(C)c1', 'CCCCCCCC(=O)C=CCC1CCC(C(O)CCC)N1C(=O)OCc1ccccc1', 'CCCC(=O)NOC1CCC2(C)C(CCC3C2CCC2(C)C(C4=CC(=O)OC4)CCC32)C1', 'CCCCCCCCCCCCCC=CC1OC(=O)N(Cc2ccccc2)C1C(=O)OC', 'CC(C)(CCCOCCCCCCNCC(O)c1ccc(O)c(CO)c1)c1ccccc1', 'CCC(C)C(CCc1ccccc1)OC(=O)C1CCCCN1C(=O)C(=O)C(C)(C)C(C)C', 'COc1cc(OC)c2ccn(CCCCCCCCC3CCCC4(CCC(C)O4)O3)c2c1', 'COc1ccc(CCNC(C)CCC(C)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC', 'CC(CCC(=O)NCC(=O)O)C1=CC23CCC4C(CCC5CC(O)CCC54C)C2CCC13', 'COCCCCC(O)(c1ccccc1OCC1CC1)C1CCCN(C(=O)C2CCCC2)C1', 'CC1(CCCCCCCCC#N)CCc2cc(OCCCCCCCC(=O)O)ccc2O1', 'CCCCCCCCN(CCCCCCCC)c1ccc2cc(C(=O)OC)c(=O)oc2c1', 'CCCCCCC(C)(C)c1cc(O)c2c(c1)OC(C)(C)C1CC=C(COC(=O)CN)CC21', 'OCc1cc(C(O)CNCCCCCCOCCCCCCc2ccccc2)ccc1O', 'CCCCCCCCc1ccc(C#CC2(NC(=O)OC(C)(C)C)COC(C)(C)OC2)cc1', 'CCCCC(c1cc(C2CC2CN2CCOCC2)cc(C(=O)OC)c1C)C1CCOCC1', 'CC12CCC(NCCCO)CC1CCC1C2CCC2(C)C(c3ccc(=O)oc3)CCC12O', 'COc1cc(CCN(C)CCCC(OC)(c2ccc(CO)c(OC)c2)C(C)C)ccc1C', 'COC1(OOC2CCC(c3ccc(OCCN(C)C)cc3)CC2)C2CC3CC(C2)CC1C3', 'COCCCCC(c1ccccc1OCC1CC1)C1CCCN(C(=O)C2CCC(O)C2)C1', 'CCCCCCCCCc1ccc(CN(CC#CCC(C)(C(=O)O)C(=O)O)C(C)C)cc1', 'Cc1c(C)c(O)n(C2CCC(C(=O)OC3(C)CCC45CCCC4C(C)(C)C3C5)CC2)c1O', 'CCCCOc1c(C)c(C)c(O)c(C)c1CNCC(C)Oc1cc(C)c(O)cc1C(C)C', 'CCCCCC(O)C=CC1C(O)CC2CC(CCOCC(=O)NCc3ccccc3)CC21', 'CCCCCCC#Cc1ccc(CCC2(NC(=O)OC(C)(C)C)COC(C)(C)OC2)cc1', 'CCCCN(C)C(CO)COc1ccc(C(C)(C)c2ccc(OCC(C)CO)cc2)cc1', 'COC(CC(=O)CC1CCC(CCN2CCC(c3cccc4c3CCO4)CC2)CC1)OC', 'CCCCCCCCCC(=O)OCOC(=O)N(CC)C1C2CCC(C2)C1c1ccccc1', 'C=C(CN(C)C)C(=O)OCCCc1ccc(OC(=O)C2CCC(CCCCC)CC2)cc1', 'COc1c(C)c(C)c(O)c(C)c1CN(CCOc1cc(C)c(O)cc1C(C)C)C(C)(C)C', 'C=C(C)C(=O)C1CCC(C(=O)CCCCCCCNC(=O)C2CCC(C(=O)C(=C)C)C2)C1', 'CCCC(CCC)OC(=O)OCOc1ccc2c(c1)C13CCCCC1(C)C(C2)N(C)CC3', 'CCCCCC=CCC=CCCCCCCCC(=O)Oc1ccc(CC(N)C(=O)O)cc1', 'CCCCCCCCCCCCCCCCNC(=O)c1cc2ccc(OC)cc2oc1=O', 'COc1ccc(CCN(C)CCCC(C)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC', 'CC(C)(C)OC(=O)CCC1CCC(CNC(C(=O)OC2CCCC2)c2ccccc2)CC1', 'O=C(CCCCCCC1C(=O)CC(O)C1CCC(O)CCc1ccccc1)NCC1CC1', 'O=C(CCCCCCC1C(O)CC(=O)C1CCC(O)CCc1ccccc1)NCC1CC1', 'CCCCCCCCCC(CCCCCCCC(=O)OC)N1C(=O)c2ccccc2C1=O', 'CCOC(N)(CC)c1c(Oc2ccccc2)c(CC(C)C)c(CC(C)C)c(OC)c1OC', 'CCCCCCCCc1ccc(C#CC2(NC(=O)OCCCC)COC(C)(C)OC2)cc1', 'CCCCCCCCCCCCCCCCCCNC(=O)c1ccc2c(c1)C(=O)OC2=O', 'CC(C)(C)c1cc(CN)cc(C(C)(C)C)c1C(O)(Cc1ccccc1)C(CO)(CO)CO', 'CCCCOc1c(C)c(C)c(O)c(C)c1CN(C)CCOc1cc(C)c(O)cc1C(C)C', 'CCCCCCC=CC=CCCCCC1(CCC(COc2ccccc2)=NO)OCCO1', 'CCCCCCCCCCOC(=O)CCCCCCCCC(=O)Oc1ccc(C#N)cc1', 'CCOc1c(C)c(C)c(O)c(C)c1CN(CCOc1cc(C)c(O)cc1C(C)C)C(C)C', 'CCN(CC)CC(CC1CCC2C3CCc4cc(OC)ccc4C3CCC12CO)C(=O)O', 'CCCCCC=CCC=CCCCCCCCC(=O)OC(=O)C(N)Cc1ccc(O)cc1', 'CNCCCOc1ccc(C2CCC(OOC3(OC)C4CC5CC(C4)CC3C5)CC2)cc1', 'CCCCCCCCC1CCC2Cc3c(cccc3OCC(=O)OC(C)CC(N)=O)CC12', 'CCCCCC=CCC=CCCCCCCCC(=O)NC(Cc1ccc(O)cc1)C(=O)O', 'C=C(CN(CC(=C)C(=O)O)c1ccc(CCCCCCCCCCCCC)cc1)C(=O)O', 'C#CCNC(=O)OC1CCC2(C)C(CCC3C4CCC(CCC(=O)OC)C4(C)CCC32)C1', 'CCCCCCCCCCCCCCCCCC(=O)c1c(OC(=O)O)[nH]c2ccccc12', 'CCC1(C)OC23CCC4C(CCC5CC6(CCC54C)CC6C(=O)NCC(=O)O)C2CCC13', 'CCCCCCCCCCCCCCCOC(=O)CCCC(=O)Oc1ccc(C#N)cc1', 'CCCCCCCCC1CCC2Cc3c(cccc3OCC(=O)N(C)CC(=O)OC)CC12', 'C#CCOC(=O)NC(Cc1ccccc1)C(=O)OCCCCCCCCCCCCCC', 'CCCCCCCCOc1ccc(C=C(C#N)C(=O)OC)cc1OCCCCCCCC', 'CCCCCCCCC1CCC2Cc3c(cccc3OCC(=O)OC(CC)C(N)=O)CC12', 'CC(CCC1=C(CCCCCCC(=O)NC(CO)CO)C(=O)CC1)CCc1ccccc1', 'CCCCCCC(=O)CCC1CCCCC12CCC(O)CN2C(=O)OCc1ccccc1', 'CCCCCCCCCCCCCC=CC(=O)C(NC(=O)OCc1ccccc1)C(C)=O', 'CC12CCC3C(CCC4CC(OC(=O)CCC(=O)N5CCCC5)CCC43C)C1CCC2=O', 'CCC(CO)NC(=O)CCCCCCC1=C(CCC(O)CCc2ccccc2)CCC1=O', 'CCCCCC(CCC1CCC2Cc3c(cccc3OCC(=O)OC(C)CC)CC12)N=O', 'COC1(OOC2CCC(Cc3ccc(OCCCN)cc3)CC2)C2CC3CC(C2)CC1C3', 'COC(=O)CCCCCCCCCCCCCCCCCN1C(=O)c2ccccc2C1=O', 'CCCCCC(C)CCC(CC1=CCCCC1)(NC(=O)OCc1ccccc1)C(=O)OC', 'CCCCCCCCCCCCOc1cccc2cc(C(=O)CCN3CCOCC3)oc12', 'CCCCCCC(O)OC(COc1ccccc1CCc1ccc(OC)cc1)CN(C)C', 'COC1(OOC2CCC(c3ccc(OCCCCN)cc3)CC2)C2CC3CC(C2)CC1C3', 'CCOC(=O)CCCCCCCCCCCCCCCCN1C(=O)c2ccccc2C1=O', 'CCCCCCCCCCCCCCCCN(C(=O)c1ccccc1)C1CC(=O)OC1=O', 'CCCCCCCC(O)OC(CNC)COc1ccccc1CCc1ccc(OC)cc1', 'CCCCCCCCCCCCCC1=C(C(O)C(CO)NC(=O)C(=O)c2ccccc2)C1', 'COC(=O)C12CCCC3CC1CC(C14CC5CC(C(=O)N(C)OC)CC(C1)C(C5)C4)(C3)C2', 'CCCCCCCCCCCCOC(=O)CCNC(CC1=CCc2ccccc21)C(=O)O', 'COC(=O)C12CC3CC(C1)CC(C14CC5CC(CC(NC(=O)OC(C)(C)C)(C5)C1)C4)(C3)C2', 'CC12CCCCC1CCC1C2CCC2(C)C(CCCC(=O)ON3C(=O)CCC3=O)CCC12', 'CCCCCC(O)CCC1C(O)CC2Cc3c(cccc3OCC(=O)NCC3CC3)CC21', 'CCCCCCCCN(CCCCCCCC)C(=O)COc1cc(=O)oc2ccccc12', 'CCCCCCCCCCCCCCNC(=O)CCCOc1ccc2ccc(=O)oc2c1', 'COC(=O)C12CCCC3CC1CC(C14CC5CC(C(=O)NCCO)CC(C1)C(C5)C4)(C3)C2', 'CCCCCCCCN(CCCCCCCC)C(=O)COc1ccc2ccc(=O)oc2c1', 'COc1ccc(-c2ccc(OCCCCCCCCCCN(CCO)CCO)cc2)cc1', 'O=C(COCCOCC(=O)NC12CC3CC(CC(C3)C1)C2)CC12CC3CC(CC(C3)C1)C2']}
    if target_smiles not in candidate_dict.keys():
        candidate_smiles = get_from_pubchem(target_smiles)
        assert len(candidate_smiles) != 0, "No candidates found!"
    else:
        candidate_smiles = candidate_dict[target_smiles]
    try:
        spec = mgf.get_spectrum(mgf_file, spec_id_from_mgf)
    except:
        assert False, "No spectrum found in MGF file"
        
    assert spec['params']['charge'][0] == 1, "Not an [M+H]+ spectrum"

    mz = spec['m/z array']
    intensity = spec['intensity array']
    mzi = norm_mzi(mz, intensity)
    data_list = load_cand_data_mzi(dataset_builder, params, target_smiles, candidate_smiles, mzi, device)
    assert len(data_list) >= 2, "Candidates could not be converted to rdkit mol objects"
    return params, dataset_builder, molgraph_dict, data_path, device, output

#Load Molecule Model
#load pre trained weights into an architecture given the training parameters
"""Input: training parameter, path to the weights, output path for logging, device, data_path
Output: ready to use model
"""
def load_molecule_encoder(params, dataset_builder, molgraph_dict, output, device, data_path, weights_path='data/MassSpecGym/pretrained_mol_enc_model_1741546103623_best.pt'):
    params['pretrained_mol_enc_model'] = weights_path
    mol_enc_model_contr, spec_enc_model_contr, models_list = train_contr(dataset_builder, molgraph_dict, params, output,
                                                             device, data_path, True)
    model = mol_enc_model_contr.eval()
    return model

#Load Spectrum Model
#load pre trained weights into an architecture given the training parameters
"""Input: training parameter, path to the weights, output path for logging, device, data_path
Output: ready to use model
"""
def load_spectrum_encoder(params, dataset_builder, molgraph_dict, output, device, data_path, weights_path='data/MassSpecGym/pretrained_spec_enc_model_1741546103623_best.pt'):
    params['pretrained_spec_enc_model'] = weights_path
    mol_enc_model_contr, spec_enc_model_contr, models_list = train_contr(dataset_builder, molgraph_dict, params, output,
                                                             device, data_path, True)
    model = spec_enc_model_contr.eval()
    return model



# Spectrum Encoder:
# Encode the spectrum using get_ms_array_batch (same approach as collate function)
""" Input: model for encoding, model parameters, mass to charge ratio, intesities, device
    Output: vector embedding(spectrum)
"""
def spectrum_encoder(spec_enc_model_contr, params, example_mz, example_intensity, device):
    from .utils import get_ms_array_batch

    # Stack mz and intensity into mzi format (same as dataset format)
    mzi = norm_mzi(example_mz,example_intensity)
    mz_tensor, int_tensor, pad, lengths = get_ms_array_batch(
        [mzi], 
        params['transformation'] if 'transformation' in params else 'log10over3',
        params['max_mz'],
        params['resolution']
    )
    mz_tensor = mz_tensor.to(device)
    int_tensor = int_tensor.to(device)
    pad = pad.to(device) if isinstance(pad, torch.Tensor) else pad

    # Get spectrum embedding
    with torch.no_grad():
        spectrum_embedding = spec_enc_model_contr(mz_tensor, int_tensor, pad, lengths)
    return spectrum_embedding


# Molecule Encoder
# First, build molecule graph and add to molgraph_dict
"""Input: model for encoding, model parameters, molecule smiles string, molgraph_dict, device  
Output:vectror embedding (molecule)"""
def molecule_encoder(mol_enc_model_contr, params, smiles, molgraph_dict, device):
    from rdkit import Chem
    from .dataset import mol_to_graph
    import dgl

    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # Create molecule graph
        mol_to_graph(mol, smiles, molgraph_dict, params, device)
        
        # Get the graph and encode
        mol_graph = molgraph_dict[smiles]
        batched_graph = dgl.batch([mol_graph])
        
        with torch.no_grad():
            molecule_embedding = mol_enc_model_contr(batched_graph, batched_graph.ndata['h'])
        return molecule_embedding

if __name__ == '__main__':
    # Example usage
    params, dataset_builder, molgraph_dict, data_path, device, output = load_variables()
    # 1. Define example spectrum (mz and intensity arrays)
    example_mz = np.array([100.0, 150.5, 200.3, 250.8, 300.1])
    example_intensity = np.array([0.5, 1.0, 0.3, 0.8, 0.2])
    # 2. Define example molecule (SMILES - Ibuprofen)
    example_smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"

    # 4. Encode the molecule
    mol_enc_model_contr = load_molecule_encoder(params, dataset_builder, molgraph_dict, output, device, data_path)
    molecule_embedding = molecule_encoder(mol_enc_model_contr, params, example_smiles, molgraph_dict, device)
    # 5. Encode the spectrum
    spec_enc_model_contr = load_spectrum_encoder(params, dataset_builder, molgraph_dict, output, device, data_path)
    spectrum_embedding = spectrum_encoder(spec_enc_model_contr, params, example_mz, example_intensity, device)

    # 5. Compute cosine similarity
    similarity = torch.nn.functional.cosine_similarity(spectrum_embedding, molecule_embedding, dim=1)

    print(f"Spectrum shape: {spectrum_embedding.shape}")
    print(f"Molecule embedding shape: {molecule_embedding.shape}")
    print(f"Cosine similarity: {similarity.item():.4f}")