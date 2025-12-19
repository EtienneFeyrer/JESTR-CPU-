import yaml
from pyteomics import mgf
from rdkit import Chem
import pubchempy as pcp
import torch
import numpy as np
from functools import partial
from pathlib import Path
import sys
from argparse import Namespace
# Import official JESTR components
from pytorch_lightning import Trainer
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from massspecgym.models.base import Stage
from jestr.data.data_module import TestDataModule
from jestr.data.datasets import ContrastiveDataset
from jestr.utils.data import get_spec_featurizer, get_mol_featurizer, get_test_ms_dataset
from jestr.utils.models import get_model
from jestr.models.spec_encoder import SpecEncMLP_BIN
from jestr.models.mol_encoder import MolEnc
from concurrent.futures import ProcessPoolExecutor

PARAMS = {}
try:
    param_path = Path(__file__).parent / "params.yaml"
    with open(param_path) as f:
        PARAMS = yaml.load(f, Loader=yaml.FullLoader)
except:
    pass

def get_from_pubchem(in_smiles):
    """Retrieve candidate molecules from PubChem by molecular formula"""
    m = Chem.MolFromSmiles(in_smiles)
    form = Chem.rdMolDescriptors.CalcMolFormula(m)
    smiles = []
    try:
        compounds = pcp.get_compounds(form, 'formula', record_format='json')
    except:
        return smiles
    for compound in compounds:
        smiles.append(compound.canonical_smiles)
    return smiles


def load_variables(param_path=None):
    """Load configuration and initialize required components"""
    # Use relative path if not provided
    if param_path is None:
        param_path = Path(__file__).parent / "params.yaml"
    else:
        param_path = Path(param_path)
    
    if not param_path.exists():
        raise FileNotFoundError(f"params.yaml not found at {param_path}")
    
    with open(param_path) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Add missing components if needed
    dataset_builder = None
    molgraph_dict = None
    data_path = param_path.parent / "data"
    output = None
    
    return params, dataset_builder, molgraph_dict, data_path, device, output

def load_jestr_model(params, checkpoint_path, device):
    """Load pre-trained JESTR model from checkpoint"""
    try:
        params_obj = Namespace(**params)
        
        in_dim = detect_in_dim()
        
        # Create both encoders - don't pass extra args
        mol_model = MolEnc(params_obj, in_dim)
        spec_model = SpecEncMLP_BIN(params_obj)
        # Load checkpoints
        mol_ckpt = params.get('checkpoint_pth_mol_enc')
        if mol_ckpt and Path(mol_ckpt).exists():
            mol_state = torch.load(mol_ckpt, map_location=device, weights_only=False)
            mol_model.load_state_dict(mol_state, strict=False)
        
        spec_ckpt = params.get('checkpoint_pth_spec_enc')
        if spec_ckpt and Path(spec_ckpt).exists():
            spec_state = torch.load(spec_ckpt, map_location=device, weights_only=False)
            spec_model.load_state_dict(spec_state, strict=False)
        
        mol_model.to(device).eval()
        spec_model.to(device).eval()
        
        class EncoderPair:
            def __init__(self, mol, spec):
                self.mol_enc_model = mol
                self.spec_enc_model = spec
        
        return EncoderPair(mol_model, spec_model)
    
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise

def load_molecule_encoder(params, dataset_builder, molgraph_dict, output, device, data_path):
    """Load JESTR molecule encoder model"""
    try:
        # Convert dict to Namespace object so MolEnc can access params as attributes
        params_obj = Namespace(**params)
        
        in_dim = detect_in_dim()
        model = MolEnc(params_obj, in_dim)
        
        checkpoint_path = params.get('checkpoint_pth_mol_enc')
        if checkpoint_path and Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict, strict=False)
            print(f"✓ Loaded mol encoder from {checkpoint_path}")
        
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Error loading molecule encoder: {e}")
        raise

def load_spectrum_encoder(params, dataset_builder, molgraph_dict, output, device, data_path):
    """Load JESTR spectrum encoder model"""
    try:
        # Convert dict to Namespace object
        params_obj = Namespace(**params)
        
        # Don't pass bin_size - let the model calculate it from params
        model = SpecEncMLP_BIN(params_obj)  # REMOVE bin_size argument
        
        checkpoint_path = params.get('checkpoint_pth_spec_enc')
        if checkpoint_path and Path(checkpoint_path).exists():
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(state_dict, strict=False)
            print(f"✓ Loaded spec encoder from {checkpoint_path}")
        
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Error loading spectrum encoder: {e}")
        raise

def detect_in_dim():
    """Determine GNN input feature dim from params."""
    # Get from params first (preferred)
    if PARAMS.get('gnn_input_dim'):
        return PARAMS['gnn_input_dim']
    
    # Fallback to detecting from sample graph
    try:
        from Molecule import Molecule
        g = Molecule("CCO").get_molgraph(atom_feature="full", bond_feature="full")
        if g and "h" in g.ndata:
            detected = g.ndata["h"].shape[1]
            print(f"[INFO] Detected gnn_input_dim: {detected}")
            return detected
    except:
        pass
    
    # Last resort default
    print("[WARNING] Using default gnn_input_dim=78")
    return 78

def detect_bin_size():
    """Compute spectrum bin size from params: max_mz / resolution."""
    max_mz = PARAMS.get("max_mz", 1000)
    res = PARAMS.get("resolution", 1)
    if max_mz is None or res is None:
        return 1000  # Default
    return int(max_mz / res)


_spec_featurizer_cache = {}
_mol_featurizer_cache = {}

def get_cached_spec_featurizer(spectra_view, params):
    """Get or create cached spectrum featurizer"""
    key = spectra_view
    if key not in _spec_featurizer_cache:
        _spec_featurizer_cache[key] = get_spec_featurizer(spectra_view, params)
    return _spec_featurizer_cache[key]

def get_cached_mol_featurizer(molecule_view, params):
    """Get or create cached molecule featurizer"""
    key = molecule_view
    if key not in _mol_featurizer_cache:
        _mol_featurizer_cache[key] = get_mol_featurizer(molecule_view, params)
    return _mol_featurizer_cache[key]

def spectrum_encoder(model, params, mz_array, intensity_array, device):
    """Encode spectrum using JESTR spectrum encoder."""
    import torch
    import numpy as np
    import matchms
    
    # Get spec featurizer - returns dict: {'SpecBinnerLog': SpecBinnerLog(...)}
    spectra_view = params.get('spectra_view', 'SpecBinnerLog')
    spec_featurizer_dict = get_spec_featurizer(spectra_view, params)
    
    # Extract the actual featurizer object from dict
    if isinstance(spec_featurizer_dict, dict):
        spec_featurizer = spec_featurizer_dict[spectra_view]
    else:
        spec_featurizer = spec_featurizer_dict
    
    # Create spectrum dict
    spectrum = matchms.Spectrum(
        mz=np.array(mz_array, dtype=float),
        intensities=np.array(intensity_array, dtype=float)
    )
    
    
    # Call the featurizer - it's now callable
    spec_features = spec_featurizer(spectrum)
    
    # Convert to tensor
    spec_tensor = torch.tensor(spec_features, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model(spec_tensor)
    
    return embedding.squeeze(0)

def molecule_encoder(model, params, smiles, device):
    """Encode molecule using JESTR molecule encoder."""
    # Get mol featurizer - returns MolToGraph object directly (single view)
    mol_view = params.get('molecule_view', 'MolGraph')
    mol_featurizer = get_mol_featurizer(mol_view, params)
    
    # mol_featurizer is already callable (not a dict for single view)
    mol_graph = mol_featurizer(smiles)
    
    if mol_graph is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    mol_graph = mol_graph.to(device)
    
    with torch.no_grad():
        embedding = model(mol_graph)
    
    return embedding.squeeze(0)


def molecule_encoder_batch(model, params, smiles_dict, device):
    """Encode molecules using JESTR molecule encoder in parallel processes.
    
    Takes a dictionary of {id: smiles} and returns {id: embedding}.
    Processes in parallel batches of 50 SMILES.
    """
    
    def encode_single(smiles):
        """Encode a single SMILES string"""
        try:
            embedding = molecule_encoder(model, params, smiles, device)
            return embedding.cpu().numpy()
        except Exception as e:
            print(f"✗ Error encoding {smiles}: {e}")
            return None
    
    batch_size = 100
    embeddings_dict = {}
    
    smiles_items = list(smiles_dict.items())
    
    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=2) as executor:
        for batch_start in range(0, len(smiles_items), batch_size):
            batch_end = min(batch_start + batch_size, len(smiles_items))
            batch = smiles_items[batch_start:batch_end]
            
            # Submit batch for parallel processing
            futures = {
                executor.submit(encode_single, smiles): mol_id 
                for mol_id, smiles in batch
            }
            
            # Collect results
            for future in futures:
                mol_id = futures[future]
                embedding = future.result()
                if embedding is not None:
                    embeddings_dict[mol_id] = embedding
            
            print(f"Processed {batch_end}/{len(smiles_items)} molecules")
    
    return embeddings_dict

def compute_similarity_batch(model, params, spectrum_mz, spectrum_intensity, 
                             candidate_smiles_list, device):
    """
    Compute cosine similarity between spectrum and multiple candidate molecules.
    
    Input: model, params, mz array, intensity array, list of SMILES, device
    Output: similarity scores (1D tensor), valid SMILES indices
    """
    # Encode spectrum once
    spectrum_emb = spectrum_encoder(model, params, spectrum_mz, spectrum_intensity, device)
    
    similarities = []
    valid_indices = []
    
    for idx, smiles in enumerate(candidate_smiles_list):
        try:
            # Validate SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"⚠ Invalid SMILES: {smiles}")
                continue
            
            # Encode molecule
            mol_emb = molecule_encoder(model, params, smiles, device)
            
            # Compute cosine similarity
            sim = torch.nn.functional.cosine_similarity(
                spectrum_emb.unsqueeze(0),
                mol_emb.unsqueeze(0),
                dim=1
            )
            
            similarities.append(sim.item())
            valid_indices.append(idx)
            
        except Exception as e:
            print(f"✗ Error encoding {smiles}: {e}")
            continue
    
    if not similarities:
        raise ValueError("No valid candidates could be encoded")
    
    return torch.tensor(similarities), valid_indices


def rank_candidates(model, params, spectrum_mz, spectrum_intensity, 
                    candidate_smiles_list, top_k=10, device=None):
    """
    Rank candidate molecules by similarity to spectrum.
    
    Input: model, params, spectrum data, candidates, top_k, device
    Output: ranked candidates with scores
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    similarities, valid_indices = compute_similarity_batch(
        model, params, spectrum_mz, spectrum_intensity, 
        candidate_smiles_list, device
    )
    
    # Get top-k
    top_k = min(top_k, len(similarities))
    top_scores, top_idx_in_sims = torch.topk(similarities, top_k)
    
    # Map back to original indices
    top_idx = [valid_indices[i] for i in top_idx_in_sims.tolist()]
    
    results = [
        {
            'rank': i + 1,
            'smiles': candidate_smiles_list[idx],
            'similarity': top_scores[i].item()
        }
        for i, idx in enumerate(top_idx)
    ]
    
    return results


if __name__ == '__main__':
    # Load configuration
    params, device = load_variables('params.yaml')
    
    # Load model (provide path to your checkpoint)
    checkpoint_path = 'path/to/your/checkpoint.ckpt'
    model = load_jestr_model(params, checkpoint_path, device)
    
    # Example spectrum
    example_mz = np.array([100.0, 150.5, 200.3, 250.8, 300.1])
    example_intensity = np.array([0.5, 1.0, 0.3, 0.8, 0.2])
    example_smiles = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
    
    # Get candidates from PubChem
    print(f"Fetching candidates for: {example_smiles}")
    candidates = get_from_pubchem(example_smiles)
    print(f"✓ Found {len(candidates)} candidates")
    
    if candidates:
        # Rank candidates
        print("\nRanking candidates...")
        results = rank_candidates(
            model, params, example_mz, example_intensity, 
            candidates, top_k=10, device=device
        )
        
        # Display results
        print("\nTop 10 Results:")
        for result in results:
            print(f"  {result['rank']:2d}. {result['smiles'][:50]:<50s} | Similarity: {result['similarity']:.4f}")