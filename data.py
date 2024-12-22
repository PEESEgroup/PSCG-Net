from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings
import pickle
import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from concurrent.futures import ThreadPoolExecutor

def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=1, pin_memory=False, **kwargs):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    total_size = len(dataset)

    if train_ratio is None:
        assert val_ratio + test_ratio < 1
        train_ratio = 1 - val_ratio - test_ratio
        print(f'[Warning] train_ratio is None, using 1 - val_ratio - '
                f'test_ratio = {train_ratio} as training data.')
    else:
        assert train_ratio + val_ratio + test_ratio <= 1

    indices = list(range(total_size))
    train_size = int(train_ratio * total_size)
    test_size = int(test_ratio * total_size)
    valid_size = int(val_ratio * total_size)

    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(
        indices[-(valid_size + test_size):-test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=val_sampler,
                            num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        test_loader = DataLoader(dataset, batch_size=batch_size,
                                 sampler=test_sampler,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=pin_memory)
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(dataset_list, scales=1):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (1, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, 1)
      Target value for prediction
    batch_cif_ids: list
    """
    # Pre-allocate lists for batch data
    batch_atom_fea, batch_target, batch_cif_ids = [], [], []
    batch_nbr_fea, batch_nbr_fea_idx = [], []
    crystal_atom_idx = []
    base_idx = 0

    # Iterate through dataset_list to collate data
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id) in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        
        # Append atom features
        batch_atom_fea.append(atom_fea)

        # Append neighbor features and indices, adjust indices by base_idx
        for idx in range(len(nbr_fea)):
            if len(batch_nbr_fea) <= idx:
                batch_nbr_fea.append([])
                batch_nbr_fea_idx.append([])
            batch_nbr_fea[idx].append(nbr_fea[idx])
            batch_nbr_fea_idx[idx].append(nbr_fea_idx[idx] + base_idx)

        # Append crystal to atom index mapping
        crystal_atom_idx.append(torch.arange(n_i, dtype=torch.long) + base_idx)
        base_idx += n_i

        # Append target and cif_id
        batch_target.append(target)
        batch_cif_ids.append(cif_id)

    # Concatenate all batch data
    batch_atom_fea = torch.cat(batch_atom_fea, dim=0)
    batch_nbr_fea = [torch.cat(batch_nbr_f, dim=0) for batch_nbr_f in batch_nbr_fea]
    batch_nbr_fea_idx = [torch.cat(batch_nbr_f_idx, dim=0) for batch_nbr_f_idx in batch_nbr_fea_idx]
    batch_target = torch.stack(batch_target, dim=0)

    return (batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, crystal_atom_idx), batch_target, batch_cif_ids

def collate_test_pool(dataset_list):
    batch_atom_fea, batch_cif_ids = [], []
    batch_nbr_fea, batch_nbr_fea_idx = None, None
    crystal_atom_idx, base_idx = [], 0
    # base_idx = [0 for _ in range(scales)]
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), cif_id) in enumerate(dataset_list):
        if batch_nbr_fea is None:
            batch_nbr_fea = [[] for _ in range(len(nbr_fea_idx))]
            batch_nbr_fea_idx = [[] for _ in range(len(nbr_fea_idx))]
            
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)

        for idx, (nbr_f, nbr_f_idx) in enumerate(zip(nbr_fea, nbr_fea_idx)):
            batch_nbr_fea[idx].append(nbr_f)
            batch_nbr_fea_idx[idx].append(nbr_f_idx+base_idx)
            
        new_idx = torch.LongTensor(np.arange(n_i)+base_idx)
        crystal_atom_idx.append(new_idx)
        base_idx += n_i

        batch_cif_ids.append(cif_id)

    batch_nbr_fea = [torch.cat(batch_nbr_f, dim=0) for batch_nbr_f in batch_nbr_fea]
    batch_nbr_fea_idx = [torch.cat(batch_nbr_f_idx, dim=0) for batch_nbr_f_idx in batch_nbr_fea_idx]

    return (torch.cat(batch_atom_fea, dim=0),
            batch_nbr_fea,
            batch_nbr_fea_idx,
            crystal_atom_idx),\
        batch_cif_ids

class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...

    id_prop.csv: a CSV file with two columns. The first column recodes a
    unique ID for each crystal, and the second column recodes the value of
    target property.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123, output_dir='./', task='regression'):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'

        file_path = os.path.join(output_dir, "data_list.pkl")
        if os.path.exists(file_path):
            with open(file_path, "rb") as file:
                self.id_prop_data = pickle.load(file)
                print("File loaded successfully.")
        else:
            id_prop_file = os.path.join(self.root_dir, 'id_prop.csv')
            assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
            with open(id_prop_file) as f:
                reader = csv.reader(f)
                self.id_prop_data = [row for row in reader]
            random.seed(random_seed)
            random.shuffle(self.id_prop_data)
            with open(file_path, "wb") as file:
                pickle.dump(self.id_prop_data, file)

        atom_init_file = os.path.join('init_weights', task, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = [GaussianDistance(dmin=dmin, dmax=r, step=step) for r in self.radius]

    def __len__(self):
        return len(self.id_prop_data)

    def get_nbr(self, crystal, max_num_nbr, radius, gdf):
        all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < max_num_nbr:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [radius + 1.] * (max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:max_num_nbr])))

        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = gdf.expand(nbr_fea)

        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)

        return nbr_fea, nbr_fea_idx

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir, 'cif', cif_id+'.cif'))
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)

        nbr_fea, nbr_fea_idx = [], []
        for i in range(len(self.radius)):
            nbr_f, nbr_f_idx = self.get_nbr(crystal, self.max_num_nbr[i], self.radius[i], self.gdf[i])
            nbr_fea.append(nbr_f)
            nbr_fea_idx.append(nbr_f_idx)

        atom_fea = torch.Tensor(atom_fea)
        target = torch.Tensor([float(target)])
        # target = torch.Tensor([float(eval(target)['reuss'])])
        # target = torch.Tensor([float('True' in target)])
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id



class CIFDataTest(Dataset):
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123, output_dir='./', task='regression'):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'

        entries = os.listdir(root_dir)
        self.id_prop_data = []
        
        # [entry for entry in entries if os.path.isfile(os.path.join(root_dir, entry))]

        # file_path = os.path.join(output_dir, "test_data_list.pkl")
        # if os.path.exists(file_path):
        #     with open(file_path, "rb") as file:
        #         self.id_prop_data = pickle.load(file)
        #         print("File loaded successfully.")
        # else:
        for entry in entries:
            if os.path.isfile(os.path.join(root_dir, entry)):
                try:
                    crystal = Structure.from_file(os.path.join(root_dir, entry))
                except:
                    continue
                self.id_prop_data.append(entry)

        # with open(file_path, "wb") as file:
        #     pickle.dump(self.id_prop_data, file)
                
        atom_init_file = os.path.join('init_weights', task, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = [GaussianDistance(dmin=dmin, dmax=r, step=step) for r in self.radius]

    def __len__(self):
        return len(self.id_prop_data)

    def get_nbr(self, crystal, max_num_nbr, radius, gdf):
        all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < max_num_nbr:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [radius + 1.] * (max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:max_num_nbr])))

        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = gdf.expand(nbr_fea)

        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)

        return nbr_fea, nbr_fea_idx

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cif_id = self.id_prop_data[idx]
        crystal = Structure.from_file(os.path.join(self.root_dir, cif_id))
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)

        nbr_fea, nbr_fea_idx = [], []
        for i in range(len(self.radius)):
            nbr_f, nbr_f_idx = self.get_nbr(crystal, self.max_num_nbr[i], self.radius[i], self.gdf[i])
            nbr_fea.append(nbr_f)
            nbr_fea_idx.append(nbr_f_idx)

        atom_fea = torch.Tensor(atom_fea)
        return (atom_fea, nbr_fea, nbr_fea_idx), cif_id


class CIFData2(Dataset):
    def __init__(self, root_dir, max_num_nbr=12, radius=8, dmin=0, step=0.2,
                 random_seed=123, output_dir='./', task_name='', task='regression'):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), 'root_dir does not exist!'

        self.task = task
        self.task_name = task_name

        # Load id_prop_data with caching
        file_path = os.path.join(output_dir, "data_list.pkl")
        if os.path.exists(file_path):
            with open(file_path, "rb") as file:
                self.id_prop_data = pickle.load(file)
                print("File loaded successfully.")
        else:
            id_prop_file = os.path.join(self.root_dir, 'attributes', f'{task_name}.csv')
            assert os.path.exists(id_prop_file), f'{task_name}.csv does not exist!'
            with open(id_prop_file) as f:
                reader = csv.reader(f)
                self.id_prop_data = [row for row in reader]
            random.seed(random_seed)
            random.shuffle(self.id_prop_data)
            with open(file_path, "wb") as file:
                pickle.dump(self.id_prop_data, file)

        # Cache atom_init and pre-load GaussianDistance
        atom_init_file = os.path.join('init_weights', task, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = [GaussianDistance(dmin=dmin, dmax=r, step=step) for r in self.radius]

        # Precompute cif file paths to avoid repetitive os.path.join
        self.cif_paths = {
            data[0]: os.path.join(self.root_dir, 'cif', data[0] + '.cif')
            for data in self.id_prop_data
        }

    def __len__(self):
        return len(self.id_prop_data)

    def get_nbr(self, crystal, max_num_nbr, radius, gdf):
        all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < max_num_nbr:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [radius + 1.] * (max_num_nbr - len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1], nbr[:max_num_nbr])))

        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = gdf.expand(nbr_fea)

        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)

        return nbr_fea, nbr_fea_idx

    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def load_crystal_structure(self, cif_id):
        return Structure.from_file(self.cif_paths[cif_id])

    def __getitem__(self, idx):
        cif_id, target = self.id_prop_data[idx]
        crystal = self.load_crystal_structure(cif_id)
        atom_fea = np.vstack([self.ari.get_atom_fea(crystal[i].specie.number)
                              for i in range(len(crystal))])
        atom_fea = torch.Tensor(atom_fea)

        nbr_fea, nbr_fea_idx = [], []
        for i in range(len(self.radius)):
            nbr_f, nbr_f_idx = self.get_nbr(crystal, self.max_num_nbr[i], self.radius[i], self.gdf[i])
            nbr_fea.append(nbr_f)
            nbr_fea_idx.append(nbr_f_idx)

        if 'shear_modulus' in self.task_name or 'bulk_modulus' in self.task_name:
            target = torch.Tensor([float(eval(target)['reuss'])])
        elif 'regression' in self.task:
            target = torch.Tensor([float(target)])
        else:
            target = torch.Tensor([float('True' in target)])

        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id

    # def preload_data(self):
    #     # Preload all crystal structures to memory (useful for small datasets)
    #     with ThreadPoolExecutor() as executor:
    #         self.preloaded_crystals = list(executor.map(self.load_crystal_structure, self.cif_paths.keys()))


