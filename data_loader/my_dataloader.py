import torch

import json
from torch_geometric.loader import DataLoader
from torchvision import datasets, transforms

from .format_data import load_geom_data
from .format_data import combining_ensembles

label_norms = {'labels': [0.5540915131568909, 0.49100208282470703]}

class EnsDataLoader(DataLoader):
    """
    EnsDataLoader that uses the pytorch geometric dataloader as its base
    """
    def __init__(self, ens_dirs, batch_size, shuffle=False, num_workers=1, training=True):

        #trsfm = transforms.Compose([
            #transforms.Normalize((-0.5,), (0.5,))
        #])
        self.ens_dirs = ens_dirs


        self.struct1, self.struct2, ddg_df=combining_ensembles(ens_dirs)
        
        self.labels=torch.tensor(ddg_df.max_diff_ddg).float()
        
        self.dataset1 = load_geom_data(self.struct1)

        self.dataset2 = load_geom_data(self.struct2)
        
        super().__init__(self, batch_size, shuffle, num_workers)

