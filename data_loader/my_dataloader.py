from torch_geometric.loader import DataLoader
from torchvision import datasets, transforms

from format_data import load_geom_data
from format_data import combining_ensembles



class EnsDataLoader(DataLoader):
    """
    EnsDataLoader that uses the pytorch geometric dataloader as its base
    """
    def __init__(self, ens_dirs, batch_size, shuffle=False, num_workers=1, training=True):

        trsfm = transforms.Compose([
            transforms.Normalize((-0.5,), (0.5,))
        ])
        self.ens_dirs = ens_dirs

        self.struct1, self.struct2,ddg_df=combining_ensembles(ens_dirs)
        
        self.labels=ddg_df.max_diff_ddg
        
        self.dataset = load_geom_data(self.struct1)
        
        super().__init__(self.dataset1, batch_size, shuffle, num_workers)
