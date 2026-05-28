from format_data import combining_ensembles
from torch_geometric.data import Dataset
from pathlib import Path

class EnsDataset(Dataset):

    def __init__(self, ens_dirs, transform=None,target_transform=None):
        self.transform = transform 
        self.target_transform = target_transform 
        
        self.data1,self.data2,self.ddg_df=combining_ensembles(ens_dirs)

        self.labels=self.ddg_df.max_diff_ddg
            
    
    def __len__(self):
        # Returns the number of samples
        return len(self.labels)


    def __getitem__(self, idx):

        # Here we need to use index to:
        # 1. grab the corresponding image and label
        # 2. run any transforms on the image
        # 3. return the transformed image and label as tensors #this part feels very important to me
        struct1_point=self.data1.loc[idx,:]
        struct2_point=self.data2.loc[idx,:]
        label=self.labels[idx]
        
        if self.transform:
            struct1_point=self.transform(struct1_point)
            struct2_point=self.transform(struct2_point)
        if self.target_transform:
            label=self.target_transform(label)
        return struct1_point, struct2_point, label
