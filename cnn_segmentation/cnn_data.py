import torch
from torch.utils.data import Dataset


class CNNData(Dataset):
    
    def __init__(self, env_patches, segmentation_patches): 
    
        self.x = torch.tensor(env_patches, requires_grad=True).type(torch.float32)

        
        self.y = torch.from_numpy(segmentation_patches).type(torch.int64) 

        # Sets length
        self.len = self.x.shape[0]
        
    
    # Overrides get method
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        return sample
    
    # Overrides len() method
    def __len__(self):
        return self.len
