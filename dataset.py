import boto3
from features import *
import numpy as np
from os import mkdir
from os.path import join
import pandas as pd
import time
import torch
from torch import multiprocessing as mp
from torch.utils.data.dataset import random_split
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from utils import file_in_s3


s3 = boto3.resource('s3').meta.client
class MultiPathDataset(Dataset):
    
    
    def __init__(self, filename, root='.', num_workers=1, use_s3=True):
           
        self.filename = filename
        
        self.data_name = filename.strip('.csv')
        self.prefix = f"proc_{self.data_name}"
        self.csv_file = pd.read_csv(f'raw/{filename}')
        self.num_molecules = self.csv_file.shape[0]
        self.num_workers = num_workers
        self.use_s3 = use_s3
        super().__init__(root)
        
        
    @property
    def raw_file_names(self):
        
        return self.filename
    
    
    @property
    def processed_file_names(self):
    
        return [f"{self.prefix}{i}.pt" for i in range(self.num_molecules)]
    
    
    def download(self):
        
        """
        This function only runs if the raw CSV file is not found locally in raw_dir. Ensure it has been uploaded to S3.
        """
        
        if not self.use_s3:
            
            raise FileNotFoundError("Error: S3 not enabled, ensure file is available locally.")
            
        s3.download_file('multipath', self.filename, join(self.raw_dir, self.filename))
        
        
    def mol_process(self, frame_idx, frames):
        """Helper function to process data, used in process pool"""
        attrs = frames[frame_idx][['label', 'SMILES']].values
        index = frames[frame_idx].index
        for idx, (label, smile) in zip(index, attrs):
            
            graph = Data(x=get_atom_features(smile),
                         edge_index=get_edge_index(smile),
                         bond_types=get_bond_types(smile),
                         coords=get_atom_coords(smile),
                         y=int(label))

            processed_name = join(self.processed_dir, f"{self.prefix}{idx}.pt")
            torch.save(graph, processed_name)

            if self.use_s3:
                s3.upload_file(processed_name, 'multipath', f"{self.prefix}/{self.prefix}{idx}.pt")
                
                
    def process(self):
        
        """
        This function only runs if the processed files are not found in processed_dir.
        """
        # Check S3 for the files and download if available
        if self.use_s3 and file_in_s3(s3, 'multipath', self.prefix):
            
            print("Data already processed, downloading from S3...")
            
            for idx in range(self.num_molecules):
                s3.download_file('multipath', f"{self.prefix}/{self.prefix}{idx}.pt", join(self.processed_dir, f"{self.prefix}{idx}.pt"))
                
            return
        
        # If the files aren't saved in S3, preprocess data and save results to S3.
        data = self.csv_file
        start_time = time.time()
        frames = np.array_split(data, self.num_workers)
        
        mp.spawn(self.mol_process, args=(frames,), nprocs=self.num_workers)
        
        end_time = time.time()
        print(f"Total preprocessing time: {end_time - start_time}")
        
        
    def get(self, idx):
        
        return torch.load(join(self.processed_dir, f"{self.prefix}{idx}.pt"))
        
    
    def len(self):
        
        return self.num_molecules

    
    def get_dataloaders(self, lengths, train_batch_size=1, test_batch_size=1, num_workers=1, already_split=False, shuffle=False):
    
        if not already_split:
            train_data, test_data = random_split(self, lengths)
            
            return (DataLoader(train_data, batch_size=train_batch_size, num_workers=num_workers, shuffle=True),
                       DataLoader(test_data, batch_size=test_batch_size, num_workers=num_workers, shuffle=False))
              
        final_loader = DataLoader(self, batch_size=train_batch_size, num_workers=num_workers, shuffle=shuffle)
    
        return final_loader
