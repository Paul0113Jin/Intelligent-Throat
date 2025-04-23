# datamodule.py
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import pytorch_lightning as pl
from Normalization import Normalizer
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    """Basic Dataset class."""
    def __init__(self, X, y):
        if X.ndim == 2:
             X = X[..., None] # Add channel dim: [samples, window, 1]
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MyDataModule(pl.LightningDataModule):
    def __init__(self, config: dict, stage: str):
        super().__init__()
        self.cfg = config['data']
        self.stage = stage # 'pretrain', 'finetune', 'distill', 'test'

        self.train_path = self.cfg['train_data_path']
        self.finetune_path = self.cfg['finetune_data_path']
        self.test_path = self.cfg['test_data_path']
        self.dump_path = self.cfg['dump_path']
        self.batch_size = self.cfg['batch_size']
        self.num_workers = self.cfg['num_workers']
        self.shuffle_train = self.cfg['shuffle_train']
        self.val_split = self.cfg['validation_split']
        self.norm_method = self.cfg['normalization_method']
        self.fill_nan = self.cfg['fill_nan']

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        os.makedirs(self.dump_path, exist_ok=True)

    def _load_raw_data(self, path):
        print(f"Loading raw data from: {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        return pd.read_csv(path)

    def _preprocess_and_save(self, data: pd.DataFrame, filename_prefix: str):
        print(f"Preprocessing data for '{filename_prefix}'...")
        x_raw = data.iloc[:, :-1].values # [samples, window]
        y = data.iloc[:, -1].values   # [samples]

        normalized_slices = []
        for i in range(x_raw.shape[0]):
            raw_slice = x_raw[i, :]
            normalizer = Normalizer(raw_slice)
            slice_norm = normalizer.normalize(method=self.norm_method, fill_nan=self.fill_nan)
            normalized_slices.append(slice_norm)

        X_norm = np.array(normalized_slices)
        X_norm = X_norm[..., None]

        # Save preprocessed data
        x_save_path = os.path.join(self.dump_path, f'{filename_prefix}_X.npy')
        y_save_path = os.path.join(self.dump_path, f'{filename_prefix}_y.npy')
        np.save(x_save_path, X_norm)
        np.save(y_save_path, y)
        print(f"Saved preprocessed data to {self.dump_path}")
        return X_norm, y

    def prepare_data(self):

        required_prefixes = set()
        if self.stage in ['pretrain', 'finetune', 'distill']:
             required_prefixes.add('train') 
             required_prefixes.add('finetune') 
        if self.stage == 'test':
            required_prefixes.add('test')

        data_paths = {
            'train': self.train_path,
            'finetune': self.finetune_path,
            'test': self.test_path,
        }

        for prefix in required_prefixes:
            x_path = os.path.join(self.dump_path, f'{prefix}_X.npy')
            y_path = os.path.join(self.dump_path, f'{prefix}_y.npy')

            if not os.path.exists(x_path) or not os.path.exists(y_path):
                print(f"Preprocessed files for '{prefix}' not found. Generating...")
                raw_data_path = data_paths.get(prefix)
                if raw_data_path:
                    raw_data = self._load_raw_data(raw_data_path)
                    self._preprocess_and_save(raw_data, prefix)
                else:
                    print(f"Warning: Raw data path for '{prefix}' not specified in config, but preprocessing requested.")
            else:
                 print(f"Preprocessed files found for '{prefix}' in {self.dump_path}")


    def setup(self, stage: str = None):

        current_stage = stage or self.stage
        print(f"Setting up DataModule for stage: {current_stage}")

        seed = pl.seed_everything(None, workers=True)

        if current_stage == 'fit' or current_stage == 'validate':
            if self.stage == 'pretrain':
                data_prefix = 'train'
            elif self.stage in ['finetune', 'distill']:
                data_prefix = 'finetune'
            else:
                raise ValueError(f"Invalid internal stage '{self.stage}' for fit/validate setup.")

            x_path = os.path.join(self.dump_path, f'{data_prefix}_X.npy')
            y_path = os.path.join(self.dump_path, f'{data_prefix}_y.npy')
            print(f"Loading preprocessed data (assuming files exist): {x_path}, {y_path}")
            X = np.load(x_path)
            y = np.load(y_path)
            full_dataset = CustomDataset(X, y)
            dataset_len = len(full_dataset)

            if self.val_split > 0.0 and dataset_len > 0:
                print(f"Performing stratified split ({1-self.val_split:.1%} Train / {self.val_split:.1%} Val)...")
                indices = list(range(dataset_len))
                train_idx, val_idx = train_test_split(
                    indices,
                    test_size=self.val_split,
                    stratify=y,
                    random_state=seed
                )
                self.train_dataset = Subset(full_dataset, train_idx)
                self.val_dataset = Subset(full_dataset, val_idx)
                print(f"Split complete: Train={len(self.train_dataset)}, Validation={len(self.val_dataset)}")
            elif dataset_len > 0 :
                print(f"Using full {data_prefix} data for training. No validation split.")
                self.train_dataset = full_dataset
                self.val_dataset = Subset(full_dataset, list(range(min(1, dataset_len)))) # Minimal dummy validation
                print("Warning: No validation split. Using a small subset for validation.")
            else:
                 print(f"Warning: Dataset for prefix '{data_prefix}' is empty (based on loaded files).")
                 self.train_dataset = None
                 self.val_dataset = None


        elif current_stage == 'test':
            data_prefix = 'test'
            x_path = os.path.join(self.dump_path, f'{data_prefix}_X.npy')
            y_path = os.path.join(self.dump_path, f'{data_prefix}_y.npy')
            print(f"Loading preprocessed test data (assuming files exist): {x_path}, {y_path}")
            X = np.load(x_path)
            y = np.load(y_path)
            self.test_dataset = CustomDataset(X, y)

        else:
            print(f"DataModule setup called with stage: {current_stage}. No specific action taken.")


    def train_dataloader(self):
        if self.train_dataset is None:
            print("Warning: train_dataset is None in train_dataloader. Calling setup('fit').")
            self.setup('fit')
        if self.train_dataset is None:
             raise RuntimeError("train_dataset is still None after setup. Cannot create DataLoader.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            print("Warning: val_dataset is None in val_dataloader. Calling setup('fit').")
            self.setup('fit')
        if self.val_dataset is None:
            print("Warning: val_dataset is None. Returning None for validation dataloader.")
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            self.setup('test')
        if self.test_dataset is None:
             raise RuntimeError("test_dataset is None. Cannot create DataLoader.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )