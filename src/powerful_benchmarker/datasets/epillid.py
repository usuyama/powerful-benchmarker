#! /usr/bin/env python3

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import os
import zipfile
from ..utils import common_functions as c_f
import logging
import tqdm

class EPillID(Dataset):
    url = 'https://github.com/usuyama/ePillID-benchmark/releases/download/ePillID_data_v1.0/ePillID_data.zip'
    filename = 'ePillID_data.zip'
    md5 = '265120c5c93d3ef9403cd03a67d78993'

    def __init__(self, root, transform=None, download=False):
        self.root = root
        if download:
            try:
                self.set_paths_and_labels(assert_files_exist=True)
            except:
                self.download_dataset()
                self.set_paths_and_labels()
        else:
            self.set_paths_and_labels()
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        output_dict = {"data": img, "label": label}
        return output_dict

    def load_labels(self):
        from pandas import read_csv
        attributes = read_csv(os.path.join(self.dataset_folder, "all_labels.csv"))
        self.class_names = attributes.label.tolist()
        self.name_to_label = {label: i for i, label in enumerate(np.unique(self.class_names))}
        self.labels = [self.name_to_label[name] for name in self.class_names]
        self.img_paths = [os.path.join(self.dataset_folder, "classification_data", x) for x in attributes.image_path]

    def set_paths_and_labels(self, assert_files_exist=False):
        self.dataset_folder = os.path.join(self.root, "ePillID_data")
        self.load_labels()
        assert len(np.unique(self.labels)) == 4902
        assert self.__len__() == 13532
        if assert_files_exist:
            logging.info("Checking if dataset images exist")
            for x in tqdm.tqdm(self.img_paths):
                assert os.path.isfile(x)

    def download_dataset(self):
        download_url(self.url, self.root, filename=self.filename, md5=self.md5)
        with zipfile.ZipFile(os.path.join(self.root, self.filename), 'r') as zip_ref:
            zip_ref.extractall(self.root, members = c_f.extract_progress(zip_ref))
