from pathlib import Path

import pandas as pd
import torch
from PIL import Image


class MyDataset(torch.utils.data.Dataset):
    """CSV (columns: image_name, label) based image classification dataset.

    image_name may include the file extension. If it does not, .jpg and .png are tried.
    label must be an integer starting from 0.
    """

    def __init__(self, img_dir, csv_file, split = None, transform=None, image_col="filepath", label_col="cancer", split_col="split"):
        self.img_dir = Path(img_dir)

        df = pd.read_csv(csv_file)

        if split is not None:
            df = df[df[split_col] == split].reset_index(drop=True)
        

        self.data = df
        self.transform = transform
        self.image_ids = self.data[image_col].astype(str).values
        self.targets = self.data[label_col].astype(int).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.get_image_path(idx)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.targets[idx]

    def get_image_name(self, idx):
        return str(self.image_ids[idx])

    def get_image_path(self, idx):
        return self.resolve_image_path(self.get_image_name(idx))

    def resolve_image_path(self, image_name):
        path = self.img_dir / image_name
        if path.exists():
            return path
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = self.img_dir / f"{image_name}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not find image {image_name} in {self.img_dir}.")
