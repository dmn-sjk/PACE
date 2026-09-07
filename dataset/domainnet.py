import os
from PIL import Image
from torchvision import transforms

import os
import json
from PIL import Image
from torch.utils.data import Dataset
from typing import Sequence, Callable, Optional


class ImageList(Dataset):
    def __init__(self, image_root: str, label_files: Sequence[str], transform: Optional[Callable] = None, split: str = "test"):
        self.image_root = image_root
        self.label_files = label_files
        self.transform = transform

        self.samples = []
        for file in label_files:
            if file.endswith(".json"):
                self.samples += self.build_index_json(label_file=file, split=split)
            else:
                self.samples += self.build_index(label_file=file)

    def build_index(self, label_file):
        """Build a list of <image path, class label, domain name> items.
        Input:
            label_file: Path to the file containing the image label pairs
        Returns:
            item_list: A list of <image path, class label> items.
        """
        with open(label_file, "r") as file:
            tmp_items = [line.strip().split() for line in file if line]

        item_list = []
        for img_file, label in tmp_items:
            img_file = f"{os.sep}".join(img_file.split("/"))
            img_path = os.path.join(self.image_root, img_file)
            domain_name = img_file.split(os.sep)[0]
            item_list.append((img_path, int(label), domain_name))

        return item_list

    def build_index_json(self, label_file, split):
        item_list = []
        with open(label_file) as fp:
            splits = json.load(fp)
            for sample in splits[split]:
                img_path = os.path.join(self.image_root, sample[0])
                item_list.append((img_path, sample[1], split))

        return item_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, domain = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label

class DomainNet126(ImageList):
    NUM_CLASSES = 126
    ENVIRONMENTS = ["clipart", "painting", "real", "sketch"]
    def __init__(self, root, domain):
        domainnet_dir = os.path.join(root, "DomainNet-126/")
        data_files = [os.path.join("dataset", "domainnet126_lists", domain + "_list.txt")]
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        super().__init__(domainnet_dir, data_files, transform)
