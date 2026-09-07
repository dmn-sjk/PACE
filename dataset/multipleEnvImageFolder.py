import os
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Union

from torchvision import transforms
from torchvision.datasets import ImageFolder



class MultipleEnvironmentImageFolder(Mapping[str, ImageFolder]):
    """
    A lightweight container mapping environment names -> torchvision `ImageFolder`.

    Notes:
    - Instances are **not** themselves a PyTorch Dataset; each environment is.
    - Supports both string indexing (`ds["photo"]`) and integer indexing
      (`ds[0]` -> first environment in `self.environments`).
    """

    # Subclasses may override to enforce a specific environment order.
    ENVIRONMENTS: Optional[Sequence[str]] = None

    def __init__(
        self,
        root: str,
        test_envs: Optional[Sequence[int]],
        augment: bool,
        hparams=None,
    ):
        self.root = root
        self.test_envs = set(test_envs or [])
        self.augment = augment
        self.hparams = hparams or {}

        envs_on_disk = sorted([f.name for f in os.scandir(root) if f.is_dir()])
        if self.ENVIRONMENTS is not None:
            missing = [e for e in self.ENVIRONMENTS if e not in envs_on_disk]
            if missing:
                raise FileNotFoundError(
                    f"Missing environment directories under {root!r}: {missing}. "
                    f"Found: {envs_on_disk}"
                )
            environments = list(self.ENVIRONMENTS)
        else:
            environments = envs_on_disk

        self.environments: List[str] = environments

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets: Dict[str, ImageFolder] = {}
        for i, environment in enumerate(self.environments):
            if self.augment and (i not in self.test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path, transform=env_transform)

            self.datasets[environment] = env_dataset

    def __getitem__(self, key: Union[str, int]) -> ImageFolder:
        if isinstance(key, int):
            env = self.environments[key]
            return self.datasets[env]
        return self.datasets[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.datasets)

    def __len__(self) -> int:
        return len(self.datasets)
