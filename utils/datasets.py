# Imports
from utils.model_parameters import iTAMLArgs


def get_dataset_params(dataset):
    if dataset == "cifar10":
        return SHAPCifar10
    if dataset == "cifar100":
        return SHAPCifar100
    if dataset == "imagenet200":
        return SHAPTinyImageNet


class SHAPDataset:
    class_per_task = 2
    num_class = 10
    num_task = 5
    shap_samples = 100

    ###---Change made to allow for non-uniform task class sizes---###
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "init_cls" not in cls.__dict__:
            cls.init_cls = cls.class_per_task
    ###-----------------------------------------------------------###

class SHAPCifar10(SHAPDataset):
    class_per_task = 2
    num_class = 10
    num_task = 5
    shap_samples = 100

class SHAPCifar100(SHAPDataset):
    class_per_task = 10
    num_class = 100
    num_task = 10
    shap_samples = 20

class SHAPTinyImageNet(SHAPDataset):
    class_per_task = 20
    num_class = 200
    num_task = 10
    shap_samples = 20


from torchvision.datasets.vision import VisionDataset
from typing import Any

class Bootstrap(VisionDataset):
    """

    Args:

    """

    def __init__(self, images, labels, orig_idx=None) -> None:

        super().__init__(images, labels)

        self.data = images
        self.targets = labels
        if orig_idx is None:
            self.orig_idx = torch.arange(len(images)).tolist()
        else:
            self.orig_idx = orig_idx

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


    def __len__(self) -> int:
        return len(self.data)
