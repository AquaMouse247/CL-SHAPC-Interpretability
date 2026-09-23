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
