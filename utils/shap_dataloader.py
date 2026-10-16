import torch
from torch.utils.data import Dataset, Subset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os
from PIL import Image

from utils.model_parameters import pycil_algs
from utils.setup_args import SHAPArgs


class ShapDataloader:
    def __init__(self, args: SHAPArgs):
        self.args = args
        self.algorithm = args.algorithm
        self.dataset = args.dataset_name
        self.mean = None
        self.std = None

    def get_indices(self, dataset, class_name):
        indices = []
        for i in range(len(dataset.targets)):
            for j in class_name:
                if dataset.targets[i] == j:
                    indices.append(i)
        return indices

    def load_data(self, desired_classes, imgs_per_class, shuffle=None, batch_size=256):
        transform = transforms.Compose(
            [transforms.ToTensor()])

        shap_set = torch.utils.data.Dataset()
        mean = None
        std = None

        if self.dataset == "mnist":
            if self.algorithm in pycil_algs:
                trsf = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
            else:
                trsf = transforms.Compose([transforms.ToTensor()])
            shap_set = datasets.MNIST(root=f"Datasets/{self.dataset}/", train=False, download=True, transform=trsf)
        elif self.dataset == "svhn":
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])

            shap_set = datasets.SVHN(root=f"Datasets/{self.dataset}/", split='test',
                                         download=True,
                                         transform=transforms.Compose([transforms.ToTensor(),
                                                                       transforms.Normalize(mean, std)]))
            shap_set.classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            shap_set.targets = shap_set.labels

        elif self.dataset == "cifar10":
            mean = torch.tensor([0.4914, 0.4822, 0.4465])
            std = torch.tensor([0.2023, 0.1994, 0.2010])

            shap_set = datasets.CIFAR10(root=f"Datasets/{self.dataset}/", train=False,
                                            download=True,
                                            transform=transforms.Compose(
                                                [transforms.ToTensor(),
                                                 transforms.Normalize(mean, std)]))

        elif self.dataset == "cifar100":
            mean = torch.tensor([0.5071, 0.4867, 0.4408])
            std = torch.tensor([0.2675, 0.2565, 0.2761])

            shap_set = datasets.CIFAR100(root=f"Datasets/{self.dataset}/", train=False,
                                             download=True,
                                             transform=transforms.Compose(
                                                 [transforms.ToTensor(),
                                                  transforms.Normalize(mean, std)]))

        elif self.dataset == "imagenet200":
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])

            idata = iImageNet200()
            idata.download_data()

            x, y = idata.test_data, idata.test_targets
            trsf = transforms.Compose([*idata.test_trsf, *idata.common_trsf])
            data, targets = [], []
            for idx in np.arange(0, 200):
                class_data, class_targets = _select(x, y, low_range=idx, high_range=idx + 1)
                data.append(class_data)
                targets.append(class_targets)
            data, targets = np.concatenate(data), np.concatenate(targets)
            shap_set = DummyDataset(data, targets, trsf, use_path=True, aug=1)

            shap_set.classes = range(200)
            shap_set.targets = shap_set.labels

        idx = self.get_indices(shap_set, desired_classes)

        subset = Subset(shap_set, idx)

        # Get samples per class
        #unique_labels, counts = torch.unique(torch.tensor(shap_set.targets), return_counts=True)

        # Create a DataLoader for the subset
        shap_dataloader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)

        if imgs_per_class is None:
            return shap_dataloader

        *_, images, labels = next(iter(shap_dataloader))

        # Order images and labels by class
        shap_idx = []
        shap_labels = []
        for i in range(len(desired_classes)):
            num = 0
            while len(shap_idx) < imgs_per_class * (i + 1):
                if labels[num] == desired_classes[i]:
                    shap_idx.append(num)
                    shap_labels.append(desired_classes[i])
                num += 1

        shap_imgs = images[shap_idx]

        self.mean, self.std = mean, std
        return shap_imgs, torch.tensor(shap_labels), shap_set.classes, mean, std

    def get_shap_train_set(self, dataset):
        '''
        Gets the train dataset for the shap explainer
        :param dataset: Desired dataset for shap explainer
        :return: train_set
        '''
        match dataset:
            case "mnist":
                if self.algorithm == "foster" or self.algorithm == "memo" or self.algorithm == "der":
                    trsf = transforms.Compose([transforms.Pad(2), transforms.ToTensor()])
                elif self.algorithm == "DGR":
                    trsf = transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(),
                                               transforms.Pad(2), transforms.ToTensor()])
                else:
                    trsf = transforms.Compose([transforms.ToTensor()])
                train_set = datasets.MNIST(root=f"datasets/mnist/", train=False, download=False, transform=trsf)
            case "cifar10":
                mean = torch.tensor([0.4914, 0.4822, 0.4465])
                std = torch.tensor([0.2023, 0.1994, 0.2010])

                train_set = datasets.CIFAR10(root=f"datasets/cifar10/", train=True,
                                             download=True,
                                             transform=transforms.Compose(
                                                 [transforms.ToTensor(),
                                                  transforms.Normalize(mean, std)]))
            case "cifar100":
                mean = torch.tensor([0.5071, 0.4867, 0.4408])
                std = torch.tensor([0.2675, 0.2565, 0.2761])

                train_set = datasets.CIFAR100(root=f"datasets/cifar100/", train=True,
                                              download=False,
                                              transform=transforms.Compose(
                                                  [transforms.ToTensor(),
                                                   transforms.Normalize(mean, std)]))
            case "imagenet200":
                idata = iImageNet200()
                idata.download_data()

                x, y = idata.train_data, idata.train_targets
                trsf = transforms.Compose([*idata.train_trsf, *idata.common_trsf])
                data, targets = [], []
                for idx in np.arange(0, 200):
                    class_data, class_targets = _select(x, y, low_range=idx, high_range=idx + 1)
                    data.append(class_data)
                    targets.append(class_targets)
                data, targets = np.concatenate(data), np.concatenate(targets)
                train_set = DummyDataset(data, targets, trsf, use_path=True,
                                         aug=2 if self.algorithm == "tagfex" else 1)

                train_set.classes = range(200)
                train_set.targets = train_set.labels
        return train_set

    def denormalize(self, img):
        return img * self.std[:, None, None] + self.mean[:, None, None]


# ---------From PyCIL Toolbox, used to load TinyImagenet (Imagenet200)---------#
class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False, aug=1):
        assert len(images) == len(labels), "Data size error!"
        self.aug = aug
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.aug == 1:
            if self.use_path:
                image = self.trsf(pil_loader(self.images[idx]))
            else:
                image = self.trsf(Image.fromarray(self.images[idx]))
            label = self.labels[idx]
            return idx, image, label
        else:
            if self.use_path:
                images = [self.trsf(pil_loader(self.images[idx])) for _ in range(self.aug)]
            else:
                images = [self.trsf(Image.fromarray(self.images[idx])) for _ in range(self.aug)]
            label = self.labels[idx]
            return idx, *images, label


# From PyCIL Toolbox, used to load TinyImagenet (Imagenet200)
def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def _select(x, y, low_range, high_range):
    idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]

    if isinstance(x, np.ndarray):
        x_return = x[idxes]
    else:
        x_return = []
        for id in idxes:
            x_return.append(x[id])
    return x_return, y[idxes]


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iImageNet200(iData):
    use_path = True
    transpose = False
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(200).tolist()

    ## Check this later...
    def organize_val_dataset(self):
        import os
        import shutil

        val_dir = 'datasets/imagenet200/val'
        val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')

        with open(val_annotations_file, 'r') as f:
            annotations = f.readlines()

        for line in annotations:
            parts = line.strip().split('\t')
            img_filename = parts[0]
            class_name = parts[1]

            # Create the class directory if it doesn't exist
            class_dir = os.path.join(val_dir, class_name)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            # Move the image
            src_path = os.path.join(val_dir, 'images', img_filename)
            dst_path = os.path.join(class_dir, img_filename)
            shutil.move(src_path, dst_path)

        # Remove the empty images directory and the annotations file
        shutil.rmtree(os.path.join(val_dir, 'images'))
        os.remove(val_annotations_file)

    def download_data(self):
        # assert 0, "You should specify the folder of your dataset"
        train_dir = "datasets/imagenet200/train"
        test_dir = "datasets/imagenet200/val"

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        if len(test_dset.classes) == 1:
            self.organize_val_dataset()
            test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)
# -----------------------------------------------------------------------------#