import numpy as np
from torchvision import datasets, transforms
import torch
from utils.synthetic_digits import SyntheticDigits

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

class iCIFAR10(iData):
    use_path = False

    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        Cutout(16)
    ]
    test_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]

    
    
    common_trsf = [
    ]

    class_order = np.arange(10).tolist()

    def download_data(self, dataset_partition = 1):
        train_dataset = datasets.cifar.CIFAR10("../../data/cifar10", train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10("../../data/cifar10", train=False, download=True)
        print(len(train_dataset))
        
        if dataset_partition != 1:
            class_indices = {}
            for index, (image, label) in enumerate(train_dataset):
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(index)

            subset_indices = []
            for label in class_indices:
                subset_indices.extend(class_indices[label][:len(class_indices[label]) // dataset_partition])

            print(len(subset_indices))

            
            sub_train_dataset = np.array([train_dataset.data[i] for i in subset_indices])


            self.train_data, self.train_targets = sub_train_dataset, np.array(
                [train_dataset.targets[i] for i in subset_indices]
            )
            self.test_data, self.test_targets = test_dataset.data, np.array(
                test_dataset.targets
            )
        else:
            self.train_data, self.train_targets = train_dataset.data, np.array(
                train_dataset.targets
            )
            self.test_data, self.test_targets = test_dataset.data, np.array(
                test_dataset.targets
            )


class iSVHN(iData):
    use_path = False
    train_trsf = []
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    class_order = np.arange(10).tolist()
    
    def change(self, data):
        tmp = data # 3,32,32 (C,W,H)
        trans_img = tmp.transpose(1,2,0)  # 32,32,3 (W,H,C)
        trans_img = trans_img.astype('uint8')
        return trans_img

    def download_data(self, dataset_partition = 1):
        train_dataset = datasets.SVHN("../../data/svhn", split='train', download=True)
        test_dataset = datasets.SVHN("../../data/svhn", split='test', download=True)
        print(len(train_dataset))
        
        whc_train_dataset = np.array([self.change(train_dataset.data[i]) for i in range(len(train_dataset.data))])
        whc_test_dataset = np.array([self.change(test_dataset.data[i]) for i in range(len(test_dataset.data))])
        
        
        
        if dataset_partition != 1:
            class_indices = {}
            for index, (image, label) in enumerate(train_dataset):
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(index)

            subset_indices = []
            for label in class_indices:
                subset_indices.extend(class_indices[label][:len(class_indices[label]) // dataset_partition])

            print(len(subset_indices))

            sub_train_dataset = np.array([whc_train_dataset[i] for i in subset_indices])


            self.train_data, self.train_targets = sub_train_dataset, np.array(
                [train_dataset.labels[i] for i in subset_indices]
            )
            self.test_data, self.test_targets = whc_test_dataset, np.array(
                test_dataset.labels
            )
        else:
            self.train_data, self.train_targets = whc_train_dataset, np.array(
                train_dataset.labels
            )
            self.test_data, self.test_targets = whc_test_dataset, np.array(
                test_dataset.labels
            )



class iFashionMNIST(iData):
    use_path = False
    train_trsf = []
    test_trsf = []
    common_trsf = [
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    class_order = np.arange(10).tolist()

    
    def download_data(self, dataset_partition = 1):
        
        train_dataset = datasets.FashionMNIST("../../data/fashionmnist", train=True, download=True)
        test_dataset = datasets.FashionMNIST("../../data/fashionmnist", train=False, download=True)
        print(len(train_dataset))
        
        if dataset_partition != 1:
            class_indices = {}
            for index, (image, label) in enumerate(train_dataset):
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(index)

            subset_indices = []
            for label in class_indices:
                subset_indices.extend(class_indices[label][:len(class_indices[label]) // dataset_partition])

            print(len(subset_indices))
            print(type(train_dataset.data))

            
            sub_train_dataset = np.array(train_dataset.data[subset_indices])


            self.train_data, self.train_targets = sub_train_dataset, np.array(
                [train_dataset.targets[i] for i in subset_indices]
            )
            self.test_data, self.test_targets = test_dataset.data, np.array(
                test_dataset.targets
            )
        else:
            self.train_data, self.train_targets = train_dataset.data, np.array(
                train_dataset.targets
            )
            self.test_data, self.test_targets = test_dataset.data, np.array(
                test_dataset.targets
            )
         

class iEMNIST(iData):
    use_path = False
    train_trsf = []
    test_trsf = []
    common_trsf = [
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    
    class_order = np.arange(47).tolist()

    def download_data(self, dataset_partition = 1):
        train_dataset = datasets.EMNIST("../../data/emnist", train=True, split='balanced', download=True)
        test_dataset = datasets.EMNIST("../../data/emnist", train=False, split='balanced', download=True)
        print(len(train_dataset))
        
        if dataset_partition != 1:
            class_indices = {}
            for index, (image, label) in enumerate(train_dataset):
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(index)

            subset_indices = []
            for label in class_indices:
                subset_indices.extend(class_indices[label][:len(class_indices[label]) // dataset_partition])

            print(len(subset_indices))

            
            sub_train_dataset = np.array(train_dataset.data[subset_indices])


            self.train_data, self.train_targets = sub_train_dataset, np.array(
                [train_dataset.targets[i] for i in subset_indices]
            )
            self.test_data, self.test_targets = test_dataset.data, np.array(
                test_dataset.targets
            )
        else:
            self.train_data, self.train_targets = train_dataset.data, np.array(
                train_dataset.targets
            )
            self.test_data, self.test_targets = test_dataset.data, np.array(
                test_dataset.targets
            )         
   
   
class iSYNNUM(iData):
    use_path = False
    train_trsf = []
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    
    # 479400
    class_order = np.arange(10).tolist()

    def download_data(self, dataset_partition = 1):
        train_dataset = SyntheticDigits("../../data/synnum", train=True, download=False)
        test_dataset = SyntheticDigits("../../data/synnum", train=False, download=False)
        print(len(train_dataset))
        
        if dataset_partition != 1:
            class_indices = {}
            for index, (image, label) in enumerate(train_dataset):
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(index)

            subset_indices = []
            for label in class_indices:
                subset_indices.extend(class_indices[label][:len(class_indices[label]) // dataset_partition])

            print(len(subset_indices))

            
            sub_train_dataset = np.array(train_dataset.data[subset_indices])


            self.train_data, self.train_targets = sub_train_dataset, np.array(
                [train_dataset.targets[i] for i in subset_indices]
            )
            self.test_data, self.test_targets = test_dataset.data, np.array(
                test_dataset.targets
            )
        else:
            self.train_data, self.train_targets = train_dataset.data, np.array(
                train_dataset.targets
            )
            self.test_data, self.test_targets = test_dataset.data, np.array(
                test_dataset.targets
            )         
