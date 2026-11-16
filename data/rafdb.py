import os
import numpy as np
from copy import deepcopy

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from data.data_utils import subsample_instances
from config import rafdb_root


def find_classes(label_file):
    """Parse RAF-DB label file, each line is '<filename> <label>'."""
    image_ids = []
    targets = []
    with open(label_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            split_line = line.split()
            image_ids.append(split_line[0])
            targets.append(int(split_line[1]))
    return image_ids, targets


class RAFDB(Dataset):
    """RAF-DB facial expression dataset (7 basic emotions).

    Expected directory layout:
        root/
            RAF-DB_train/
                EmoLabel/train_label.txt
                Image/aligned/train_00001.jpg
            RAF-DB_test/
                EmoLabel/test_label.txt
                Image/aligned/test_0001.jpg
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 loader=default_loader):

        root = os.path.expanduser(root)
        self.root = os.path.join(root, 'RAF-DB_train' if train else 'RAF-DB_test')
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.train = train

        label_file = os.path.join(self.root, 'EmoLabel',
                                  'train_label.txt' if train else 'test_label.txt')
        image_ids, targets = find_classes(label_file)

        # 图片存放在 Image/aligned 下，直接以文件名拼接
        self.samples = [(os.path.join(self.root, 'Image', 'aligned', img_id), t)
                        for img_id, t in zip(image_ids, targets)]
        self.uq_idxs = np.array(range(len(self.samples)))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, self.uq_idxs[index]

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str


def subsample_dataset(dataset, idxs):
    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True
    dataset.samples = [s for i, s in enumerate(dataset.samples) if i in idxs]
    dataset.uq_idxs = dataset.uq_idxs[mask]
    return dataset


def subsample_classes(dataset, include_classes=range(4)):
    cls_idxs = [i for i, (p, t) in enumerate(dataset.samples) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):
    all_targets = [t for (p, t) in train_dataset.samples]
    train_classes = np.unique(all_targets)

    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(np.array(all_targets) == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=(int(val_split * len(cls_idxs)),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_rafdb_datasets(train_transform, test_transform, train_classes=range(4),
                       prop_train_labels=0.8, split_train_val=False, seed=0):

    np.random.seed(seed)

    # 完整训练集（包含所有 7 类）
    whole_training_set = RAFDB(root=rafdb_root, transform=train_transform, train=True)

    # 有标签训练集：只保留 known 类，再随机采样一部分样本带标签
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # 划分 train / val
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # 无标签数据 = 完整训练集 减去 有标签训练集（即 unknown 类 + known 类里没标签的样本）
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # 测试集（包含所有 7 类）
    test_dataset = RAFDB(root=rafdb_root, transform=test_transform, train=False)

    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets


if __name__ == '__main__':

    x = get_rafdb_datasets(None, None, split_train_val=False,
                           train_classes=range(4), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    train_labelled_targets = set(t for (p, t) in x['train_labelled'].samples)
    train_unlabelled_targets = set(t for (p, t) in x['train_unlabelled'].samples)
    print(f'Num Labelled Classes: {len(train_labelled_targets)}')
    print(f'Num Unlabelled Classes: {len(train_unlabelled_targets)}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')
