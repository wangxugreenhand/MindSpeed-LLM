import os
import torch
import math
import functools
import numpy as np
import torch.distributed as dist
from operator import add
from functools import reduce
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# from partition_data import DataPartitioner
# from prepare_data import get_dataset

def load_data_batch(_input, _target):
    """Load a mini-batch and record the loading time."""
    _input, _target = _input.cuda(), _target.cuda()
    return _input, _target

def define_dataset(rank, world_size, args, force_shuffle=False):
    return define_cv_dataset(rank, world_size, args, force_shuffle)

def define_cv_dataset(rank, world_size, args, force_shuffle):
    # print("Create dataset: {} for rank {}.".format(conf.data, conf.graph.rank))
    train_dataset = get_dataset(args.data , args.data_dir, split="train")
    val_dataset = get_dataset(args.data , args.data_dir, split="test")

    train_loader, train_partitioner = _define_cv_dataset(
        rank, world_size, args,
        partition_type=args.partition_data,
        dataset=train_dataset,
        dataset_type="train",
        force_shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=100, pin_memory=True,shuffle=False, num_workers=2)

    '''
    val_loader, val_partitioner = _define_cv_dataset(
        rank, world_size, args,
        partition_type=args.partition_data,
        dataset=val_dataset,
        dataset_type="test",
        force_shuffle=True,
    )
    '''
    # _get_cv_data_stat(conf, train_loader, val_loader)
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_partitioner": train_partitioner,
        # "val_partitioner": val_partitioner,
    }


def _define_cv_dataset(
    rank, 
    world_size,
    args,
    partition_type,
    dataset,
    dataset_type,
    force_shuffle=False,
    data_to_load=None,
    task=None,
):
    """ Given a dataset, partition it. """
    if data_to_load is None:
        # determine the data to load,
        # either the whole dataset, or a subset specified by partition_type.
        if partition_type is not None : #and conf.distributed:
            partition_sizes = [1.0 / world_size for _ in range(world_size)]

            partitioner = DataPartitioner(
                rank, 
                world_size,
                args,
                dataset,
                partition_sizes,
                partition_type=args.partition_data,
                consistent_indices=True
            )
            data_to_load = partitioner.use(rank)
            print("Data partition: partitioned data and use subdata.")
        else:
            partitioner = None
            data_to_load = dataset
            print("Data partition: used whole data.")
    else:
        print("Data partition: use inputed 'data_to_load'.")
        partitioner = None

    # use Dataloader.
    if rank == world_size:
        data_loader = torch.utils.data.DataLoader(
            data_to_load,
            batch_size=args.batch_size * world_size,
            shuffle=force_shuffle,
            # num_workers=conf.num_workers,
            num_workers=world_size,
            pin_memory=True,
            drop_last=False,
        )
    else:
        data_loader = torch.utils.data.DataLoader(
            data_to_load,
            batch_size=args.batch_size,
            shuffle=force_shuffle,
            # num_workers=conf.num_workers,
            num_workers=world_size,
            pin_memory=True,
            drop_last=False,
        )        
    if rank == world_size:
        print(
            (
                "Data stat: we have {} samples for {}, "
                + "load {} data for process (rank {}). "
                + "The batch size is {}, number of batches is {}."
            ).format(
                len(dataset),
                dataset_type,
                len(data_to_load),
                # conf.graph.rank,
                rank,
                args.batch_size * world_size,
                len(data_loader),
            )
        )
    else:
        print(
            (
                "Data stat: we have {} samples for {}, "
                + "load {} data for process (rank {}). "
                + "The batch size is {}, number of batches is {}."
            ).format(
                len(dataset),
                dataset_type,
                len(data_to_load),
                # conf.graph.rank,
                rank,
                args.batch_size,
                len(data_loader),
            )
        )        

    return data_loader, partitioner

# partition_data
class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, indices):
        self.data = data
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        data_idx = self.indices[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(
        self,
        rank,
        world_size,
        args,
        data,
        partition_sizes,
        partition_type,
        consistent_indices=True,
    ):
        # prepare info.
        self.rank = rank
        self.world_size = world_size
        self.args = args
        self.data = data
        self.partition_sizes = partition_sizes
        self.partition_type = partition_type
        self.consistent_indices = consistent_indices
        self.partitions = []

        # get data, data_size, indices of the data.
        self.data_size = len(data)
        if type(data) is not Partition:
            self.data = data
            indices = np.array([x for x in range(0, self.data_size)])
        else:
            self.data = data.data
            indices = data.indices

        # apply partition function.
        self.partition_indices(indices)

    def partition_indices(self, indices):
        if self.rank == self.world_size:
            indices = self._create_indices(indices)
        if self.consistent_indices:
            indices = self._get_consistent_indices(indices)

        # partition indices.
        from_index = 0
        for partition_size in self.partition_sizes:
            to_index = from_index + int(partition_size * self.data_size)
            self.partitions.append(indices[from_index:to_index])
            from_index = to_index

        # display the class distribution over the partitions.
        # if self.rank == self.world_size:
        if self.rank == 0:
            self.targets_of_partitions = record_class_distribution(
                self.partitions,
                self.data.targets if hasattr(self.data, "targets") else self.data.golds,
                # print_fn=self.conf.logger.log,
            )

    def _create_indices(self, indices):
        if self.partition_type == "origin":
            pass
        elif self.partition_type == "random":
            # it will randomly shuffle the indices.
            np.random.seed(0)
            np.random.shuffle(indices)
        elif self.partition_type == "sorted":
            # it will sort the indices based on the data label.
            indices = [
                i[0]
                for i in sorted(
                    [
                        (idx, target)
                        for idx, target in enumerate(self.data.targets)
                        if idx in indices
                    ],
                    key=lambda x: x[1],
                )
            ]
        elif self.partition_type == "non_iid_dirichlet":
            num_indices = len(indices)
            n_workers = len(self.partition_sizes)

            targets = (
                self.data.targets if hasattr(self.data, "targets") else self.data.golds
            )
            num_classes = len(np.unique(targets))
            indices2targets = np.array(list(enumerate(targets)))

            list_of_indices = build_non_iid_by_dirichlet(
                indices2targets=indices2targets,
                non_iid_alpha=self.args.non_iid_alpha,
                num_classes=num_classes,
                num_indices=num_indices,
                n_workers=n_workers,
            )
            indices = functools.reduce(lambda a, b: a + b, list_of_indices)
        else:
            raise NotImplementedError(
                f"The partition scheme={self.partition_type} is not implemented yet"
            )
        return indices

    def _get_consistent_indices(self, indices):
        if dist.is_initialized():
            # sync the indices over clients.
            indices = torch.IntTensor(indices)
            dist.broadcast(indices, src=self.world_size)
            return list(indices)
        else:
            return indices

    def use(self, partition_ind):
        if self.rank == self.world_size:
            return Partition(self.data, reduce(add, self.partitions))
        else:
            return Partition(self.data, self.partitions[partition_ind])


def build_non_iid_by_dirichlet(
    indices2targets, non_iid_alpha, num_classes, num_indices, n_workers
):
    n_auxi_workers = 10

    # random shuffle targets indices.
    np.random.seed(1)
    np.random.shuffle(indices2targets)

    # partition indices.
    from_index = 0
    splitted_targets = []
    num_splits = math.ceil(n_workers / n_auxi_workers)
    split_n_workers = [
        n_auxi_workers
        if idx < num_splits - 1
        else n_workers - n_auxi_workers * (num_splits - 1)
        for idx in range(num_splits)
    ]
    split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
    for idx, ratio in enumerate(split_ratios):
        to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
        splitted_targets.append(
            indices2targets[
                from_index : (num_indices if idx == num_splits - 1 else to_index)
            ]
        )
        from_index = to_index

    #
    idx_batch = []
    for _targets in splitted_targets:
        # rebuild _targets.
        _targets = np.array(_targets)
        _targets_size = len(_targets)

        # use auxi_workers for this subset targets.
        _n_workers = min(n_auxi_workers, n_workers)
        n_workers = n_workers - n_auxi_workers

        # get the corresponding idx_batch.
        min_size = 0
        while min_size < int(0.50 * _targets_size / _n_workers):
            _idx_batch = [[] for _ in range(_n_workers)]
            for _class in range(num_classes):
                # get the corresponding indices in the original 'targets' list.
                idx_class = np.where(_targets[:, 1] == _class)[0]
                idx_class = _targets[idx_class, 0]

                # sampling.
                try:
                    # np.random.seed(2)
                    proportions = np.random.dirichlet(
                        np.repeat(non_iid_alpha, _n_workers)
                    )
                    # balance
                    proportions = np.array(
                        [
                            p * (len(idx_j) < _targets_size / _n_workers)
                            for p, idx_j in zip(proportions, _idx_batch)
                        ]
                    )
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                        :-1
                    ]
                    _idx_batch = [
                        idx_j + idx.tolist()
                        for idx_j, idx in zip(
                            _idx_batch, np.split(idx_class, proportions)
                        )
                    ]
                    sizes = [len(idx_j) for idx_j in _idx_batch]
                    min_size = min([_size for _size in sizes])
                except ZeroDivisionError:
                    pass
        idx_batch += _idx_batch
    return idx_batch


def record_class_distribution(partitions, targets):
    targets_of_partitions = {}
    targets_np = np.array(targets)
    for idx, partition in enumerate(partitions):
        unique_elements, counts_elements = np.unique(
            targets_np[partition], return_counts=True
        )
        targets_of_partitions[idx] = list(zip(unique_elements, counts_elements))
    print(f"the histogram of the targets in the partitions: {targets_of_partitions.items()}")
    '''
    print_fn(
        f"the histogram of the targets in the partitions: {targets_of_partitions.items()}"
    )
    '''
    return targets_of_partitions

# prepare_data.py
def _get_cifar(name, root, split, transform, target_transform, download):
    is_train = (split == "train")

    if name == "cifar10":
        dataset_loader = datasets.CIFAR10
    elif name == "cifar100":
        dataset_loader = datasets.CIFAR100
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    # 根据你的 preprocessor_config.json：
    # - 图像需 resize 到 224x224
    # - 归一化使用 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    VIT_IMAGE_SIZE = (224, 224)
    VIT_MEAN = [0.5, 0.5, 0.5]
    VIT_STD = [0.5, 0.5, 0.5]

    if is_train:
        transform = transforms.Compose([
            transforms.Resize(VIT_IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # 将 [0, 255] 转为 [0.0, 1.0]
            transforms.Normalize(mean=VIT_MEAN, std=VIT_STD),  # 转为 [-1.0, 1.0]
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(VIT_IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=VIT_MEAN, std=VIT_STD),
        ])

    return dataset_loader(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


"""the entry for different supported dataset."""


def get_dataset(
    name,
    datasets_path,
    split="train",
    transform=None,
    target_transform=None,
    download=True,
):
    # create data folder if it does not exist.
    root = os.path.join(datasets_path, name)
    if name == "cifar10" or name == "cifar100":
        return _get_cifar(name, root, split, transform, target_transform, download)
    else:
        raise NotImplementedError
