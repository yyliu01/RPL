import torch
import numpy
from typing import Iterator, List, TypeVar
from torch.utils import data
T_co = TypeVar('T_co', covariant=True)


class MixContextLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, shuffle, num_workers, batch_size, city_img_num, coco_img_num, gpu_num,
                 pin_memory=True):

        self.dataset = dataset
        self.batch_size = batch_size

        if gpu_num <= 1:
            sampler = torch.utils.data.SequentialSampler if shuffle else torch.utils.data.RandomSampler
            batch_sampler = MixContextBatchSampler(sampler(dataset), batch_size, city_img_num, coco_img_num,
                                                   drop_last=True)
        else:
            coco_img_num = int(coco_img_num/gpu_num)
            city_img_num = int(city_img_num/gpu_num)
            batch_sampler = MixContextBatchSampler(torch.utils.data.DistributedSampler(self.dataset, shuffle=shuffle,
                                                                                       drop_last=True),
                                                   batch_size, city_img_num, coco_img_num,
                                                   drop_last=True, gpu_num=gpu_num)

        super(MixContextLoader, self).__init__(dataset=self.dataset, num_workers=num_workers,
                                               batch_sampler=batch_sampler, pin_memory=pin_memory)


class MixContextBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, sampler, batch_size, city_img_num, coco_img_num, drop_last=True, gpu_num=1):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.gpu_num_ = gpu_num
        self.ood_appear_batch = self.calculates_ood_batch_idx(city_img_num, coco_img_num)

        super(MixContextBatchSampler, self).__init__(sampler=self.sampler, batch_size=self.batch_size,
                                                     drop_last=self.drop_last)

    def calculates_ood_batch_idx(self, city, ood):
        city_appear_batch = numpy.asarray(range(int(city//self.batch_size)))
        ood_appear_batch = int(ood//self.batch_size)
        numpy.random.shuffle(city_appear_batch)
        return city_appear_batch[:ood_appear_batch]

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        for idx in self.sampler:
            if self.gpu_num_ <= 1:
                batch.append([idx, (idx//self.batch_size in self.ood_appear_batch).__bool__()])
            else:
                batch.append([idx, True])

            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

