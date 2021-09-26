import cv2
import logging
import math
import numpy as np
import pandas as pd
import random
from collections import defaultdict
from itertools import chain
from operator import itemgetter
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import precomputed as P


def convert(root=Path('/mnt/sda1/Data/rxrx1/')):
    # meta = pd.read_csv(root / 'metadata.csv')
    # meta_dict = dict()
    # for m in meta.itertuples():
    #     # print(m.site_id, m.sirna_id)
    #     if m.site_id[:-2] in meta_dict:
    #         assert m.sirna_id == meta_dict[m.site_id[:-2]]
    #     meta_dict[m.site_id[:-2]] = m.sirna_id

    # csv_lists = ['train.csv', 'train_controls.csv', 'test_controls.csv']
    # for clist in csv_lists:
    #     print(clist)
    #     dt = pd.read_csv(root / 'old' / clist)
    #     for d_idx, d in dt.iterrows():
    #         # print(d.id_code)
    #         if d['id_code'] not in meta_dict:
    #             print(d)
    #             continue
    #         else:
    #             dt.at[d_idx, 'sirna'] = meta_dict[d['id_code']]
    #             # d['sirna'] = meta_dict[d['id_code']]
    #     dt.to_csv(str(root/clist), index=False)
    csv_lists = ['train.csv', 'train_controls.csv', 'test_controls.csv', 'test.csv']
    for clist in csv_lists:
        print(clist)
        dt = pd.read_csv(root /  clist)
        for d_idx, d in dt.iterrows():
            for channel in range(1, 7):
                if 'train' in clist:
                    phase = 'train'
                else:
                    phase = 'test'
                for site in range(1, 3):
                    path = root / phase / \
                        d['experiment'] / 'Plate{}'.format(d['plate']) / \
                        '{}_s{}_w{}.png'.format(d['well'], site, channel)
                    if not path.exists():
                        print(clist, path)


def main():
    convert()


if __name__ == '__main__':
    main()
