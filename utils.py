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


def convert(root=Path('/home/histopath/Data/rxrx1/')):
    meta = pd.read_csv(root / 'metadata.csv')
    meta_dict = dict()
    for m in meta.itertuples():
        # print(m.site_id, m.sirna_id)
        if m.site_id[:-2] in meta_dict:
            assert m.sirna_id == meta_dict[m.site_id[:-2]]
        else:
            meta_dict[m.site_id[:-2]] = m.sirna_id

    dt = pd.read_csv(root / 'old' / 'test.csv')
    sirna = list()
    for d_idx, d in dt.iterrows():
        # print(d.id_code)
        assert d['id_code'] in meta_dict
        sirna.append(meta_dict[d['id_code']])
    dt = dt.assign(sirna=sirna)
    dt.to_csv(str(root / 'test.csv'), index=False)

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

    # csv_lists = ['train.csv', 'train_controls.csv',
    #              'test_controls.csv', 'test.csv']
    # for clist in csv_lists:
    #     print(clist)
    #     dt = pd.read_csv(root / clist)
    #     for d_idx, d in dt.iterrows():
    #         for channel in range(1, 7):
    #             if 'train' in clist:
    #                 phase = 'train'
    #             else:
    #                 phase = 'test'
    #             for site in range(1, 3):
    #                 path = root / phase / \
    #                     d['experiment'] / 'Plate{}'.format(d['plate']) / \
    #                     '{}_s{}_w{}.png'.format(d['well'], site, channel)
    #                 if not path.exists():
    #                     print(clist, path)


def cmp(root=Path('/home/histopath/Data')):
    meta1 = pd.read_csv(root / 'rxrx1' / 'metadata.csv')
    meta1_dict = dict()
    meta2 = pd.read_csv(root / 'rxrx1_v1.0' / 'metadata.csv')
    meta2_dict = dict()

    # compare two versions of metadata
    # they only differ from the 'dataset' column
    for m in meta1.itertuples():
        m_val = [m.well_id, m.cell_type, m.experiment,
                 m.plate, m.well, m.site, m.sirna_id, m.sirna]
        if m.dataset in ('train',):
            m_val.append(m.dataset)
        meta1_dict[m.site_id] = m_val
    print('meta1 done')

    for m in meta2.itertuples():
        m_val = [m.well_id, m.cell_type, m.experiment,
                 m.plate, m.well, m.site, m.sirna_id, m.sirna]
        if m.dataset in ('train',):
            m_val.append(m.dataset)
        meta2_dict[m.site_id] = m_val
    print('meta2 done')

    assert set(meta1_dict.keys()) == set(meta2_dict.keys())
    for key in meta1_dict.keys():
        # print(meta1_dict[key], meta2_dict[key])
        assert meta1_dict[key] == meta2_dict[key], '{} | {}'.format(
            meta1_dict[key], meta2_dict[key])

    # check if all the meta data are in the train.csv, test.csv ...
    meta1_dict = {'test': dict(), 'train': dict()}
    for m in meta1.itertuples():
        meta1_dict[m.dataset][m.site_id[:-2]] = m.sirna_id

    split_dict = {'test': dict(), 'train': dict()}
    csv_lists = ['train.csv',
                 'train_controls.csv',
                 'test.csv',
                 'test_controls.csv']
    for clist in csv_lists:
        print(clist)
        dt = pd.read_csv(root / 'rxrx1' / clist)
        for d in dt.itertuples():
            if 'train' in clist:
                split_dict['train'][d.id_code] = d.sirna
            else:
                split_dict['test'][d.id_code] = d.sirna

    for dataset in ['train', 'test']:
        assert set(meta1_dict[dataset].keys()) == set(
            split_dict[dataset].keys())


def main():
    # convert()
    cmp()


if __name__ == '__main__':
    main()
