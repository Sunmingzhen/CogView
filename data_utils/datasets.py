# -*- encoding: utf-8 -*-
'''
@File    :   datasets.py
@Time    :   2021/01/11 21:01:51
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import ipdb
import sys
import math
import random
from tqdm import tqdm
import logging


import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision import transforms as T
import pickle
from collections import namedtuple

import PIL
import lmdb
import json
from einops import rearrange

import sys
sys.path.append('..')
from utils import get_logger
from .unified_tokenizer import get_tokenizer
from .templates import TextCodeTemplate, TextFramesTemplate

logger = logging.getLogger(__name__)


class LMDBDataset(Dataset):
    def __init__(self, path, process_fn):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.process_fn = process_fn
        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        with self.env.begin(write=False) as txn:
            key = str(idx).encode('utf-8')

            row = pickle.loads(txn.get(key))

            return self.process_fn(row)

def get_dataset_by_type(dataset_type, path: str, args):
    Logger = get_logger()
    if dataset_type == "MsrvttTokensDataset":
        Logger.info(f"Loading MsrvttTokensDataset for train...")
        train_ds = MsrvttTokensDataset(path, 'trainval', args)
        valid_ds = MsrvttTokensDataset(path, 'test', args)
        Logger.info(f"t=Train data {len(train_ds)}.")
        return train_ds, valid_ds, None
    elif dataset_type == "WebvidTokensDataset":
        Logger.info(f"Loading WebvidTokensDataset for train...")
        train_ds = WebvidTokensDataset(path, 'train', args)
        Logger.info(f"t=Train data {len(train_ds)}.")
        return train_ds, None, None
    elif dataset_type == "WebvidFramesDataset":
        Logger.info(f"Loading WebvidFramesDataset for train...")
        train_ds = WebvidFramesDataset(path, 'train', args)
        Logger.info(f"t=Train data {len(train_ds)}.")
        return train_ds, None, None
    else:
        assert ValueError(f"No implemention for {dataset_type}")

def get_dataset_by_type_origin(dataset_type, path: str, args):
    tokenizer = get_tokenizer()
    if args.finetune and args.max_position_embeddings_finetune > args.max_position_embeddings:
        ml = args.max_position_embeddings_finetune
    else:
        ml = args.max_position_embeddings

    def pad_to_len(ret):
        if len(ret) < ml:  # pad
            return np.concatenate((ret,
                                   np.array([tokenizer['[PAD]']] * (ml - len(ret)))),
                                  axis=0), len(ret)
        else:
            if len(ret) > ml:
                logger.warning('Out of max len, truncated.')
            return ret[:ml], ml

    def process_fn(row):
        ret, attention_mask_sep = pad_to_len(row.flatten())
        return {'text': ret,
                'loss_mask':  np.array([1] * attention_mask_sep + [0] * (len(ret) - attention_mask_sep))}

    Logger = get_logger()
    if dataset_type == "MsrvttDataset":
        Logger.info(f"Loading MsrvttTokensDataset for train...")
        return MsrvttTokensDataset(path, 'train', args)
    elif dataset_type == "WebvidDataset":
        Logger.info(f"Loading WebvidTokensDataset for train...")
        return WebvidTokensDataset(path, 'train', args)
    else:
        assert ValueError(f"No implemention for {dataset_type}")

# def get_dataset_by_type(dataset_type, path: str, args, DS_CLASS=LMDBDataset):
#
#     tokenizer = get_tokenizer()
#     if args.finetune and args.max_position_embeddings_finetune > args.max_position_embeddings:
#         ml = args.max_position_embeddings_finetune
#     else:
#         ml = args.max_position_embeddings
#
#     def pad_to_len(ret):
#
#         if len(ret) < ml: # pad
#             return np.concatenate((ret,
#                 np.array([tokenizer['[PAD]']] * (ml - len(ret)))),
#                 axis=0), len(ret)
#         else:
#             if len(ret) > ml:
#                 logger.warning('Out of max len, truncated.')
#             return ret[:ml], ml
#     Logger = get_logger()
#     if dataset_type == "MsrvttDataset":
#         Logger.info("Loading MsrvttTokensDataset...")
#         return MsrvttTokensDataset(path, args)
#     elif dataset_type == 'TokenizedDataset':
#         Logger.info("Loading TokenizedDataset...")
#         # already tokenized when saved
#         def process_fn(row):
#             ret, attention_mask_sep = pad_to_len(row.flatten())
#             return {'text': ret,
#                 'loss_mask':  np.array([1] * attention_mask_sep + [0] * (len(ret) - attention_mask_sep))
#                 }
#
#     elif dataset_type == 'TextCodeDataset':
#         Logger.info("Loading TextCodeDataset...")
#         def process_fn(row):
#             text, code = row[0], row[1].flatten()
#             ret = TextCodeTemplate(text, code)
#             ret, attention_mask_sep = pad_to_len(ret)
#             return {'text': ret,
#                 'loss_mask':  np.array([1] * attention_mask_sep + [0] * (len(ret) - attention_mask_sep))
#                 }
#     return DS_CLASS(path, process_fn)

class MsrvttTokensDataset(Dataset):
    def __init__(self, dataset_path, split, args,  image_size=128, text_len=35, truncate_captions=True,
                 shuffle=True):
        super(MsrvttTokensDataset).__init__()
        self.args = args
        self.shuffle = shuffle

        self.truncate_captions = truncate_captions

        self.text_len = text_len
        self.image_size = image_size

        # get video_ids
        self.frame_path = os.path.join(dataset_path, split)
        self.video_ids = os.listdir(self.frame_path)
        assert self.video_ids[0].endswith('.npy')
        self.is_tokens = True
        self.video_ids = [id[:-4] for id in self.video_ids]

        # get video_captions
        self.caption_path = os.path.join(dataset_path, 'caption.pkl')
        self.caption_dict = pickle.load(open(self.caption_path, 'rb'))

        self.logger = get_logger()
        self.tokenizer = get_tokenizer()
        if args.finetune and args.max_position_embeddings_finetune > args.max_position_embeddings:
            self.ml = args.max_position_embeddings_finetune
        else:
            self.ml = args.max_position_embeddings

    def process_fn(self, row):
        ret = row.flatten()
        if len(ret) <= self.ml:  # pad
            ret_tokens = np.concatenate((ret,
                                   np.array([self.tokenizer['[PAD]']] * (self.ml - len(ret)))),
                                  axis=0)
            attention_mask_sep= len(ret)
        else:
            logger.warning(f'length of {len(ret)} out of max len, truncated.')
            ret_tokens = ret[:self.ml]
            attention_mask_sep= self.ml

        return {'text': ret_tokens,
                'loss_mask': np.array([1] * attention_mask_sep + [0] * (self.ml - attention_mask_sep))}

    def __len__(self):
        return len(self.video_ids)

    def random_sample(self):
        return self.__getitem__(np.random.randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, index):
        video_id = self.video_ids[index]

        # get video tokens of 10 frames
        frame_tokens = torch.from_numpy(np.load(os.path.join(self.frame_path, '{}.npy'.format(video_id))))
        if frame_tokens.shape[0] != 2560:
            print("+" * 20 + "  " + video_id + "  " + "+" * 20)
        assert frame_tokens.shape[0] == 2560, "frame_tokens.shape[0] != 2560"

        # currently only predict 4 frames
        frame_num = 4
        frame_tokens = frame_tokens[:256 * frame_num]

        # get video captions by random choice
        captions = self.caption_dict[video_id]
        captions = list(filter(lambda t: len(t) > 0, captions))

        caption = np.random.choice(captions)

        # wrap txt&img tokens
        tokens = TextFramesTemplate(caption, frame_tokens, frame_num)
        return self.process_fn(tokens)

class WebvidTokensDataset(Dataset):
    def __init__(self, dataset_path, split, args,  image_size=64, text_len=35, truncate_captions=True,
                 shuffle=True):
        super(WebvidTokensDataset).__init__()
        self.shuffle = shuffle

        self.truncate_captions = truncate_captions

        self.text_len = text_len
        self.image_size = image_size

        self.frame_path = os.path.join(dataset_path, 'webvid_tokens')
        self.json_path = dataset_path
        if split == 'train':
            self.frame_path = os.path.join(self.frame_path, 'train')
            self.json_path = os.path.join(self.json_path, 'webvid_2M_train.json')
            self.video_ids = os.listdir(self.frame_path)
        else:
            raise ValueError(f"No Implemention for {split}")

        # get video_ids
        assert self.video_ids[0].endswith('.npy')
        self.is_tokens = True
        self.video_ids = [id[:-4] for id in self.video_ids]

        # get video_captions
        self.json_dict = json.load(open(self.json_path, 'r'))

        self.logger = get_logger()
        self.tokenizer = get_tokenizer()
        if args.finetune and args.max_position_embeddings_finetune > args.max_position_embeddings:
            self.ml = args.max_position_embeddings_finetune
        else:
            self.ml = args.max_position_embeddings

    def process_fn(self, row):
        ret = row.flatten()
        if len(ret) <= self.ml:  # pad
            ret_tokens = np.concatenate((ret,
                                   np.array([self.tokenizer['[PAD]']] * (self.ml - len(ret)))),
                                  axis=0)
            attention_mask_sep= len(ret)
        else:
            # logger.warning(f'length of {len(ret)} out of max len, truncated.')
            ret_tokens = ret[:self.ml]
            attention_mask_sep= self.ml

        return {'text': torch.from_numpy(np.int64(ret_tokens)),
                'loss_mask': np.array([1] * attention_mask_sep + [0] * (self.ml - attention_mask_sep))}

    def __len__(self):
        return len(self.video_ids)

    def random_sample(self):
        return self.__getitem__(np.random.randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        self.logger.info(f'skip sample idx: {ind}!!')
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, index):
        video_id = self.video_ids[index]

        # get video tokens of 10 frames
        frame_tokens = np.load(os.path.join(self.frame_path, '{}.npy'.format(video_id)))
        if frame_tokens.shape[0] != 2560:
            print("+" * 20 + "  " + video_id + "  " + "+" * 20)
        assert frame_tokens.shape[0] == 2560, "frame_tokens.shape[0] != 2560"

        # currently only predict 4 frames
        frame_num = 4
        frame_tokens = frame_tokens[:256 * frame_num]

        # get video captions by random choice
        caption = self.json_dict[video_id]['caption']

        # wrap txt&img tokens
        tokens = TextFramesTemplate(caption, frame_tokens, frame_num)
        return self.process_fn(tokens)


class WebvidFramesDataset(Dataset):
    def __init__(self, dataset_path, split, args,  image_size=128, text_len=35, truncate_captions=True,
                 shuffle=True):
        super().__init__()
        self.shuffle = shuffle

        self.text_len = text_len
        self.image_size = image_size
        self.json_path = dataset_path
        self.truncate_captions = truncate_captions
        self.frame_path = os.path.join(dataset_path, 'webvid_frames_10')

        if split == 'train':
            self.frame_path = os.path.join(self.frame_path, 'train')
            self.json_path = os.path.join(self.json_path, 'webvid_2M_train.json')
            # get video_ids
            video_ids = os.listdir(self.frame_path)
            self.video_ids = video_ids * 20
            assert self.video_ids[0].endswith('.npy') is False
        else:
            raise ValueError(f"No Implemention for {split}")

        # get video_captions
        self.json_dict = json.load(open(self.json_path, 'r'))

        self.logger = get_logger()
        self.tokenizer = get_tokenizer()
        if args.finetune and args.max_position_embeddings_finetune > args.max_position_embeddings:
            self.ml = args.max_position_embeddings_finetune
        else:
            self.ml = args.max_position_embeddings

        self.transform = transforms.Compose([
                                transforms.Resize((args.image_size, args.image_size),
                                                  interpolation=PIL.Image.BILINEAR),
                                transforms.ToTensor()
                            ])


    def process_fn(self, row):
        ret = row.flatten()
        if len(ret) <= self.ml:  # pad
            ret_tokens = np.concatenate((ret,
                                   np.array([self.tokenizer['[PAD]']] * (self.ml - len(ret)))),
                                  axis=0)
            attention_mask_sep= len(ret)
        else:
            # logger.warning(f'length of {len(ret)} out of max len, truncated.')
            ret_tokens = ret[:self.ml]
            attention_mask_sep= self.ml

        return {'text': torch.from_numpy(np.int64(ret_tokens)),
                'loss_mask': np.array([1] * attention_mask_sep + [0] * (self.ml - attention_mask_sep))}

    def __len__(self):
        return len(self.video_ids)

    def random_sample(self):
        return self.__getitem__(np.random.randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        self.logger.info(f'skip sample idx: {ind}!!')
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, index):
        video_id = self.video_ids[index]

        # get video tokens of 10 frames
        imgs = []
        cur_path = os.path.join(self.frame_path, video_id)
        for i in range(10):
            img_path = os.path.join(cur_path, f"{video_id}_{i}.jpg")
            img = torchvision.datasets.folder.pil_loader(img_path)
            cur_img = self.transform(img).unsqueeze(0)
            imgs.append(cur_img)

        # predict 10 frames
        frame_num = 10
        frame_tokens = self.tokenizer.img_tokenizer.get_codebook_indices(torch.cat(imgs, dim=0)).flatten() # tokens: 10 * (8*8) -> frames, tokens_flatten
        # self.logger.info(f"frame_tokens' len: {frame_tokens.shape}")
        # get video captions by random choice
        caption = self.json_dict[video_id]['caption']

        # wrap txt&img tokens
        tokens = TextFramesTemplate(caption, frame_tokens, frame_num)
        return self.process_fn(tokens)
