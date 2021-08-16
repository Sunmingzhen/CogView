# -*- encoding: utf-8 -*-
'''
@File    :   templates.py
@Time    :   2021/01/11 22:28:57
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import ipdb
import math
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from .unified_tokenizer import get_tokenizer
from .vqvae_tokenizer import sqrt_int

def concat_codes(*codes):
    is_numpy = is_tensor = False
    for code in codes:
        if isinstance(code, np.ndarray):
            is_numpy = True
        if isinstance(code, torch.Tensor):
            is_tensor = True
            device = code.device
    if is_tensor:
        return torch.cat(
            [
                code.to(device) if isinstance(code, torch.Tensor) else torch.tensor(code, device=device)
                for code in codes
            ]
        )
    elif is_numpy:
        return np.concatenate(
            [
                np.array(code)
                for code in codes
            ],
            axis=0
        )
    else:
        ret = []
        for code in codes:
            ret = ret + code
        return ret

def TextCodeTemplate(text, code):
    tokenizer = get_tokenizer()
    text_ids = [tokenizer['[ROI1]']] + tokenizer(text)
    code = tokenizer.wrap_code(code)
    return concat_codes(text_ids, code)

def TextFramesTemplate(text, code, frame_num):
    # ipdb.set_trace()
    tokenizer = get_tokenizer()
    text_ids = [tokenizer['[ROI1]']] + tokenizer(text)
    assert len(code) % frame_num == 0
    per_frame_code = len(code) // frame_num

    tokens = text_ids
    for i in range(frame_num):
        img_code = code[i * per_frame_code : (i+1) * per_frame_code]
        img_code = tokenizer.wrap_code(img_code, idx=i+1)
        tokens = concat_codes(tokens, img_code)
    # ipdb.set_trace()
    return tokens


def Code2CodeTemplate(text, code0, code1):
    tokenizer = get_tokenizer()
    text_ids = tokenizer.parse_query(text) if isinstance(text, str) else text
    code0 = tokenizer.wrap_code(code0)
    code1 = tokenizer.wrap_code(code1, idx=2)
    return concat_codes(text_ids, code0, code1)

def PureTextTemplate(text):
    tokenizer = get_tokenizer()
    return tokenizer(text) + [tokenizer['[SEP]']]







