# -*- encoding: utf-8 -*-
'''
@File    :   unified_tokenizer.py
@Time    :   2021/01/11 16:36:33
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

from .Simple_tokenizer import SimpleTokenizer
from .sp_tokenizer import from_pretrained
from .vqvae_tokenizer import OpenAIDiscreteVAE, sqrt_int
sys.path.append('..')
from utils import get_logger

class UnifiedTokenizer(object):
    def __init__(self, img_tokenizer_path, device, isChinese=False, img_tokenizer_num_tokens=None):
        self.device = device
        self.Logger = get_logger()

        if img_tokenizer_path is None and img_tokenizer_num_tokens is not None:
            # pretraining but only know the vocab size of VQVAE, which is developing fast
            self.img_tokenizer = FakeTokenizer(img_tokenizer_num_tokens)
        else:
            self.img_tokenizer = OpenAIDiscreteVAE(model_path=img_tokenizer_path, device=self.device)
        if isChinese is False:
            self.Logger.info("Loading SimpleTokenizer for English text.")
            self.txt_tokenizer = SimpleTokenizer()  # English Tokenizer
        else:
            self.Logger.info("Loading PretrainedTokenizer for chinese text.")
            self.txt_tokenizer = from_pretrained() #chinese Tokenizer
        self.num_tokens = self.img_tokenizer.num_tokens + self.txt_tokenizer.num_tokens
        self.raw_command_tokens = [
                ('[PAD]', 0),
                ('[BOI1]', 1),  # Begin
                ('[BOI2]', 2),
                ('[BOI3]', 3),
                ('[BOI4]', 4),
                ('[BOI5]', 5),
                ('[BOI6]', 6),
                ('[BOI7]', 7),
                ('[BOI8]', 8),
                ('[BOI9]', 9),
                ('[BOI10]', 10),
                ('[EOI1]', 11),  # End
                ('[EOI2]', 12),
                ('[EOI3]', 13),
                ('[EOI4]', 14),
                ('[EOI5]', 15),
                ('[EOI6]', 16),
                ('[EOI7]', 17),
                ('[EOI8]', 18),
                ('[EOI9]', 19),
                ('[EOI10]', 20),
                ('[ROI1]', 21),  # Reference
                ('[ROI2]', 22),
                ('[ROI3]', 23),
                ('[SEP]', 24),
                ('[MASK]', 25),
                ('[CLS]', 26),
                ('[ENC]', 27),
                ('[TINY]', 28),  # 8 * 8
                ('[SMALL]', 29),  # 16 * 16
                ('[BASE]', 30),  # 32 * 32
                ('[BIG]', 31),  # 64 * 64
                ('[POS0]', 32),  # 58210
                ('[POS1]', 33),
                ('[POS2]', 34),
                ('[POS3]', 35),
                ('[POS4]', 36),
                ('[POS5]', 37),
                ('[POS6]', 38),
                ('[POS7]', 39),
                ('[POS8]', 40),
                ('[BOI4]', 41),
                ('[EOI4]', 42)]
        # self.raw_command_tokens = [
        #     ('[PAD]', 0),
        #     ('[BOI1]', 1), # Begin
        #     ('[BOI2]', 2),
        #     ('[BOI3]', 3),
        #     ('[EOI1]', 4), # End
        #     ('[EOI2]', 5),
        #     ('[EOI3]', 6),
        #     ('[ROI1]', 7), # Reference
        #     ('[ROI2]', 8),
        #     ('[ROI3]', 9),
        #     ('[SEP]', 10),
        #     ('[MASK]', 11),
        #     ('[CLS]', 12),
        #     ('[ENC]', 13),
        #     ('[TINY]', 14), # 8 * 8
        #     ('[SMALL]', 15), # 16 * 16
        #     ('[BASE]', 16), # 32 * 32
        #     ('[BIG]', 17), # 64 * 64
        #     ('[POS0]', 18), # 58210
        #     ('[POS1]', 19),
        #     ('[POS2]', 20),
        #     ('[POS3]', 21),
        #     ('[POS4]', 22),
        #     ('[POS5]', 23),
        #     ('[POS6]', 24),
        #     ('[POS7]', 25),
        #     ('[POS8]', 26),
        #     ('[BOI4]', 27),
        #     ('[EOI4]', 28)
        #     # Please leave the ``size tokens'' at the back of command tokens
        # ]
        self.command_tokens = {
            k: v + self.num_tokens
            for k, v in self.raw_command_tokens
        }
        self.num_tokens += len(self.raw_command_tokens)

        self.img_class_mask = torch.zeros(self.num_tokens)
        self.img_class_mask[:self.img_tokenizer.num_tokens] += 1
        self.txt_class_mask = torch.zeros(self.num_tokens)
        self.txt_class_mask[self.img_tokenizer.num_tokens : self.img_tokenizer.num_tokens + self.txt_tokenizer.num_tokens] += 1
    
    def __getitem__(self, command_token):
        return self.command_tokens[command_token]

    def __len__(self):
        """total number of tokens"""
        return self.num_tokens

    def __call__(self, inputs, process_fn=None):
        """run preprocessing and encode inputs as Ids
            CANNOT contain command tokens"""
        if isinstance(inputs, torch.Tensor): # image
            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(0)
            return self.img_tokenizer.EncodeAsIds(inputs)
        return self.EncodeAsIds(inputs, process_fn=process_fn)
    
    def EncodeAsIds(self, text, process_fn=None):
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)
        ids = self.txt_tokenizer.encode(processed_text)
        return [x + self.img_tokenizer.num_tokens for x in ids]
    
    def DecodeIds(self, ids):
        ret, img_buffer, txt_buffer, ret_imgs = [], [], [], []
        try:
            for x in ids:
                if self.num_tokens - len(self.raw_command_tokens) <= x:
                    # command tokens
                    token = self.raw_command_tokens[x - (self.num_tokens - len(self.raw_command_tokens))][0]
                    if token.startswith('[EOI') and len(img_buffer) > 0:
                        ipdb.set_trace()
                        # test
                        # img_buffer = np.load('/raid/datasets/video_datasets/msrvtt/msrvtt_tokens/trainval/video0.npy')[:256]
                        img_code = torch.tensor(img_buffer).unsqueeze(0)
                        ret_imgs.append(self.img_tokenizer.decode(img_code))
                        img_buffer = []
                    if len(txt_buffer) > 0:
                        # dump text
                        ret.append(self.txt_tokenizer.decode(txt_buffer))
                        txt_buffer = []
                    ret.append(token)
                elif x < self.img_tokenizer.num_tokens:
                    img_buffer.append(x)
                else:
                    txt_buffer.append(x - self.img_tokenizer.num_tokens)
            
            if len(img_buffer) > 0:
                # dump image
                ret_imgs.append(self.img_tokenizer.decode(img_buffer))
                img_buffer = []
            if len(txt_buffer) > 0:
                # dump text
                ret.append(self.txt_tokenizer.decode(txt_buffer))
                txt_buffer = []
        except ValueError:
            print('Value error in tokenization, skipping...')
        return ret, ret_imgs

    def wrap_code(self, code, idx=1):
        s = sqrt_int(len(code))
        prefix = {8:'[TINY]', 16:'[SMALL]', 32:'[BASE]', 64:'[BIG]'}[s]
        boi = {1:'[BOI1]', 2: '[BOI2]', 3:'[BOI3]', 4:'[BOI4]', 5:'[BOI5]', 6:'[BOI6]', 7: '[BOI7]', 8:'[BOI8]', 9:'[BOI9]', 10:'[BOI10]'}[idx]
        eoi = {1:'[EOI1]', 2: '[EOI2]', 3:'[EOI3]', 4:'[EOI4]', 5:'[EOI5]', 6:'[EOI6]', 7: '[EOI7]', 8:'[EOI8]', 9:'[EOI9]', 10:'[EOI10]'}[idx]
    
        if isinstance(code, list):
            return [self.command_tokens[prefix], self.command_tokens[boi]] + \
                code + [self.command_tokens[eoi]]
        elif isinstance(code, np.ndarray):
            return np.concatenate(
                (
                    np.array([self.command_tokens[prefix], self.command_tokens[boi]]),
                    code,
                    np.array([self.command_tokens[eoi]])
                ),
                axis=0
            )
        elif isinstance(code, torch.Tensor):
            return torch.cat(
                (
                    torch.tensor([self.command_tokens[prefix], self.command_tokens[boi]]),
                    code, 
                    torch.tensor([self.command_tokens[eoi]])
                )
            )
        else:
            raise ValueError('')

    def parse_query(self, query, img_size=256):
        text_buffer = []
        ret = []
        for part in query.split(' '):
            if part in self.command_tokens:
                if len(text_buffer) > 0:
                    # dump text ids
                    ret.extend(self.EncodeAsIds(' '.join(text_buffer)))
                    text_buffer = []
                if part == '[MASK]':
                    ret.append(-1)
                else:
                    ret.append(self.command_tokens[part])
            elif part.startswith('[MASK]*'): # special lang *N
                c = int(part[7:])
                assert c > 0
                if len(text_buffer) > 0:
                    # dump text ids
                    ret.extend(self.EncodeAsIds(' '.join(text_buffer)))
                    text_buffer = []
                ret.extend([-1] * c)
            elif part.startswith('[Image'): # [Image*N]path
                c = part[6:]
                assert len(c) > 0
                num_codes, img_path = c.split(']')
                if num_codes == '':
                    num_codes = 1024
                else:
                    num_codes = int(num_codes)
                
                raw_img = self.img_tokenizer.read_img(img_path, img_size=img_size)
                img_codes = self.img_tokenizer.EncodeAsIds(raw_img) # [1, 32*32]
                img_codes[0, num_codes:] = -1
                img_codes = img_codes[0].tolist()
                ret.extend(img_codes)
            else:
                text_buffer.append(part)

        if len(text_buffer) > 0:
            # dump text ids
            ret.extend(self.EncodeAsIds(' '.join(text_buffer)))
            text_buffer = []
        return ret

def get_tokenizer(args=None):
    if not hasattr(get_tokenizer, 'tokenizer'):
        # the first time to load the tokenizer, specify img_tokenizer_path
        get_tokenizer.tokenizer = UnifiedTokenizer(
            args.img_tokenizer_path, 
            device=torch.cuda.current_device(),
            img_tokenizer_num_tokens=args.img_tokenizer_num_tokens
            )
    return get_tokenizer.tokenizer

class FakeTokenizer(object):
    def __init__(self, num_tokens):
        self.num_tokens = num_tokens
    def __len__(self):
        return self.num_tokens