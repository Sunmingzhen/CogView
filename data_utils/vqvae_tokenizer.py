# -*- encoding: utf-8 -*-
'''
@File    :   vqvae_tokenizer.py
@Time    :   2021/01/11 17:57:43
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
from tqdm import tqdm
from einops import rearrange

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from vqvae import new_model, img2code, code2img
from torchvision import transforms
from PIL import Image

def is_exp2(x):
    t = math.log2(x)
    return abs(t - int(t)) < 1e-4
def sqrt_int(x):
    r = int(math.sqrt(x) + 1e-4)
    assert r * r == x
    return r

# pretrained Discrete VAE from OpenAI
def make_contiguous(module):
    with torch.no_grad():
        for param in module.parameters():
            param.set_(param.contiguous())

def unmap_pixels(x, eps = 0.1):
    return torch.clamp((x - eps) / (1 - 2 * eps), 0, 1)

def load_model(path):
    with open(path, 'rb') as f:
        return torch.load(f, map_location = torch.device('cpu'))

def map_pixels(x, eps = 0.1):
    return (1 - 2 * eps) * x + eps

class OpenAIDiscreteVAE(nn.Module):
    def __init__(self, model_path, device):
        super().__init__()

        OPENAI_PATH = model_path
        self.enc = load_model(os.path.join(OPENAI_PATH, 'encoder.pkl'))
        self.dec = load_model(os.path.join(OPENAI_PATH, 'decoder.pkl'))
        make_contiguous(self)

        self.num_layers = 3
        self.image_size = 256
        self.num_tokens = 8192
        self.device = device

    @torch.no_grad()
    def get_codebook_indices(self, img):
        img = map_pixels(img)
        z_logits = self.enc.blocks(img)
        z = torch.argmax(z_logits, dim = 1)
        return rearrange(z, 'b h w -> b (h w)')

    def decode(self, img_seq):
        if isinstance(img_seq, list):
            img_seq = torch.tensor(img_seq, device=self.device)
        b, n = img_seq.shape
        img_seq = rearrange(img_seq, 'b (h w) -> b h w', h = int(math.sqrt(n)))

        z = F.one_hot(img_seq, num_classes = self.num_tokens)
        z = rearrange(z, 'b h w c -> b c h w').float().to(self.device)
        x_stats = self.dec(z).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        return x_rec

    def forward(self, img):
        return self.decode(self.get_codebook_indices(img))

class VQVAETokenizer(object):
    def __init__(self, 
            model_path, 
            device='cuda'
        ):
        ckpt = torch.load(model_path, map_location=torch.device(device))

        model = new_model()

        if list(ckpt.keys())[0].startswith('module.'):
            ckpt = {k[7:]: v for k, v in ckpt.items()}

        model.load_state_dict(ckpt)
        model = model.to(device)
        model.eval()

        self.model = model
        self.device = device
        self.image_tokens = model.quantize_t.n_embed
        self.num_tokens = model.quantize_t.n_embed

    def __len__(self):
        return self.num_tokens

    def EncodeAsIds(self, img):
        assert len(img.shape) == 4 # [b, c, h, w]
        return img2code(self.model, img)

    def DecodeIds(self, code, shape=None):
        if shape is None:
            if isinstance(code, list):
                code = torch.tensor(code, device=self.device)
            s = sqrt_int(len(code.view(-1)))
            assert s * s == len(code.view(-1))
            shape = (1, s, s)
        code = code.view(shape)
        out = code2img(self.model, code)
        return out

    def read_img(self, path, img_size=256):
        tr = transforms.Compose([
            transforms.Resize(img_size), 
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ])
        img = tr(Image.open(path))
        if img.shape[0] == 4:
            img = img[:-1]
        tr_normalize = transforms.Normalize([0.79093, 0.76271, 0.75340], [0.30379, 0.32279, 0.32800])
        img = tr_normalize(img)
        img = img.unsqueeze(0).float().to(self.device) # size [1, 3, h, w]
        return img  