import os
import PIL
import ipdb
import math
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.multiprocessing as mp

from einops import rearrange
from tqdm import tqdm
import matplotlib as plt
from dall_e import map_pixels, unmap_pixels, load_model


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def show_img(vector):
    low = vector < 1e-8
    high = vector > 1 - 1e-8
    vector = vector * (~low & ~high)
    num_images = min(vector.shape[0], 10)
    show_img = vector[:num_images, :, :, :].permute(2, 0, 3, 1).numpy()
    show_img = show_img.reshape(args.image_size, num_images * args.image_size, 3)
    print(show_img.shape)

    plt.figure(figsize=(20, 10))  # 设定图像的尺寸(x, y)
    plt.imshow(show_img)


# pretrained Discrete VAE from OpenAI
def make_contiguous(module):
    with torch.no_grad():
        for param in module.parameters():
            param.set_(param.contiguous())


def unmap_pixels(x, eps=0.1):
    return torch.clamp((x - eps) / (1 - 2 * eps), 0, 1)


def load_model(path):
    with open(path, 'rb') as f:
        return torch.load(f, map_location=torch.device('cpu'))


def map_pixels(x, eps=0.1):
    return (1 - 2 * eps) * x + eps


class OpenAIDiscreteVAE(nn.Module):
    def __init__(self, model_path):
        super().__init__()

        OPENAI_PATH = os.path.expanduser(os.path.join(model_path, 'OPENAI'))
        self.enc = load_model(os.path.join(OPENAI_PATH, 'encoder.pkl'))
        self.dec = load_model(os.path.join(OPENAI_PATH, 'decoder.pkl'))
        make_contiguous(self)

        self.num_layers = 3
        self.image_size = 256
        self.num_tokens = 8192

    @torch.no_grad()
    def get_codebook_indices(self, img):
        img = map_pixels(img)
        z_logits = self.enc.blocks(img)
        z = torch.argmax(z_logits, dim=1)
        #         print(z.shape)
        return rearrange(z, 'b h w -> b (h w)')

    def decode(self, img_seq):
        b, n = img_seq.shape
        img_seq = rearrange(img_seq, 'b (h w) -> b h w', h=int(math.sqrt(n)))

        z = F.one_hot(img_seq, num_classes=self.num_tokens)
        z = rearrange(z, 'b h w c -> b c h w').float()
        x_stats = self.dec(z).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        return x_rec

    def forward(self, img):
        return self.decode(self.get_codebook_indices(img))


class WebvidFramesDataset(Dataset):
    def __init__(self, args, split='train'):
        super().__init__()

        self.image_size = args.image_size
        self.frame_path = args.data_dir

        self.transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size),
                              interpolation=PIL.Image.BILINEAR),
            transforms.ToTensor()
        ])

        if split == 'train':
            self.frame_path = os.path.join(self.frame_path, 'train')
            self.video_ids = os.listdir(self.frame_path)
        else:
            raise ValueError(f"No Implemention for {split}")

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]

        imgs = []
        cur_path = os.path.join(self.frame_path, video_id)
        cur_imgs = os.listdir(cur_path)
        for i in range(10):
            img_path = os.path.join(cur_path, f"{video_id}_{i}.jpg")
            try:
                img = torchvision.datasets.folder.pil_loader(img_path)
                cur_img = self.transform(img).unsqueeze(0)
            except:
                raise RuntimeError('Fail to load ' + img_path)

            #             ipdb.set_trace()
            imgs.append(cur_img)

        return torch.cat(imgs, dim=0), video_id

def generate(gpu, args):
    # multi process
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=args.world_size,
                            rank=rank)

    # save_path for npy files
    save_path = os.path.join(args.save_dir, 'train')
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # set random seed
    setup_seed(10)
    torch.cuda.set_device(gpu)

    # set model
    vae = OpenAIDiscreteVAE('/home/user/mzsun/codes/Video_VQVAE/pretrained/')
    model = vae.cuda(gpu)
    # warp the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # dataloader
    trainset = WebvidFramesDataset(args)
    trainsampler = torch.utils.data.distributed.DistributedSampler(
        trainset, num_replicas=args.world_size, rank=rank
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True, sampler=trainsampler)

    td = tqdm(range(len(iter(trainloader))), desc='rank: '+str(rank))
    for _ in td:
        try:
            image, ids = next(iter(trainloader))
        except:
            continue
        #     print(image.shape)
        inputs = image.reshape(args.batch_size * 10, 3, args.image_size, args.image_size)
        indices = model.module.get_codebook_indices(inputs.cuda(gpu))
        outputs = rearrange(indices, '(b n) l -> b (n l)', b=args.batch_size, n=10).detach().cpu().numpy()

        for i, video_id in enumerate(ids):
            npy_file = os.path.join(save_path, '{}.npy'.format(video_id))
            if os.path.exists(npy_file) is True:
                continue
            np.save(npy_file, outputs[i])
            # if gpu == 0:
            #     print(video_id, 'is finished.')

def main():
    def get_arg_parser():
        parser = argparse.ArgumentParser(description="DiffVQVAE")
        parser.add_argument('-n', '--nodes', default=1,
                            type=int, metavar='N')
        parser.add_argument('-g', '--gpus', default=1, type=int,
                            help='number of gpus per node')
        parser.add_argument('-nr', '--nr', default=0, type=int,
                            help='ranking within the nodes')

        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--image_size', default=64, type=int)
        parser.add_argument('--dataset', type=str, default='webvid')
        parser.add_argument('--data_dir', type=str, default='/raid/datasets/video_datasets/webvid/webvid_frames_10/')
        parser.add_argument('--save_dir', type=str,
                            default='/raid/datasets/video_datasets/webvid/webvid_tokens_im64_ds8')
        return parser.parse_args()
    args = get_arg_parser()
    # multi process
    args.world_size = args.nodes * args.gpus
    os.environ['MASTER_ADDR'] = '127.0.0.2'
    os.environ['MASTER_PORT'] = '29678'
    mp.spawn(generate, nprocs=args.gpus, args=(args,))

    # generate(0, args)

if __name__ == '__main__':
    main()


    #     break