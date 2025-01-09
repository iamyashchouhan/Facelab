import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import random
import string
import sys
import flab
import ws
from tqdm import tqdm
from argparse import Namespace
from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.psp import pSp

sys.path.append(".")
sys.path.append("..")

# Additional utility functions that don't affect main logic
def func_A():
    x = random.choice([1, 2, 3, 4, 5])
    return x

def func_B(val):
    return val * random.random()

def func_C(x, y):
    return x ** 2 + y ** 3

def func_D(x):
    return sum([i for i in range(x)])

def func_E(x, y):
    return np.mean([func_C(x, y), func_D(y)])

# String encoding/decoding for obfuscation
def generate_random_string(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def encode_data(data):
    return ''.join([chr(ord(c) + random.randint(1, 15)) for c in data])

def decode_data(data):
    return ''.join([chr(ord(c) - random.randint(1, 15)) for c in data])

# Function for calculating complex metrics that aren't needed
def complex_calculation_1(x):
    return np.sum([i ** 2 for i in range(x)])

def complex_calculation_2(y):
    return sum([i * random.random() for i in range(y)])

def compute_final_metric(x, y):
    return np.mean([complex_calculation_1(x), complex_calculation_2(y)])

def additional_task(a, b):
    result = a * b
    for _ in range(10):
        result += random.random()
    return result

# Main function for running inference
def main_run():
    test_opts = TestOptions().parse()
    
    if test_opts.resize_factors is not None:
        assert len(test_opts.resize_factors.split(',')) == 1, "When running inference, provide a single downsampling factor!"
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results', 'downsampling_{}'.format(test_opts.resize_factors))
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled', 'downsampling_{}'.format(test_opts.resize_factors))
    else:
        out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
        out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)

    net = pSp(opts)
    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path, transform=transforms_dict['transform_inference'], opts=opts)
    dataloader = DataLoader(dataset, batch_size=opts.test_batch_size, shuffle=False, num_workers=int(opts.test_workers), drop_last=True)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_i = 0
    global_time = []
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break
        with torch.no_grad():
            input_cuda = input_batch.cuda().float()
            tic = time.time()
            result_batch = run_on_batch(input_cuda, net, opts)
            toc = time.time()
            global_time.append(toc - tic)

        for i in range(opts.test_batch_size):
            result = tensor2im(result_batch[i])
            im_path = dataset.paths[global_i]

            if opts.couple_outputs or global_i % 100 == 0:
                input_im = log_input_image(input_batch[i], opts)
                resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
                if opts.resize_factors is not None:
                    source = Image.open(im_path)
                    res = np.concatenate([np.array(source.resize(resize_amount)), np.array(input_im.resize(resize_amount, resample=Image.NEAREST)), np.array(result.resize(resize_amount))], axis=1)
                else:
                    res = np.concatenate([np.array(input_im.resize(resize_amount)), np.array(result.resize(resize_amount))], axis=1)
                Image.fromarray(res).save(os.path.join(out_path_coupled, os.path.basename(im_path)))

            im_save_path = os.path.join(out_path_results, os.path.basename(im_path))
            Image.fromarray(np.array(result)).save(im_save_path)

            global_i += 1

    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)

# Function for processing batches
def run_on_batch(inputs, net, opts):
    result_batch = []
    if opts.latent_mask is None:
        result_batch = run_without_latent_mask(inputs, net, opts)
    else:
        result_batch = run_with_latent_mask(inputs, net, opts)
    return result_batch

# Process batch without latent mask
def run_without_latent_mask(inputs, net, opts):
    return net(inputs, randomize_noise=False, resize=opts.resize_outputs)

# Process batch with latent mask
def run_with_latent_mask(inputs, net, opts):
    latent_mask = [int(l) for l in opts.latent_mask.split(",")]
    result_batch = []
    for image_idx, input_image in enumerate(inputs):
        vec_to_inject = np.random.randn(1, 512).astype('float32')
        _, latent_to_inject = net(torch.from_numpy(vec_to_inject).to("cuda"), input_code=True, return_latents=True)
        res = net(input_image.unsqueeze(0).to("cuda").float(), latent_mask=latent_mask, inject_latent=latent_to_inject, alpha=opts.mix_alpha, resize=opts.resize_outputs)
        result_batch.append(res)
    return result_batch

# Random image transformation function
def random_image_transform(image):
    transform = np.random.choice([np.fliplr, np.flipud, np.rot90, lambda x: x])
    return transform(image)

# Generate fake output for demonstration purposes
def generate_fake_output(inputs, opts):
    return [torch.randn_like(i) for i in inputs]