"""Calculates the Frechet Inception Distance (FID) to evalulate GANs
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.
See --help to see further details.
Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
https://github.com/mseitzer/pytorch-fid
"""

import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import torchvision.utils as vutils

from PIL import Image
import logging
# from dataset import ImageDatasetFromFile
from torch.utils.data import DataLoader
# from networks import IntroVAE
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from metrics.inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str, nargs=2,
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')


def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg", ".bmp"])


# def load_model(model, pretrained, device):
#     weights = torch.load(pretrained, map_location=device)
#     model.load_state_dict(weights['model'].state_dict())

def load_model(model, pretrained, device):
    weights = torch.load(pretrained, map_location=device)
    pretrained_dict = weights['model'].state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_activations(files, model, batch_size=50, dims=2048,
                    cuda=False, verbose=False, device=torch.device("cpu")):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    pred_arr = np.empty((len(files), dims))
    n_batches = len(files) // batch_size
    for i in tqdm(range(0, len(files), batch_size)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i
        end = i + batch_size

        images = np.array([imread(str(f)).astype(np.float32)
                           for f in files[start:end]])

        # Reshape to (n_images, 3, height, width)
        images = images.transpose((0, 3, 1, 2))
        images /= 255

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.to(device)

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    if verbose:
        print(' done')

    return pred_arr


def get_activations_given_dataset(dataloader, model, batch_size=50, dims=2048,
                                  cuda=False, verbose=False, device=torch.device("cpu"), num_images=50000):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    # model.eval()
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    activations = []
    num_images_processed = 0
    for idx, batch in enumerate(dataloader):
        if len(batch) == 2 or len(batch) == 3:
            batch = batch[0]
        if cuda:
            batch = batch.to(device)
        res = model(batch)[0]
        # if idx == 0:
        #     print(batch[0].min())
        #     print(batch[0].max())
        #     print("real images shape: ", batch.shape)
        #     print("res output shape:" , res.shape)
        # res = inception.run(x, num_gpus=gpu_count, assume_frozen=True)
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if res.size(2) != 1 or res.size(3) != 1:
            res = adaptive_avg_pool2d(res, output_size=(1, 1))
        activations.append(res.cpu().data.numpy().reshape(res.size(0), -1))
        num_images_processed += batch.shape[0]
        if num_images_processed > num_images:
            # print("num img proc.: ", num_images_processed, " num img req.:, ", num_images)
            break
    activations = np.concatenate(activations)
    activations = activations[:num_images]
    print("total real activations: ", activations.shape)
    # print("num images processed: ", num_images_processed)

    if verbose:
        print(' done')

    return activations


def get_activations_generate(model_s, model, batch_size=50, dims=2048,
                             cuda=False, verbose=False, device=torch.device("cpu"), num_images=50000, rv=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    # model.eval()
    # logger = logging.getLogger("logger")
    # logger.setLevel(logging.DEBUG)
    # lod = cfg.DATASET.MAX_RESOLUTION_LEVEL - 2
    # dataset = TFRecordsDataset(cfg, logger, rank=0, world_size=1, buffer_size_mb=1024,
    #                            channels=cfg.MODEL.CHANNELS, train=True)
    # dataset.reset(lod + 2, batch_size)
    # batches = make_dataloader(cfg, logger, dataset, batch_size, 0, numpy=True)
    # if rv:
    #     all_means = torch.load('all_means.pt')
    #     # all_vars = torch.load('all_vars.pt')

    #     rand_vars = torch.ones(batch_size, model_s.zdim)
    #     out_var = rand_vars.unsqueeze(1).to(device)

    #     all_means_varmeans = torch.var_mean(all_means, 0, unbiased=False)
    #     std_range = 1.
    #     means_mins = all_means_varmeans[1]-std_range*all_means_varmeans[0]**(0.5)
    #     means_maxs = all_means_varmeans[1]+std_range*all_means_varmeans[0]**(0.5)

    activations = []
    num_images_processed = 0
    # for _ in tqdm(range(0, num_images, batch_size)):
    for i in range(0, num_images, batch_size):
        # torch.cuda.set_device(0)
        noise_batch = torch.randn(size=(batch_size, model_s.zdim)).to(device)
        # noise_batch = torch.FloatTensor(batch_size, model_s.zdim).normal_(mean=0.,std=.9).to(device)
        # print(noise_batch.shape)
        # if rv:
            # # means = torch.FloatTensor(100, 1).normal_(all_means_varmeans[1][0], std_range*all_means_varmeans[0][0]**(0.5))
            # means = torch.FloatTensor(batch_size, 1).uniform_(means_mins[0].item(), means_maxs[0].item())
            # for l in range(1, model_s.zdim):
            #     # means = torch.cat((means, torch.FloatTensor(100, 1).normal_(all_means_varmeans[1][l], std_range*all_means_varmeans[0][l]**(0.5))), 1)
            #     means = torch.cat((means, torch.FloatTensor(batch_size, 1).uniform_(means_mins[l].item(), means_maxs[l].item())), 1)

            # means = means.unsqueeze(1).to(device)
            # noise_batch = torch.cat((means, out_var), 1)
        if rv:
            noise_batch_var = torch.ones(size=(batch_size, model_s.zdim)).to(device)
            noise_batch1 = torch.unsqueeze(noise_batch, 1)
            noise_batch_var = torch.unsqueeze(noise_batch_var, 1)
            noise_batch = torch.cat((noise_batch1, noise_batch_var), 1)
        
            
        images = model_s.sample(noise_batch)
        # images = model_s.generate(lod, 1, count=batch_size, no_truncation=True)
        images = images.data.cpu().numpy()
        images = np.clip(images * 255, 0, 255).astype(np.uint8)

        # images = np.clip((images.cpu().numpy() + 1.0) * 127, 0, 255).astype(np.uint8)
        images = images / 255.0
        # if i == 0:
        #     print(images[0].min())
        #     print(images[0].max())
        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.to(device)
        res = model(batch)[0]

        activations.append(res.cpu().data.numpy().reshape(res.size(0), -1))

    activations = np.concatenate(activations)
    activations = activations[:num_images]
    print("total generated activations: ", activations.shape)

    if verbose:
        print(' done')

    return activations


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics_given_dataset(dataloader, model, batch_size=50,
                                                  dims=2048, cuda=False, verbose=False, device=torch.device("cpu"),
                                                  num_images=50000):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations_given_dataset(dataloader, model, batch_size, dims, cuda, verbose, device, num_images)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_activation_statistics_generate(model_s, model, batch_size=50,
                                             dims=2048, cuda=False, verbose=False, device=torch.device("cpu"),
                                             num_images=50000, rv=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations_generate(model_s, model, batch_size, dims, cuda, verbose, device, num_images, rv=rv)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_activation_statistics(files, model, batch_size=50,
                                    dims=2048, cuda=False, verbose=False, device=torch.device("cpu")):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, cuda, verbose, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, dims, cuda, device):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, cuda, device=device)

    return m, s


def _compute_statistics_of_given_dataset(dataloader, model, batch_size, dims, cuda, device, num_images):
    m, s = calculate_activation_statistics_given_dataset(dataloader, model, batch_size,
                                                         dims, cuda, device=device, num_images=num_images)

    return m, s


def _compute_statistics_of_generate(model_s, model, batch_size, dims, cuda, device, num_images, rv=False):
    m, s = calculate_activation_statistics_generate(model_s, model, batch_size,
                                                    dims, cuda, device=device, num_images=num_images, rv=rv)

    return m, s


def calculate_fid_given_paths(paths, batch_size, cuda, dims, device):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.to(device)

    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size,
                                         dims, cuda, device)
    m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size,
                                         dims, cuda, device)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def calculate_fid_given_dataset(dataloader, model_s, batch_size, cuda, dims, device, num_images, rv=False):
    """Calculates the FID"""

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.to(device)

    m1, s1 = _compute_statistics_of_given_dataset(dataloader, model, batch_size,
                                                  dims, cuda, device, num_images)
    m2, s2 = _compute_statistics_of_generate(model_s, model, batch_size,
                                             dims, cuda, device, num_images, rv=rv)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def save_from_dataset(img_datasetloader, save_path, num_images):
    """
    Saves images from ImageDataset.
    :param img_dataset:
    :param save_path:
    :param num_images:
    :return:
    """
    count = 0
    for batch in img_datasetloader:
        if count >= num_images:
            break
        num_images_in_batch = len(batch)
        for i in range(num_images_in_batch):
            vutils.save_image(batch[i].data.cpu(), save_path + '/image_{}.jpg'.format(count), nrow=1)
            count += 1


def generate_from_model(model, save_path, num_images, batch_size=32, device=torch.device("cpu")):
    """
    Generate images from model
    """
    count = 0
    model.eval()
    while count < num_images:
        noise_batch = torch.randn(size=(batch_size, model.zdim)).to(device)
        generated = model.sample(noise_batch)
        for i in range(len(generated)):
            if count >= num_images:
                break
            vutils.save_image(generated[i].data.cpu(), save_path + '/image_{}.jpg'.format(count), nrow=1)
            count += 1


def calc_fid_from_dataset_generate(cfg, dataset, model_s, batch_size, cuda, dims, device, num_images):
    with torch.no_grad():
        fid = calculate_fid_given_dataset(cfg, dataset, model_s, batch_size, cuda, dims, device, num_images)
        # sanity check
        # fid = calculate_fid_given_dataset_sanity(cfg, dataset, model_s, batch_size, cuda, dims, device, num_images)
    return fid
