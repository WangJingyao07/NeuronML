import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, BatchNorm2d


def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
              for i, path in zip(labels, paths) \
              for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images


def conv_block(inp, cweight, bweight, activation=F.relu, max_pool_pad='VALID', residual=False):
    """ Perform convolution, batch normalization, nonlinearity, and max pool """
    stride = 2
    no_stride = 1

    if FLAGS.max_pool:
        conv_output = F.conv2d(inp, cweight, stride=no_stride, padding='same') + bweight
    else:
        conv_output = F.conv2d(inp, cweight, stride=stride, padding='same') + bweight
    normed = normalize(conv_output, activation)
    if FLAGS.max_pool:
        normed = F.max_pool2d(normed, kernel_size=stride, stride=stride, padding=0 if max_pool_pad == 'VALID' else 1)
    return normed


def normalize(inp, activation):
    if FLAGS.norm == 'batch_norm':
        norm_layer = BatchNorm2d(inp.size(1)).to(inp.device)
        normed = norm_layer(inp)
    elif FLAGS.norm == 'layer_norm':
        norm_layer = LayerNorm(inp.size()[1:]).to(inp.device)
        normed = norm_layer(inp)
    else:
        normed = inp
    if activation is not None:
        normed = activation(normed)
    return normed


def mse(pred, label):
    pred = pred.view(-1)
    label = label.view(-1)
    return F.mse_loss(pred, label)

def xent(pred, label):
    return F.cross_entropy(pred, label) / FLAGS.update_batch_size
