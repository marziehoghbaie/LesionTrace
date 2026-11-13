import cv2
import math
import torch
import numpy as np


def write_on_image(img, text):
    cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.putText(cv2_img, f"{text}", (10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return img


def reshape_transform(tensor):
    # torch.Size([1, wxh+1, 768])
    _, sp, _ = tensor.shape
    height = width = int(math.sqrt(sp - 1))
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform_with_registers(tensor):
    _, sp, _ = tensor.shape
    height = width = int(math.sqrt(sp - 1))
    # x: shape (batch_size, seq_len, embed_dim)
    result = tensor[:, 5:].reshape(tensor.size(0),
                                   height, width, tensor.size(2))  # remaining patch tokens
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform_swin(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))
    # Bring the channels to the first dimension, like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result



def denormalize(tensor, mean, std):
    """
    tensor: (C, H, W)
    mean, std: list or tuple of means and stds per channel
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean