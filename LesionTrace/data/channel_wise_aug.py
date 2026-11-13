import random

import PIL
import cv2
import numpy as np
from imgaug import augmenters as iaa
from scipy import ndimage
from vidaug import augmentors as va
import warnings

warnings.filterwarnings("ignore")
random.seed(42)
import numpy as np
from scipy import ndimage
import cv2
import torchvision.transforms.functional as F
import torch

from PIL import Image, ImageEnhance

def identity(images):
    return images


def eraser_gray(images, s_l=0.1, s_h=0.2, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=255, pixel_level=False):
    input_img = images[0]
    img_h, img_w = input_img.shape

    while True:
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)

        if left + w <= img_w and top + h <= img_h:
            break

    if pixel_level:
        c = np.random.uniform(v_l, v_h, (h, w))

    else:
        c = np.random.uniform(v_l, v_h)

    cutOuts = []
    for idx, input_img in enumerate(images):
        input_img[top:top + h, left:left + w] = c
        cutOuts.append(input_img)
    return np.array(cutOuts)


def eraser(images, s_l=0.1, s_h=0.2, r_1=0.3, r_2=1 / 0.3, pixel_level=False):
    """
    Random erasing / cutout augmentation that works safely for images and masks.

    Args:
        images: list of np.array images/masks to erase, each (H, W, C) or (H, W)
        s_l, s_h: min/max proportion of area to erase
        r_1, r_2: aspect ratio limits
        pixel_level: whether to use pixel-wise random fill

    Returns:
        List of erased images/masks
    """
    input_img = images[0]
    inp_shape = input_img.shape

    if len(inp_shape) == 3:
        img_h, img_w, img_c = inp_shape
    else:
        img_h, img_w = inp_shape
        img_c = 1

    v_l, v_h = 0, 255

    # Randomly select erase rectangle
    while True:
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)

        if left + w <= img_w and top + h <= img_h:
            break

    # # Determine replacement value
    # if pixel_level and img_c > 1:
    #     c = np.random.uniform(v_l, v_h, (h, w, img_c))
    # else:
    #     c = 0

    # Apply erase
    cutOuts = []
    c = 0
    for idx, input_img in enumerate(images):
        img_copy = input_img.copy()
        if len(img_copy.shape) == 2:
            img_copy[top:top + h, left:left + w] = c
        else:
            img_copy[top:top + h, left:left + w, :] = c
        cutOuts.append(img_copy)

    return cutOuts


def translation(images):
    transformed_imgs = []
    T = np.float32([[1, 0, 10], [0, 1, 10]])

    input_img = images[0]
    inp_shape = input_img.shape

    if len(inp_shape) == 3:
        height, width, channel = inp_shape
    else:
        height, width = inp_shape

    for image in images:
        transformed_img = cv2.warpAffine(image, T, (width, height))
        transformed_imgs.append(transformed_img)

    return np.array(transformed_imgs)


def blur(images, k=5):
    aug = iaa.MotionBlur(k=k, angle=[-45, 45])
    blurred_images = aug(images=images)
    return np.array(blurred_images)


# def brightness(images): # debug dont remember when i used this one
#     v = random.uniform(0.1, 1.9)
#
#     result = []
#     for image in images:
#         tensor_img = torch.from_numpy(image).float()
#
#         # add channel dimension if missing
#         if tensor_img.ndim == 2:
#             tensor_img = tensor_img.unsqueeze(0)
#         tensor_img= tensor_img.permute(2, 0, 1)  # -> (C, H, W)
#         bright = F.adjust_brightness(tensor_img, v)
#         result.append(np.transpose(bright.numpy(), (1, 2, 0)))
#
#     return np.array(result)


def brightness(images):  # [0.1,1.9]
    v = random.uniform(0.1, 1.9)
    assert 0.1 <= v <= 1.9
    return np.array([np.array(PIL.ImageEnhance.Brightness(PIL.Image.fromarray(img)).enhance(v)) for img in images])


def salt(images):
    seq = va.Sequential([va.Salt(ratio=20)])
    video_aug = seq(images)
    return np.array(video_aug)


def pepper(images):
    seq = va.Sequential([va.Pepper(ratio=20)])
    video_aug = seq(images)
    return np.array(video_aug)


def consistent_brightness_i(images):
    value = 20.0
    brightness_imgs = []
    for image in images:
        brightness_img = image + value
        brightness_imgs.append(brightness_img)
    return np.array(brightness_imgs)


def consistent_brightness_d(images):
    value = -20.0
    brightness_imgs = []
    for image in images:
        brightness_img = image + value
        brightness_imgs.append(brightness_img)
    return np.array(brightness_imgs)


def rotate(images, angle=10, keep_size=True, order=1, cval=0):
    """
    Rotate images without adding artifacts in extended areas.
    """
    if isinstance(images, np.ndarray) and images.ndim == 3:
        images = [images]

    rotated_imgs = []
    for img in images:
        # Rotate with constant fill
        rotated = ndimage.rotate(img, angle, reshape=True, order=order, mode='constant', cval=cval)

        if keep_size:
            h, w = img.shape[:2]
            rotated = cv2.resize(rotated, (w, h), interpolation=cv2.INTER_LINEAR)

        rotated_imgs.append(rotated)

    return rotated_imgs


def flip_hr(images, angle=180, reshape=False):
    """
    Rotate array safely:
    - Nearest-neighbor for binary masks
    - Cubic for images
    Works with 2D (H,W) and 3D (H,W,C)
    """
    rotated_imgs = []
    for array in images:
        array = np.asarray(array)

        # Determine if array is a mask or image
        is_mask = False

        if array.ndim == 2:
            # 2D array: could be mask or grayscale image
            unique_vals = np.unique(array)
            if len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0, 1])):
                is_mask = True
        elif array.ndim == 3:
            # 3D array: RGB image or mask with channels
            if array.shape[2] == 1:  # single-channel mask
                unique_vals = np.unique(array)
                if len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0, 1])):
                    is_mask = True
            else:
                is_mask = False  # multi-channel image

        # Choose interpolation order
        order = 0 if is_mask else 3

        rotated = ndimage.rotate(array, angle, reshape=reshape, order=order)

        # If mask, threshold to ensure strict binary
        if is_mask:
            rotated = (rotated > 0.5).astype(np.uint8)
        rotated_imgs.append(rotated)
    return rotated_imgs


def offset_aug(imgs):
    offsets = [-100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    results = []
    for img in imgs:
        offset = random.choice(offsets)
        M = np.float32([[1, 0, 0], [0, 1, offset]])
        # Get image size
        height, width = img.shape[:2]
        # Apply affine transformation
        shifted_image = cv2.warpAffine(img, M, (width, height))
        results.append(np.array(shifted_image))
    return results


def salt_and_pepper_noise(image, amount=0.005, salt_vs_pepper=0.5):
    noisy = image.copy()
    h, w = noisy.shape[:2]
    num_pixels = int(amount * h * w)

    # Generate salt noise
    num_salt = int(num_pixels * salt_vs_pepper)
    coords = [np.random.randint(0, i - 1, num_salt) for i in (h, w)]
    if image.ndim == 2:
        noisy[coords[0], coords[1]] = 255
    else:
        noisy[coords[0], coords[1], :] = 255

    # Generate pepper noise
    num_pepper = num_pixels - num_salt
    coords = [np.random.randint(0, i - 1, num_pepper) for i in (h, w)]
    if image.ndim == 2:
        noisy[coords[0], coords[1]] = 0
    else:
        noisy[coords[0], coords[1], :] = 0

    return np.array(noisy)


def salt_pepper(imgs):
    results = []
    noise_ratios = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    for img in imgs:
        amount = random.choice(noise_ratios)
        results.append(salt_and_pepper_noise(img, amount=amount))
    return np.array(results)


def random_crop_pair(imgs, crop_h=400, crop_w=500):
    h, w = imgs[0].shape[:2]

    # Adjust crop sizes if image is smaller
    crop_h = min(crop_h, h)
    crop_w = min(crop_w, w)

    top = np.random.randint(0, h - crop_h + 1)
    left = np.random.randint(0, w - crop_w + 1)

    cropped_imgs = [img[top:top + crop_h, left:left + crop_w] for img in imgs]
    return cropped_imgs


def adjust_hue(img, hue_factor):
    """
    Hue adjustment in [-0.5, 0.5]
    """
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError("hue_factor must be in [-0.5, 0.5]")

    img = np.array(img.convert('HSV'), dtype=np.uint8)
    img[..., 0] = (img[..., 0].astype(int) + int(hue_factor * 255)) % 255
    return Image.fromarray(img, mode='HSV').convert('RGB')

def color_jitter(images, brightness=0, contrast=0, saturation=0, hue=0):
    """
    images: numpy array (N, H, W, C)
    params:
        brightness: float (e.g. 0.8)
        contrast: float
        saturation: float
        hue: float (<= 0.5)
    """

    def get_factor(val, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
        if val == 0:
            return None
        low = max(center - val, bound[0])
        high = min(center + val, bound[1])
        return random.uniform(low, high)

    output = []

    for img in images:
        img = Image.fromarray(img)

        # Sample random factors per image
        b = get_factor(brightness)
        c = get_factor(contrast)
        s = get_factor(saturation)
        h = get_factor(hue, center=0, bound=(-0.5, 0.5))

        # Collect transforms
        transforms = []

        if b is not None:
            transforms.append(lambda im: ImageEnhance.Brightness(im).enhance(b))
        if c is not None:
            transforms.append(lambda im: ImageEnhance.Contrast(im).enhance(c))
        if s is not None:
            transforms.append(lambda im: ImageEnhance.Color(im).enhance(s))
        if h is not None:
            transforms.append(lambda im: adjust_hue(im, h))

        # Shuffle order (important!)
        random.shuffle(transforms)

        # Apply transforms
        for t in transforms:
            img = t(img)

        output.append(np.array(img))

    return np.array(output)

def ensure_3_channels(img):
    if img.ndim == 2:  # grayscale (H, W)
        img = np.stack([img]*3, axis=-1)  # -> (H, W, 3)
    elif img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return img

def random_affine(
        images,
        degrees=30,
        translate=(0.1, 0.1),
        scale=(0.8, 1.2),
        shear=10
):
    """
    images: numpy array (N, H, W, C)
    """

    output = []

    for img in images:
        img = Image.fromarray(img)
        w, h = img.size

        # Random params per image
        angle = random.uniform(-degrees, degrees)

        max_dx = translate[0] * w
        max_dy = translate[1] * h
        translations = (random.uniform(-max_dx, max_dx),
                        random.uniform(-max_dy, max_dy))

        scale_factor = random.uniform(scale[0], scale[1])

        shear_x = random.uniform(-shear, shear)
        shear_y = random.uniform(-shear, shear)

        # Apply affine
        img = img.transform(
            img.size,
            Image.AFFINE,
            data=(
                scale_factor, np.tan(np.radians(shear_x)), translations[0],
                np.tan(np.radians(shear_y)), scale_factor, translations[1]
            ),
            resample=Image.BILINEAR
        )

        # Apply rotation separately (PIL does this better)
        img = img.rotate(angle)
        augmented = ensure_3_channels(np.array(img))
        output.append(augmented)

    return np.array(output)

#
# augmentations = {'identity': identity,
#                  'brightness': brightness,
#                  'blur': blur,
#                  'salt_pepper': salt_pepper,
#                  'rotate': rotate,
#                  'eraser': eraser,
#                  'flip_hr': flip_hr,
#                  'offset_aug': offset_aug,
#                  'random_crop_pair': random_crop_pair,
#                  }


augmentations = {'identity': identity,
                 'brightness': brightness,
                 'blur': blur,
                 'salt_pepper': salt_pepper,
                 'rotate': rotate,
                 'eraser': eraser,
                 'flip_hr': flip_hr,
                 'random_crop_pair': random_crop_pair,
                 'offset_aug': offset_aug,
                 'color_jitter': color_jitter,
                 'random_affine': random_affine
                 }

available_augmentations = [
    'identity',
    'brightness',
    'blur',
    'salt_pepper',
    'rotate',
    'eraser',
    'flip_hr',
    'random_crop_pair',
    'color_jitter',
    'random_affine'
]

tta_augs = [
    'identity',
    'salt_pepper',
    'rotate',        # small angles only
    'brightness',
]