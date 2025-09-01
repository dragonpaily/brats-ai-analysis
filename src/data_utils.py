
import tensorflow as tf
import numpy as np
from scipy.ndimage import map_coordinates

import tensorflow as tf
import numpy as np
import nibabel as nib

def Load_nifti(data_path):
    """Load a NIfTI file and return its data as numpy array."""
    img = nib.load(data_path)
    img_data = img.get_fdata()
    return img_data

def z_score_normalize(image, mask=None):
    """Z-score normalization with optional masking to compute statistics only from specific regions"""
    if mask is not None:
        valid_pixels = image[mask > 0]
        if valid_pixels.size == 0:
            valid_pixels = image.flatten()  # Fallback to all pixels if mask is empty
    else:
        valid_pixels = image.flatten()
    mean = np.mean(valid_pixels)
    std = np.std(valid_pixels)
    if std == 0:
        std = 1.0  #Prevent division by zero
    return (image - mean) / std

def resize_image(image, target_shape=(128, 128, 128), method='bilinear'):
    """Resize a 3D image to target shape using TensorFlow operations."""
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(image, dtype=tf.float32)
    if len(image.shape) == 3:
        image = tf.expand_dims(image, axis=-1)#compatable with tf.image.resize
    image = tf.image.resize(image, [target_shape[0], target_shape[1]], method=method)
    image = tf.transpose(image, [2, 0, 1, 3])
    image = tf.image.resize(image, [target_shape[2], target_shape[1]], method=method)
    image = tf.transpose(image, [1, 2, 0, 3])
    image = image[:target_shape[0], :target_shape[1], :target_shape[2], :]# ensuring the
    return image



def preprocess_segmentation_mask(seg, target_shape=(128, 128, 128), deep_supervision=False):
    """Resize and one-hot encode segmentation mask."""

    if not isinstance(seg, tf.Tensor):
        seg = tf.convert_to_tensor(seg, dtype=tf.float32)
    if len(seg.shape) == 3:
        seg = tf.expand_dims(seg, axis=-1)

    seg = tf.image.resize(seg, [target_shape[0], target_shape[1]], method='nearest')
    seg = tf.transpose(seg, [2, 0, 1, 3])
    seg = tf.image.resize(seg, [target_shape[2], target_shape[1]], method='nearest')
    seg = tf.transpose(seg, [1, 2, 0, 3])
    seg = seg[:target_shape[0], :target_shape[1], :target_shape[2], :]
    seg_int = tf.cast(tf.round(seg), tf.int32)
    seg_int = tf.squeeze(seg_int, axis=-1)
    seg_one_hot = tf.one_hot(seg_int, depth=4)
    seg_one_hot = tf.ensure_shape(seg_one_hot, (target_shape[0], target_shape[1], target_shape[2], 4))

    if deep_supervision:
        seg_out2 = resize_image(seg, (64, 64, 64), method='nearest')
        seg_out2 = tf.cast(tf.round(seg_out2), tf.int32)
        seg_out2 = tf.squeeze(seg_out2, axis=-1)
        seg_out2 = tf.one_hot(seg_out2, depth=4)
        seg_out2 = tf.ensure_shape(seg_out2, (64, 64, 64, 4))
        seg_out3 = resize_image(seg, (32, 32, 32), method='nearest')
        seg_out3 = tf.cast(tf.round(seg_out3), tf.int32)
        seg_out3 = tf.squeeze(seg_out3, axis=-1)
        seg_out3 = tf.one_hot(seg_out3, depth=4)
        seg_out3 = tf.ensure_shape(seg_out3, (32, 32, 32, 4))
        seg_out4 = resize_image(seg, (16, 16, 16), method='nearest')
        seg_out4 = tf.cast(tf.round(seg_out4), tf.int32)
        seg_out4 = tf.squeeze(seg_out4, axis=-1)
        seg_out4 = tf.one_hot(seg_out4, depth=4)
        seg_out4 = tf.ensure_shape(seg_out4, (16, 16, 16, 4))
        return (seg_one_hot, seg_out2, seg_out3, seg_out4)
    return seg_one_hot

def preprocess_sample(t1c, t1n, t2f, t2w, seg=None, target_shape=(128, 128, 128)):
    """Normalize, resize, and prepare MRI images and segmentation masks with Z-score normalization."""
    threshold = np.percentile(t1c, 5)
    brain_mask = (t1c > threshold)
    t1c_norm = z_score_normalize(t1c, brain_mask)
    t1n_norm = z_score_normalize(t1n, brain_mask)
    t2f_norm = z_score_normalize(t2f, brain_mask)
    t2w_norm = z_score_normalize(t2w, brain_mask)
    t1c_resized = resize_image(t1c_norm, target_shape)
    t1n_resized = resize_image(t1n_norm, target_shape)
    t2f_resized = resize_image(t2f_norm, target_shape)
    t2w_resized = resize_image(t2w_norm, target_shape)
    t1c_final = tf.squeeze(t1c_resized, axis=-1)
    t1n_final = tf.squeeze(t1n_resized, axis=-1)
    t2f_final = tf.squeeze(t2f_resized, axis=-1)
    t2w_final = tf.squeeze(t2w_resized, axis=-1)
    image_tensor = tf.stack([t1c_final, t1n_final, t2f_final, t2w_final], axis=-1)
    return image_tensor
import tensorflow as tf



@tf.function
def augment_mri_sample(image, seg_tensor):
    """Apply stable data augmentation to MRI images and segmentation masks to improve Tumor Core (TC) segmentation."""
    # Handle deep supervision: unpack tuple of segmentation masks if provided
    if isinstance(seg_tensor, tuple):
        seg_one_hot, seg_out2, seg_out3, seg_out4 = seg_tensor
    else:
        seg_one_hot = seg_tensor

    # Helper function to resize tensors back to original shape after cropping
    def resize_to_original(tensor, target_shape=(128, 128, 128), is_seg=False):
        if len(tensor.shape) == 4:  # Image or segmentation (H, W, D, C)
            # Resize XY plane
            tensor = tf.image.resize(tensor, target_shape[:2], method='bilinear' if not is_seg else 'nearest')
            # Transpose to resize Z dimension
            tensor = tf.transpose(tensor, [2, 0, 1, 3])
            tensor = tf.image.resize(tensor, [target_shape[2], target_shape[1]], method='bilinear' if not is_seg else 'nearest')
            # Restore original axis order
            tensor = tf.transpose(tensor, [1, 2, 0, 3])
        return tensor

    # TC-focused Random Cropping to emphasize TC boundaries
    if tf.random.uniform(()) > 0.4:  # 60% probability to prioritize TC
        tc_mask = seg_one_hot[..., 1]  # Extract TC channel
        if tf.reduce_sum(tc_mask) > 0:  # Ensure TC is present
            # Compute TC region center for cropping
            coords = tf.where(tc_mask > 0)
            center = tf.reduce_mean(tf.cast(coords, tf.float32), axis=0)
            crop_size = 96  # Crop to 96x96x96 to focus on TC
            start = tf.cast(center - crop_size // 2, tf.int32)
            start = tf.clip_by_value(start, 0, 128 - crop_size)
            # Crop image and segmentation
            image = image[start[0]:start[0]+crop_size, start[1]:start[1]+crop_size, start[2]:start[2]+crop_size, :]
            seg_one_hot = seg_one_hot[start[0]:start[0]+crop_size, start[1]:start[1]+crop_size, start[2]:start[2]+crop_size, :]
            # Resize back to 128x128x128
            image = resize_to_original(image, is_seg=False)
            seg_one_hot = resize_to_original(seg_one_hot, is_seg=True)
            # Update auxiliary outputs for deep supervision
            if isinstance(seg_tensor, tuple):
                seg_out2 = resize_to_original(seg_one_hot, target_shape=(64, 64, 64), is_seg=True)
                seg_out3 = resize_to_original(seg_one_hot, target_shape=(32, 32, 32), is_seg=True)
                seg_out4 = resize_to_original(seg_one_hot, target_shape=(16, 16, 16), is_seg=True)

    # Random Flipping to balance with other augmentations
    if tf.random.uniform(()) > 0.6:  # 40% probability
        axis = tf.random.uniform((), minval=0, maxval=3, dtype=tf.int32)
        image = tf.reverse(image, axis=[axis])
        seg_one_hot = tf.reverse(seg_one_hot, axis=[axis])
        if isinstance(seg_tensor, tuple):
            seg_out2 = tf.reverse(seg_out2, axis=[axis])
            seg_out3 = tf.reverse(seg_out3, axis=[axis])
            seg_out4 = tf.reverse(seg_out4, axis=[axis])

    # Intensity Scaling to preserve TC contrast
    if tf.random.uniform(()) > 0.5:  # 50% probability
        intensity_factor = tf.random.uniform((), minval=0.95, maxval=1.05)
        image = image * intensity_factor

    # Gaussian Noise for robustness
    if tf.random.uniform(()) > 0.4:  # 60% probability
        noise_level = tf.random.uniform((), minval=0.0, maxval=0.02)
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=noise_level)
        image = image + noise
        image = tf.clip_by_value(image, -3.0, 3.0)

    # Ensure correct data types
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    seg_one_hot = tf.convert_to_tensor(seg_one_hot, dtype=tf.float32)
    if isinstance(seg_tensor, tuple):
        seg_out2 = tf.convert_to_tensor(seg_out2, dtype=tf.float32)
        seg_out3 = tf.convert_to_tensor(seg_out3, dtype=tf.float32)
        seg_out4 = tf.convert_to_tensor(seg_out4, dtype=tf.float32)
        return image, (seg_one_hot, seg_out2, seg_out3, seg_out4)
    return image, seg_one_hot