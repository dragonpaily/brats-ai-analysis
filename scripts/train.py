import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# Add the project's root directory to the Python path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Import our custom toolboxes ---
from src.data_utils import load_nifti, preprocess_mri_volumes, preprocess_segmentation_mask
from src.model import nnunet_3d

# ===================================================================
# --- Training-Specific Functions (from your notebook) ---
# ===================================================================

# Source: Cell 4 (augment_mri_sample function)
@tf.function
def augment_mri_sample(image, seg_tensor):
    """Applies data augmentation to MRI images and segmentation masks."""
    if isinstance(seg_tensor, tuple):
        seg_one_hot, seg_out2, seg_out3, seg_out4 = seg_tensor
    else:
        seg_one_hot = seg_tensor

    def resize_to_original(tensor, target_shape=(128, 128, 128), is_seg=False):
        method = 'nearest' if is_seg else 'bilinear'
        if len(tensor.shape) == 4:
            tensor = tf.image.resize(tensor, target_shape[:2], method=method)
            tensor = tf.transpose(tensor, [2, 0, 1, 3])
            tensor = tf.image.resize(tensor, [target_shape[2], target_shape[1]], method=method)
            tensor = tf.transpose(tensor, [1, 2, 0, 3])
        return tensor

    if tf.random.uniform(()) > 0.4:
        tc_mask = seg_one_hot[..., 1]
        if tf.reduce_sum(tc_mask) > 0:
            coords = tf.where(tc_mask > 0)
            center = tf.reduce_mean(tf.cast(coords, tf.float32), axis=0)
            crop_size = 96
            start = tf.cast(center - crop_size // 2, tf.int32)
            start = tf.clip_by_value(start, 0, 128 - crop_size)
            image = image[start[0]:start[0]+crop_size, start[1]:start[1]+crop_size, start[2]:start[2]+crop_size, :]
            seg_one_hot = seg_one_hot[start[0]:start[0]+crop_size, start[1]:start[1]+crop_size, start[2]:start[2]+crop_size, :]
            image = resize_to_original(image, is_seg=False)
            seg_one_hot = resize_to_original(seg_one_hot, is_seg=True)
            if isinstance(seg_tensor, tuple):
                seg_out2 = resize_to_original(seg_one_hot, target_shape=(64, 64, 64), is_seg=True)
                seg_out3 = resize_to_original(seg_one_hot, target_shape=(32, 32, 32), is_seg=True)
                seg_out4 = resize_to_original(seg_one_hot, target_shape=(16, 16, 16), is_seg=True)

    if tf.random.uniform(()) > 0.6:
        axis = tf.random.uniform((), minval=0, maxval=3, dtype=tf.int32)
        image = tf.reverse(image, axis=[axis])
        seg_one_hot = tf.reverse(seg_one_hot, axis=[axis])
        if isinstance(seg_tensor, tuple):
            seg_out2, seg_out3, seg_out4 = [tf.reverse(s, axis=[axis]) for s in [seg_out2, seg_out3, seg_out4]]

    if tf.random.uniform(()) > 0.5:
        image *= tf.random.uniform((), minval=0.95, maxval=1.05)

    if tf.random.uniform(()) > 0.4:
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=tf.random.uniform((), minval=0.0, maxval=0.02))
        image = tf.clip_by_value(image + noise, -3.0, 3.0)

    image = tf.convert_to_tensor(image, dtype=tf.float32)
    seg_one_hot = tf.convert_to_tensor(seg_one_hot, dtype=tf.float32)
    if isinstance(seg_tensor, tuple):
        seg_out2, seg_out3, seg_out4 = [tf.convert_to_tensor(s, dtype=tf.float32) for s in [seg_out2, seg_out3, seg_out4]]
        return image, (seg_one_hot, seg_out2, seg_out3, seg_out4)
    return image, seg_one_hot

# Source: Cell 5 of your notebook
def data_generator(patient_folders, data_path, is_training=True, target_shape=(128, 128, 128), n_classes=4):
    """A generator function to yield batches of data for training/validation."""
    for i, patient_id in enumerate(patient_folders):
        try:
            patient_dir = os.path.join(data_path, patient_id)
            image_tensor, _ = preprocess_mri_volumes(patient_dir, patient_id, target_shape)
            
            seg_path = os.path.join(patient_dir, f"{patient_id}-seg.nii.gz")
            seg_data, _ = load_nifti(seg_path)
            
            # For training, we need the deep supervision masks
            seg_tensor_tuple = preprocess_segmentation_mask(seg_data, target_shape, n_classes=n_classes, deep_supervision=True)
            
            if is_training:
                image_tensor, seg_tensor_tuple = augment_mri_sample(image_tensor, seg_tensor_tuple)
            
            yield image_tensor, seg_tensor_tuple
            
        except Exception as e:
            print(f"\nSkipping patient {patient_id} due to an error: {e}")
            continue

# Source: Cell 8 (Custom Keras Metrics)
class DiceET(tf.keras.metrics.Metric):
    def __init__(self, name='dice_et', **kwargs):
        super(DiceET, self).__init__(name=name, **kwargs)
        self.intersection = self.add_weight(name='et_inter', initializer='zeros')
        self.total = self.add_weight(name='et_total', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_f = tf.cast(y_true[..., 3], tf.float32)
        y_pred_f = tf.cast(y_pred[..., 3] > 0.5, tf.float32)
        self.intersection.assign_add(tf.reduce_sum(y_true_f * y_pred_f))
        self.total.assign_add(tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f))
    def result(self):
        dice = (2. * self.intersection + K.epsilon()) / (self.total + K.epsilon())
        return tf.where(tf.equal(self.total, 0), 1.0, dice)
    def reset_state(self):
        self.intersection.assign(0.0)
        self.total.assign(0.0)

class DiceTC_Derived(tf.keras.metrics.Metric):
    def __init__(self, name='dice_tc_derived', **kwargs):
        super(DiceTC_Derived, self).__init__(name=name, **kwargs)
        self.intersection = self.add_weight(name='tc_inter', initializer='zeros')
        self.total = self.add_weight(name='tc_total', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_tc = tf.cast(tf.logical_or(tf.cast(y_true[..., 1], tf.bool), tf.cast(y_true[..., 3], tf.bool)), tf.float32)
        y_pred_tc = tf.cast(tf.logical_or(tf.cast(y_pred[..., 1] > 0.5, tf.bool), tf.cast(y_pred[..., 3] > 0.5, tf.bool)), tf.float32)
        self.intersection.assign_add(tf.reduce_sum(y_true_tc * y_pred_tc))
        self.total.assign_add(tf.reduce_sum(y_true_tc) + tf.reduce_sum(y_pred_tc))
    def result(self):
        dice = (2. * self.intersection + K.epsilon()) / (self.total + K.epsilon())
        return tf.where(tf.equal(self.total, 0), 1.0, dice)
    def reset_state(self):
        self.intersection.assign(0.0)
        self.total.assign(0.0)

class DiceWT_Derived(tf.keras.metrics.Metric):
    def __init__(self, name='dice_wt_derived', **kwargs):
        super(DiceWT_Derived, self).__init__(name=name, **kwargs)
        self.intersection = self.add_weight(name='wt_inter', initializer='zeros')
        self.total = self.add_weight(name='wt_total', initializer='zeros')
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_wt = tf.cast(tf.logical_or(tf.logical_or(tf.cast(y_true[..., 1], tf.bool), tf.cast(y_true[..., 2], tf.bool)), tf.cast(y_true[..., 3], tf.bool)), tf.float32)
        y_pred_wt = tf.cast(tf.logical_or(tf.logical_or(tf.cast(y_pred[..., 1] > 0.5, tf.bool), tf.cast(y_pred[..., 2] > 0.5, tf.bool)), tf.cast(y_pred[..., 3] > 0.5, tf.bool)), tf.float32)
        self.intersection.assign_add(tf.reduce_sum(y_true_wt * y_pred_wt))
        self.total.assign_add(tf.reduce_sum(y_true_wt) + tf.reduce_sum(y_pred_wt))
    def result(self):
        dice = (2. * self.intersection + K.epsilon()) / (self.total + K.epsilon())
        return tf.where(tf.equal(self.total, 0), 1.0, dice)
    def reset_state(self):
        self.intersection.assign(0.0)
        self.total.assign(0.0)

class CombinedDice_Derived_TC_WT_ET(tf.keras.metrics.Metric):
    def __init__(self, name='combined_dice_derived', **kwargs):
        super(CombinedDice_Derived_TC_WT_ET, self).__init__(name=name, **kwargs)
        self.et_dice = DiceET()
        self.tc_dice = DiceTC_Derived()
        self.wt_dice = DiceWT_Derived()
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.et_dice.update_state(y_true, y_pred, sample_weight)
        self.tc_dice.update_state(y_true, y_pred, sample_weight)
        self.wt_dice.update_state(y_true, y_pred, sample_weight)
    def result(self):
        return (self.et_dice.result() + self.tc_dice.result() + self.wt_dice.result()) / 3.0
    def reset_state(self):
        self.et_dice.reset_state()
        self.tc_dice.reset_state()
        self.wt_dice.reset_state()

# Source: Cell 11 (combo_loss function)
@tf.function
def combo_loss(y_true_list, y_pred_list, smooth=1e-6, alpha=0.5, beta=1.0, gamma_focal=1.5):
    class_weights = tf.constant([0.1, 6.2, 2.5, 6.0], dtype=tf.float32)
    def weighted_focal_ce(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
        focal_weight = tf.pow(1.0 - y_pred, gamma_focal)
        ce = -tf.reduce_sum(y_true * focal_weight * tf.math.log(y_pred) * class_weights, axis=-1)
        return tf.reduce_mean(ce)
    def dice_loss(y_true, y_pred):
        num = sum(tf.reduce_sum(y_true[..., c] * y_pred[..., c]) * class_weights[c] for c in range(1, 4))
        den = sum(tf.reduce_sum(y_true[..., c] + y_pred[..., c]) * class_weights[c] for c in range(1, 4))
        return 1.0 - (2.0 * num + smooth) / (den + smooth)
    def tversky_loss(y_true, y_pred, alpha_tv=0.7, beta_tv=0.3):
        num = sum(tf.reduce_sum(y_true[..., c] * y_pred[..., c]) * class_weights[c] for c in range(1, 4))
        den = num + alpha_tv * sum(tf.reduce_sum(y_true[..., c] * (1 - y_pred[..., c])) * class_weights[c] for c in range(1, 4)) + \
              beta_tv * sum(tf.reduce_sum((1 - y_true[..., c]) * y_pred[..., c]) * class_weights[c] for c in range(1, 4))
        return 1.0 - (num + smooth) / (den + smooth)

    total_loss = 0.0
    for y_true, y_pred in zip(y_true_list, y_pred_list):
        ce_component = weighted_focal_ce(y_true, y_pred)
        dice_component = dice_loss(y_true, y_pred)
        tversky_component = tversky_loss(y_true, y_pred)
        combined = alpha * ce_component + beta * dice_component + 0.4 * tversky_component
        total_loss += combined
    return total_loss / float(len(y_true_list))

def main():
    parser = argparse.ArgumentParser(description="Train a 3D U-Net model for BraTS segmentation.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing all patient folders.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model weights and logs.")
    parser.add_argument("--experiment_name", type=str, default="BraTS_Segmentation_Run", help="Name for this specific experiment run.")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs to train for.")
    args = parser.parse_args()

    # --- Setup Directories ---
    CHECKPOINT_DIR = os.path.join(args.output_dir, args.experiment_name, "checkpoints")
    LOG_DIR = os.path.join(args.output_dir, args.experiment_name, "logs")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- Data Splitting (from Cell 7) ---
    all_patients = sorted([os.path.basename(p) for p in glob.glob(os.path.join(args.data_dir, "*")) if os.path.isdir(p)])
    train_val_patients, test_patients = train_test_split(all_patients, test_size=0.15, random_state=42)
    train_patients, val_patients = train_test_split(train_val_patients, test_size=0.15/0.85, random_state=42)

    # --- Create tf.data Pipelines (from Cell 10) ---
    output_signature = (
        tf.TensorSpec(shape=(128, 128, 128, 4), dtype=tf.float32),
        (
            tf.TensorSpec(shape=(128, 128, 128, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(64, 64, 64, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(32, 32, 32, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(16, 16, 16, 4), dtype=tf.float32)
        )
    )
    train_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(train_patients, args.data_dir, is_training=True),
        output_signature=output_signature
    ).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(val_patients, args.data_dir, is_training=False),
        output_signature=output_signature
    ).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    
    # --- Build and Compile Model ---
    model = nnunet_3d(input_shape=(128, 128, 128, 4), n_classes=4)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=combo_loss,
        metrics={'out_final': [CombinedDice_Derived_TC_WT_ET(), DiceET(), DiceTC_Derived(), DiceWT_Derived()]}
    )
    model.summary()

    # --- Setup Callbacks ---
    callbacks = [
        ModelCheckpoint(filepath=os.path.join(CHECKPOINT_DIR, "best_model.weights.h5"), save_best_only=True, monitor="val_out_final_combined_dice_derived", mode="max", save_weights_only=True, verbose=1),
        CSVLogger(os.path.join(LOG_DIR, "training_log.csv")),
        TensorBoard(log_dir=LOG_DIR),
        ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1)
    ]

    # Start Training ---
    print("\n--- Starting Model Training ---")
    model.fit(
        train_dataset,
        epochs=args.epochs,
        steps_per_epoch=max(1, len(train_patients) // args.batch_size),
        validation_data=val_dataset,
        validation_steps=max(1, len(val_patients) // args.batch_size),
        callbacks=callbacks
    )
    print("\n--- Model Training Finished ---")

if __name__ == '__main__':
    main()



