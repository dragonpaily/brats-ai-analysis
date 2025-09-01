
#cell 8
import tensorflow as tf
from tensorflow.keras import backend as K

# --- Metrics for Original Classes (Channels: 0:BG, 1:NCR, 2:ED, 3:ET) ---

class DiceET(tf.keras.metrics.Metric):
    def __init__(self, name='dice_et', **kwargs):
        super(DiceET, self).__init__(name=name, **kwargs)
        self.intersection = self.add_weight(name='dice_et_intersection', initializer='zeros')
        self.total = self.add_weight(name='dice_et_total', initializer='zeros')
        self.built = True

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_region = tf.cast(y_true[..., 3], tf.float32)
        y_pred_region = tf.cast(y_pred[..., 3] > 0.5, tf.float32)
        self.intersection.assign_add(tf.reduce_sum(y_true_region * y_pred_region))
        self.total.assign_add(tf.reduce_sum(y_true_region) + tf.reduce_sum(y_pred_region))

    def result(self):
        dice = (2. * self.intersection + K.epsilon()) / (self.total + K.epsilon())
        return tf.where(tf.equal(self.total, 0), tf.where(tf.equal(self.intersection, 0), 1.0, 0.0), dice)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.total.assign(0.0)

    def reset_states(self):
        self.reset_state()

class DiceNCR(tf.keras.metrics.Metric):
    def __init__(self, name='dice_ncr', **kwargs):
        super(DiceNCR, self).__init__(name=name, **kwargs)
        self.intersection = self.add_weight(name='dice_ncr_intersection', initializer='zeros')
        self.total = self.add_weight(name='dice_ncr_total', initializer='zeros')
        self.built = True

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_region = tf.cast(y_true[..., 1], tf.float32)
        y_pred_region = tf.cast(y_pred[..., 1] > 0.5, tf.float32)
        self.intersection.assign_add(tf.reduce_sum(y_true_region * y_pred_region))
        self.total.assign_add(tf.reduce_sum(y_true_region) + tf.reduce_sum(y_pred_region))

    def result(self):
        dice = (2. * self.intersection + K.epsilon()) / (self.total + K.epsilon())
        return tf.where(tf.equal(self.total, 0), tf.where(tf.equal(self.intersection, 0), 1.0, 0.0), dice)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.total.assign(0.0)

    def reset_states(self):
        self.reset_state()

class DiceED(tf.keras.metrics.Metric):
    def __init__(self, name='dice_ed', **kwargs):
        super(DiceED, self).__init__(name=name, **kwargs)
        self.intersection = self.add_weight(name='dice_ed_intersection', initializer='zeros')
        self.total = self.add_weight(name='dice_ed_total', initializer='zeros')
        self.built = True

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_region = tf.cast(y_true[..., 2], tf.float32)
        y_pred_region = tf.cast(y_pred[..., 2] > 0.5, tf.float32)
        self.intersection.assign_add(tf.reduce_sum(y_true_region * y_pred_region))
        self.total.assign_add(tf.reduce_sum(y_true_region) + tf.reduce_sum(y_pred_region))

    def result(self):
        dice = (2. * self.intersection + K.epsilon()) / (self.total + K.epsilon())
        return tf.where(tf.equal(self.total, 0), tf.where(tf.equal(self.intersection, 0), 1.0, 0.0), dice)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.total.assign(0.0)

    def reset_states(self):
        self.reset_state()

# --- Metrics for Derived Regions (TC, WT) from original BG,NCR,ED,ET channels ---

class DiceTC_Derived(tf.keras.metrics.Metric):
    def __init__(self, name='dice_tc_derived', **kwargs):
        super(DiceTC_Derived, self).__init__(name=name, **kwargs)
        self.intersection = self.add_weight(name='dtc_intersection', initializer='zeros')
        self.total = self.add_weight(name='dtc_total', initializer='zeros')
        self.built = True

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)
        # TC = NCR (channel 1) or ET (channel 3)
        y_true_region = tf.cast(tf.logical_or(tf.cast(y_true_f[..., 1], tf.bool), tf.cast(y_true_f[..., 3], tf.bool)), tf.float32)
        y_pred_region = tf.cast(tf.logical_or(tf.cast(y_pred_f[..., 1] > 0.5, tf.bool), tf.cast(y_pred_f[..., 3] > 0.5, tf.bool)), tf.float32)
        self.intersection.assign_add(tf.reduce_sum(y_true_region * y_pred_region))
        self.total.assign_add(tf.reduce_sum(y_true_region) + tf.reduce_sum(y_pred_region))

    def result(self):
        dice = (2. * self.intersection + K.epsilon()) / (self.total + K.epsilon())
        return tf.where(tf.equal(self.total, 0), tf.where(tf.equal(self.intersection, 0), 1.0, 0.0), dice)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.total.assign(0.0)

    def reset_states(self):
        self.reset_state()

class DiceWT_Derived(tf.keras.metrics.Metric):
    def __init__(self, name='dice_wt_derived', **kwargs):
        super(DiceWT_Derived, self).__init__(name=name, **kwargs)
        self.intersection = self.add_weight(name='dwt_intersection', initializer='zeros')
        self.total = self.add_weight(name='dwt_total', initializer='zeros')
        self.built = True

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)
        # WT = NCR (channel 1) or ED (channel 2) or ET (channel 3)
        y_true_region = tf.cast(tf.logical_or(tf.logical_or(tf.cast(y_true_f[..., 1], tf.bool), tf.cast(y_true_f[..., 2], tf.bool)), tf.cast(y_true_f[..., 3], tf.bool)), tf.float32)
        y_pred_ncr = tf.cast(y_pred_f[..., 1] > 0.5, tf.bool)
        y_pred_ed = tf.cast(y_pred_f[..., 2] > 0.5, tf.bool)
        y_pred_et = tf.cast(y_pred_f[..., 3] > 0.5, tf.bool)
        y_pred_region = tf.cast(tf.logical_or(tf.logical_or(y_pred_ncr, y_pred_ed), y_pred_et), tf.float32)
        self.intersection.assign_add(tf.reduce_sum(y_true_region * y_pred_region))
        self.total.assign_add(tf.reduce_sum(y_true_region) + tf.reduce_sum(y_pred_region))

    def result(self):
        dice = (2. * self.intersection + K.epsilon()) / (self.total + K.epsilon())
        return tf.where(tf.equal(self.total, 0), tf.where(tf.equal(self.intersection, 0), 1.0, 0.0), dice)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.total.assign(0.0)

    def reset_states(self):
        self.reset_state()

# --- Combined Dice for ET, Derived TC, Derived WT ---

class CombinedDice_Derived_TC_WT_ET(tf.keras.metrics.Metric):
    def __init__(self, name='combined_dice_derived', **kwargs):
        super(CombinedDice_Derived_TC_WT_ET, self).__init__(name=name, **kwargs)
        self.et_dice_metric = DiceET()
        self.tc_dice_derived_metric = DiceTC_Derived()
        self.wt_dice_derived_metric = DiceWT_Derived()
        self.built = True

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.et_dice_metric.update_state(y_true, y_pred, sample_weight)
        self.tc_dice_derived_metric.update_state(y_true, y_pred, sample_weight)
        self.wt_dice_derived_metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return (self.et_dice_metric.result() + self.tc_dice_derived_metric.result() + self.wt_dice_derived_metric.result()) / 3.0

    def reset_state(self):
        self.et_dice_metric.reset_state()
        self.tc_dice_derived_metric.reset_state()
        self.wt_dice_derived_metric.reset_state()

    def reset_states(self):
        self.reset_state()
        
        
        
@tf.function
def combo_loss(y_true_list, y_pred_list, smooth=1e-6, alpha=0.5, beta=1.0, gamma_focal=1.5, tversky_coeff=0.4):
    # Class weights for [BG, NCR, ED, ET] as per your setup that achieved good results
    class_weights = tf.constant([0.1, 6.2, 2.5, 6.0], dtype=tf.float32)

    def weighted_focal_categorical_crossentropy(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        focal_weight = tf.pow(1.0 - y_pred, gamma_focal)
        crossentropy = -tf.reduce_sum(y_true * focal_weight * tf.math.log(y_pred) * class_weights, axis=-1)
        return tf.reduce_mean(crossentropy)

    def dice_loss_internal(y_true, y_pred):
        numerator = 0.0
        denominator = 0.0
        for c in range(1, 4):
            class_true = y_true[..., c]
            class_pred = y_pred[..., c]
            focal_weight = tf.pow(1.0 - tf.reduce_mean(class_pred), gamma_focal / 2.0)
            intersection = tf.reduce_sum(class_true * class_pred) * focal_weight
            union = tf.reduce_sum(class_true) + tf.reduce_sum(class_pred)
            numerator += intersection * class_weights[c]
            denominator += union * class_weights[c]
        dice = (2.0 * numerator + smooth) / (denominator + smooth)
        return 1.0 - dice

    def tversky_loss_internal(y_true, y_pred, alpha_tv=0.7, beta_tv=0.3):
        numerator = 0.0
        denominator = 0.0
        for c in range(1, 4):
            class_true = y_true[..., c]
            class_pred = y_pred[..., c]
            true_pos = tf.reduce_sum(class_true * class_pred)
            false_neg = tf.reduce_sum(class_true * (1 - class_pred))
            false_pos = tf.reduce_sum((1 - class_true) * class_pred)
            tversky = (true_pos + smooth) / (true_pos + alpha_tv * false_neg + beta_tv * false_pos + smooth)
            numerator += tversky * class_weights[c]
            denominator += class_weights[c]
        tversky = numerator / denominator
        return 1.0 - tversky

        # Average of weighted Tversky indices
        average_weighted_tversky = 1.0 # Default to max loss if no relevant classes considered
        if sum_of_class_weights_for_norm > 0:
             average_weighted_tversky = weighted_tversky_sum / sum_of_class_weights_for_norm
        return 1.0 - average_weighted_tversky

    total_loss = tf.constant(0.0, dtype=tf.float32)
    for i in tf.range(len(y_true_list)):
        y_true = y_true_list[i]
        y_pred = y_pred_list[i]
        if y_pred.shape[1:-1] != y_true.shape[1:-1]:
            y_pred = tf.image.resize(y_pred, y_true.shape[1:-1], method='nearest')
        ce_component = weighted_focal_categorical_crossentropy(y_true, y_pred)
        dice_component = dice_loss_internal(y_true, y_pred)
        tversky_component = tversky_loss_internal(y_true, y_pred, alpha_tv=0.7, beta_tv=0.3)
        combined = alpha * ce_component + beta * dice_component + 0.4 * tversky_component
        total_loss += combined
    return total_loss / tf.cast(len(y_true_list), tf.float32)
    #11111