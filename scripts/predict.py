import os
import numpy as np
import tensorflow as tf
import nibabel as nib
import argparse
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#import our custom "toolboxes" from the 'src' directory ---
from src.data_utils import preprocess_mri_volumes
from src.model import nnunet_3d, InstanceNormalization
from src.analysis import calculate_volumes, generate_llm_report, configure_gemini

def main():
    """
    This is the main function that orchestrates our entire pipeline.
    """
    
    parser = argparse.ArgumentParser(description="Run full segmentation and analysis pipeline for a BraTS patient.")
    parser.add_argument("--patient_id", type=str, required=True, help="ID of the patient (e.g., BraTS-GoAT-00000)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the directory containing all patient folders.")
    parser.add_argument("--weights_path", type=str, required=True, help="Path to the trained model's .h5 weights file.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the output segmentation mask.")
    args = parser.parse_args()

    # --- 2. Configuration and Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    if not configure_gemini():
        print("Exiting: Gemini API could not be configured.")
        return # Exit the script if the API key isn't set up

    patient_dir = os.path.join(args.data_dir, args.patient_id)
    if not os.path.isdir(patient_dir):
        print(f"Error: Patient directory not found at {patient_dir}")
        return

    # --- 3. Load Model ---
    print("🧠 Building model and loading weights...")
    # We must use a 'custom_object_scope' because our model contains a custom layer
    # (InstanceNormalization) that TensorFlow doesn't know about by default.
    with tf.keras.utils.custom_object_scope({'InstanceNormalization': InstanceNormalization}):
        model = nnunet_3d(dropout_rate=0.0) # Dropout is disabled for inference
        model.load_weights(args.weights_path)
    print("✅ Model is ready.")

    # --- 4. Load and Preprocess Patient Data ---
    print(f"📂 Loading and preprocessing data for patient {args.patient_id}...")
    try:
        # We call our toolbox function from src/data_utils.py
        input_tensor, original_affine = preprocess_mri_volumes(patient_dir, args.patient_id, target_shape=(128, 128, 128))
        input_tensor_batch = np.expand_dims(input_tensor, axis=0)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please ensure all NIfTI files for the patient are present.")
        return

    # --- 5. Run Inference ---
    print("🚀 Running inference to get segmentation mask...")
    # The model outputs a list of 4 tensors (due to deep supervision). We only need the first, full-resolution one.
    pred_probs = model.predict(input_tensor_batch)[0][0] 
    # Convert the probability map (e.g., [0.1, 0.8, 0.1]) into a single class label (e.g., 1) for each voxel
    pred_labels = np.argmax(pred_probs, axis=-1)
    print("✅ Segmentation complete.")

    # --- 6. Save the Predicted Mask ---
    mask_save_path = os.path.join(args.output_dir, f"{args.patient_id}_pred_mask.nii.gz")
    nib.save(nib.Nifti1Image(pred_labels.astype(np.uint8), original_affine), mask_save_path)
    print(f"💾 Segmentation mask saved to: {mask_save_path}")

    # --- 7. Perform Analysis ---
    print("📏 Calculating tumor volumes...")
    # We call our toolbox function from src/analysis.py
    volumes = calculate_volumes(pred_labels, original_affine)
    print(f"   - Calculated Volumes (cm³): ET={volumes['ET']:.2f}, TC={volumes['TC']:.2f}, WT={volumes['WT']:.2f}")

    # --- 8. Generate Final Report ---
    print("\n📝 Calling Gemini to generate the report...")
    # We call our final toolbox function from src/analysis.py
    final_report = generate_llm_report(volumes, args.patient_id)
    
    # --- 9. Display the Final Output ---
    print("\n" + "="*50)
    print("          FINAL AI-GENERATED REPORT")
    print("="*50)
    print(final_report)
    print("="*50)

# This is a standard Python convention. It means: "only run the main() function
# if this file is executed directly from the command line."
if __name__ == '__main__':
    main()

