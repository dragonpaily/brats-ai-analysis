End-to-End Brain Tumor Analysis Pipeline
This repository contains the code for a complete pipeline for brain tumor analysis from MRI scans, developed based on the BraTS 2024 challenge dataset. The project uses a high-performance 3D U-Net for semantic segmentation and integrates with a Large Language Model (Gemini 1.5 Flash) to generate descriptive reports from the segmentation results.

This project is structured as a modular, reusable, and professional research codebase, suitable for extension and reproduction.

📊 Performance
The core segmentation model was trained and validated on the BraTS 2024 dataset. It achieved the following mean Dice Scores on a hold-out test set of 203 patients:

Tumor Region                   Mean Dice Score

Enhancing Tumor (ET)            0.7743



Tumor Core (TC)                 0.8443



Whole Tumor (WT)                0.8813



Average (ET, TC, WT)            0.8333



✨ Features
High-Performance 3D U-Net: A robust 3D neural network architecture with residual connections and deep supervision for accurate segmentation.

Advanced Loss Function: A custom combo_loss that combines Focal Tversky and Dice losses to effectively handle the severe class imbalance in tumor data.

Modular & Reproducible Codebase: The code is structured into separate modules for data utilities, model architecture, and analysis, following best practices for research software engineering.

LLM-Powered Reporting: Integrates with the Google Gemini API to automatically generate structured, descriptive reports from the calculated tumor volumes, providing a proof-of-concept for AI-driven interpretation.

🚀 Getting Started
Prerequisites
Python 3.9+

An NVIDIA GPU with CUDA and CuDNN installed is recommended for model inference.

A Google Gemini API Key.

Installation
Clone the repository:

git clone https://github.com/dragonpaily/brats-ai-analysis.git 
cd brats-ai-analysis


Install dependencies:

pip install -r requirements.txt


Download Model Weights:
Download the trained model weights (best_model.weights.h5) from 

https://drive.google.com/file/d/1_FfzkPQOW53Q-hZeU1PJGQONm70vTz7G/view?usp=sharing
 and place them in a checkpoints/ folder in the root directory.

Set up Gemini API Key:
For the report generation to work, you need to provide your API key as an environment variable:

export GEMINI_API_KEY="your_api_key_here"


Usage
To run the full analysis pipeline on a single patient's data, use the predict.py script. You must provide the path to the patient's data folder and the patient's ID.

python scripts/predict.py \
    --patient_data_dir /path/to/your/BraTS_data/ \
    --patient_id BraTS-GoAT-00000 \
    --model_weights /path/to/your/checkpoints/best_model.weights.h5 \
    --output_dir /path/to/save/results/

