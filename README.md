# Enhancing CNN Classification Robustness to Low-Light and Occlusion

This repository contains the source code, datasets, and experimental evaluation for the project **"Enhancing CNN Classification Robustness to Low-Light and Occlusion with Digital Image Processing vs. Domain Adaptation."**

This project quantifies the degradation of ResNet-50 performance under environmental distortions (Low-Light and Water Occlusion) and compares two recovery strategies:
1.  **Data-Centric:** Restoring images using DIP algorithms (CLAHE, IAGC, Inpainting).
2.  **Model-Centric:** Fine-tuning the model using Domain Adaptation.

## Repository Structure
```
.
├── low_light/                        # Low-Light Track (Strategy 1 & 2)
│   ├── create_low_light_images.py    # Generates synthetic low-light dataset
│   ├── ablation_clahe.py             # Runs CLAHE ablation study
│   ├── ablation_iagc.py              # Runs IAGC ablation study
│   ├── evaluate.py                   # Main evaluation script for Low-Light
│   └── model_centric_low_light.ipynb # Fine-tuning workflow (Jupyter)
│
├── water_occlusion/                  # Occlusion Track (Strategy 1 & 2)
│   ├── create_water_occlusion_with_masks.py
│   ├── preprocess_images.py          # Preprocess Smartphone camera images
│   ├── ablation_inpainting_with_masks.py        # Compares Telea vs. Navier-Stokes, run ablation study
│   ├── apply_exemplar_inpainting.py        # Apply exemplar inpainting
│   ├── apply_diffusion_inpainting.py        # Apply diffusion inpainting
│   ├── apply_exemplar_inpainting_real_images.py        # Apply exemplar inpainting on phone-captured images
│   ├── evaluate_water_occlusion_final.py # Main evaluation script for Occlusion
│   └── model_centric_water_occlusion.ipynb # Fine-tuning workflow (Jupyter)
│
├── Imagenet/                         # Source Clean Data (Subset of ImageNet)
├── Distorted_Images/                 # Generated Low-Light & Occluded Images
├── Processed_Images/                 # Output of DIP Algorithms (Restored)
├── Manual_Images/                    # Raw Real-World Data Collection
│   ├── Low_Light/
│   └── Water Occlusion/
│
├── Baseline.ipynb                    # Evaluation of Clean Baseline (ResNet-50)
├── resnet50_clean_baseline.pth       # Pre-trained Baseline Weights
├── Final_Report.pdf
├── Project_Proposal.pdf
└── README.md
```
## Prerequisites

- Python 3.8+
- PyTorch (Torchvision)
- OpenCV (opencv-python)
- NumPy, Matplotlib

## Usage Instructions
### Data Generation
Run the generation scripts to apply physics-based degradation models (Gamma/Noise for Low-Light, Refraction for Occlusion) to the source images in Imagenet/.

```
python low_light/create_low_light_images.py
python water_occlusion/create_water_occlusion_with_masks.py
```

Outputs are saved to `Distorted_Images/`

## Strategy 1: Data-Centric Evaluation (DIP)
Run the ablation studies to test image restoration algorithms.

### Low-Light (CLAHE & IAGC):

```
python low_light/ablation_clahe.py
python low_light/ablation_iagc.py
```

### Water Occlusion (Inpainting):
```
python water_occlusion/ablation_inpainting_with_masks.py
```

Apply inpainting to the images-
```
python water_occlusion/apply_exemplar_inpainting.py
python water_occlusion/apply_exemplar_inpainting_real_images.py
python water_occlusion/apply_diffusion_inpainting.py

```

Restored images are saved to `Processed_Images/`

## Strategy 2: Model-Centric Evaluation (Fine-Tuning)
Open the Jupyter Notebooks to perform Transfer Learning on the mixed datasets.

**Low-Light:** `low_light/model_centric_low_light.ipynb`

**Occlusion:** `water_occlusion/model_centric_water_occlusion.ipynb`

These notebooks handle:

1. Loading the pre-trained ResNet-50 (Layers 1-4 Frozen).
2. Training the classifier head on the mixed dataset.
3. Saving the weights (e.g., `resnet50_finetuned_water.pth`).

## Final Evaluation
Run the final evaluation scripts to compare Baseline vs. DIP vs. Fine-Tuned Model.

```
python low_light/evaluate.py
python water_occlusion/evaluate_water_occlusion_final.py
```

## Authors
**Hariram Ramakrishnan** - Occlusion Track (Simulation, Inpainting, Data Collection)

**Shail Vaidya** - Low-Light Track (Noise Modeling, CLAHE/IAGC, Fine-Tuning)
