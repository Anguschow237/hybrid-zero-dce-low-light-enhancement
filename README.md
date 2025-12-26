# Low-Light Image Enhancement: U-Net vs. ZeroDCE Comparison

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Anguschow237/hybrid-zero-dce-low-light-enhancement/blob/main/hybrid_zero_dce_low_light_image_enhancement.ipynb)

## Overview
This repository contains a deep learning project for enhancing low-light images. It benchmarks supervised and unsupervised approaches using the LOL dataset, comparing a U-Net model with Charbonnier loss against the original ZeroDCE and two hybrid ZeroDCE variants. The hybrids integrate supervised signals to improve performance, demonstrating trade-offs in accuracy, stability, and perceptual quality.

Key contributions:
- Loss function ablation on U-Net (L1, SSIM, Combo, Charbonnier).
- Hybrid ZeroDCE models for bridging supervised and unsupervised paradigms.
- Quantitative (PSNR/SSIM) and qualitative evaluations.

Built with PyTorch.

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-image (for metrics)

Install via:
```
pip install torch torchvision numpy matplotlib scikit-image
```

## Dataset
The project uses the LOL (Low-Light) dataset: 485 training pairs and 15 test pairs of low/normal-light images. Download from [Kaggle](https://www.kaggle.com/datasets/soumikrakshit/lol-dataset).

## Usage
1. Clone the repo:
   ```
   git clone https://github.com/Anguschow237/hybrid-zero-dce-low-light-enhancement.git
   cd hybrid-zero-dce-low-light-enhancement
   ```

2. Open and run the notebook in Colab (link above) or locally with Jupyter.

3. Key sections:
   - Setup environment and load dataset.
   - Train/evaluate models (U-Net, ZeroDCE, hybrids).
   - View results (metrics, before/after images).

## Results
### Quantitative Metrics (Test Set Averages)
| Model                              | Avg PSNR | Avg SSIM |
|------------------------------------|----------|----------|
| U-Net + Charbonnier (Supervised)   | 19.53    | 0.769    |
| ZeroDCE (Unsupervised)             | 14.60    | 0.470    |
| Hybrid ZeroDCE (Dual Loss)         | 17.05    | 0.555    |
| Hybrid ZeroDCE (Charbonnier Only)  | 16.74    | 0.499    |

### Visual Examples
(Include side-by-side comparisons here—upload your before/after images to the repo and embed them, e.g.:)

![Example 1](path/to/result_image1.png)  
Input (low-light) | Enhanced (U-Net) | Enhanced (Hybrid Dual) | Ground Truth

(Repeat for more examples.)

Supervised U-Net excels in fidelity, while hybrids improve unsupervised baselines with better stability.

## References
- ZeroDCE: [Paper](https://arxiv.org/abs/2001.06826), [GitHub](https://github.com/Li-Chongyi/Zero-DCE)
- U-Net: [Paper](https://arxiv.org/abs/1505.04597)
- LOL Dataset: [Kaggle](https://www.kaggle.com/datasets/soumikrakshit/lol-dataset)

## License
[MIT License](LICENSE)

Built by Chow Tsz Hin, December 2025.
