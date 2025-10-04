# An Optimized Deep Learning-Based Structure-from-Motion Pipeline for Efficient Glacier 3D Reconstruction

**Authors:** SiCheng Zhao¹, Edward Lai², Soumyabrata Dev¹  \
**Affiliations:**  
¹ University College Dublin (UCD), Ireland  
² Yale University, United States \
**Supervisor:** Prof. Soumyabrata Dev

---

## Abstract
Glacier three-dimensional (3D) reconstruction remains challenging for traditional Structure-from-Motion (SfM) pipelines under low-texture and reflective conditions. We propose an optimized deep learning-based pipeline using **SuperPoint** and **LightGlue**, accelerated with **ONNX Runtime**. Experiments on UAV glacier imagery show sub-pixel reprojection error, denser point clouds, and reduced GPU memory usage compared to **COLMAP** and **PyTorch-native** baselines. This highlights the efficiency and scalability of deep learning approaches for large-scale glacier reconstruction.

## Prerequisites

All experiments and evaluations in this study were conducted under the following testing environment:

- **Operating System:** Ubuntu 24.04 LTS  
- **Python:** 3.12(in conda or miniconda)
- **CUDA Toolkit:** 12.8
- **PyTorch:** 2.3.1 (with CUDA support)  
- **ONNX Runtime:** 1.17.0  
- **COLMAP:** 3.8 (installed from source)  

Hardware configuration:
- **GPU:** NVIDIA RTX 4090 (24 GB VRAM)  
- **CPU:** AMD Ryzen 9 7950X  
- **Memory:** 48 GB RAM

### Environment Setup
```bash
# download source code
git clone --recursive https://github.com/AppleJack9050/glacier_pipeline.git
cd glacier_pipeline

# install environment
conda create -n env_name python=3.12 ## choose environment name you like
conda activate env_name
pip install -r requirements.txt
conda activate env_name
```
## Data
You could download data from this link https://tls.unavco.org/data/B-425/PS06/SV02/projectfiles/2016-11-28_Howchin-AlphLake_Imagery-Files.beh.tar.gz

## Usage

The pipeline can be executed in two modes:

### 1. Standard Execution
Run the optimized SfM pipeline on your dataset:
```bash
python sp_lightglue_sfm_colmap_clean.py --project /path/to/Proj1
```
Replace the path with your project directory and ensure it contains a subfolder named images.

### 2. Performance Monitoring Mode
To record runtime statistics (including GPU usage, GPU memory, and execution time), run:
```bash
./monitor_runner.sh -o run.log -c "python sp_lightglue_sfm_colmap_clean.py --project /path/to/Proj1"
```

## License
This project is licensed under the **MIT License © 2025 UCD THEIA LAB**.