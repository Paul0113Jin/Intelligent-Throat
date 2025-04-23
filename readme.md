# Wearable intelligent throat enables natural speech in stroke patients with dysarthria

This repo contains the source code of [paper](https://arxiv.org/abs/2411.18266).

## 1. System Requirements

### Software Dependencies

* **Python:** 3.12 (or compatible version, e.g., 3.9, 3.11)
* **Cuda:** 12.4
* **Core ML:** PyTorch, Torchvision, Torchaudio, PyTorch Lightning, TorchMetrics
* **Data Handling:** Pandas, NumPy, PyYAML, Scikit-learn
* **Plotting:** Matplotlib, Seaborn
* **Environment Management:** Conda (for creating the base environment)

A detailed list of Python packages for installation via `pip` is provided in `requirements.txt`.

### Operating System
The code has been primarily tested on:
* **Linux** Ubuntu 22.04

### Hardware Requirements
* **GPU:** An NVIDIA GPU compatible with CUDA is **highly recommended**
    * CUDA Version: tested on 12.4 (make sure installed torch version is compatiable with Cuda).
    * GPU Memory: tested on NVIDIA RTX4090 with 24GB VRAM each

## 2. Installation Guide

### Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Create and Activate Conda Environment:**
    ```bash
    # Example: Create environment
    conda create --name wearable_dl_env python=3.12
    conda activate wearable_dl_env
    ```
3.  **Install Dependencies using pip:** Install the remaining required packages using the provided `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

### Typical Install Time
* Assuming the Conda environment with Python and potentially PyTorch is already set up, installing the remaining packages via `pip` typically takes less than **5 minutes**, depending on the internet speed.

## 3. Run

This instruction shows how to modify config.yaml file to run model at each stage (pretrain + finetune + distillation + test)

### Prerequisites
* Installation completed (Step 2).
* Preprocessed data is saved as .npy files in './processed_data' folder, or raw data is saved as .csv files in './dataset' folder.
* Processed data should follow "stage_X.npy" or "{stage}_Y.npy" format. Raw data should follow "{stage}_raw.csv" format, where the last column of it should be the label column.

### Instructions
1.  **Pretrain stage**
    * Set `stage: pretrain`.
    * Set `num_blocks` in `model` to `[3, 4, 23, 3]` -> ResNet 101
    * Run the main.py:
    ```bash
    python main.py
    ```

2.  **Finetune stage**
    * Set `stage: finetune`.
    * Set `pretrained_ckpt_path` in `finetune` to the pretrained model checkpoint path from pretrain stage
    * Run the main.py:
    ```bash
    python main.py
    ```

3.  **Distillation stage**
    * Set `stage: distillation`.
    * Set `teacher_ckpt_path` in `distill` to the finetuneed model checkpoint path from finetune stage
    * Set `num_blocks` in `distill/student_model_config` to `[2, 2, 2, 2]` -> ResNet 18
    * Run the main.py:
    ```bash
    python main.py
    ```

4.  **Test stage**
    * Set `stage: test`.
    * Set `num_blocks` in `model` to `[2, 2, 2, 2]` -> ResNet 101
    * Set `ckpt_path` in `test` to the distilled model checkpoint path from distillation stage. This path could be any stage's checkpoint path as long as the model arch matches (e.g. num_blocks)
    * Run the main.py:
    ```bash
    python main.py
    ```

### Expected Output
* An example of test stage result can be found in `./demo`.

### Expected Run Time
* Varies each stage. The maximum run time for each stage should not exceeding 10 minutes if running with DDP on 4 GPUs.


