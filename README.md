
# EndoVLM: An Endoscopy Vision-Language Pre-training Model via Anatomy-Guided Sparsity and Progressive Alignment


## 📖 Introduction

**EndoVLM** is the first vision-language foundation model pre-trained on a massive dataset of 348K unordered endoscopic image sets paired with comprehensive gastrointestinal clinical reports. By resolving the profound semantic gap between highly redundant visual frames and structured clinical narratives, EndoVLM provides a highly scalable and robust foundation for next-generation AI-assisted endoscopy.

<p align="center">
  <img src="figs/method.png" width="600">
</p>

## 🌟 Key Highlights

* **Anatomy-Guided Sparse Pooling (AGSP)**: Efficiently distills semantically salient frames from noisy, unordered image sets without relying on unreliable temporal priors.
* **Progressive Semantic-Aware Alignment (PSAA)**: Explicitly encodes clinical taxonomy into soft targets, enabling a robust transition from global patient-level matching to fine-grained anatomical and pathological alignment.
* **Semantic-Concentrated Masked Autoencoder (SC-MAE)**: Preserves fine-grained diagnostic textures and geometric precision as a complementary regularization.
* **Exceptional Zero-Shot Generalization**: Achieves flawless transferability (~100% AUC) in Upper-GI Anatomy Recognition and significantly outperforms existing baselines (e.g., BiomedCLIP) in challenging dynamic video analysis.
<p align="center">
  <img src="figs/zero_shot.png" width="400">
</p>
<!-- <img src="figs/zero_shot.png" width="40%"> <img src="figs/retrieval.png" width="55%"> -->
<!-- <p align="center">
  <img src="figs/retrieval.png" width="800">
</p> -->

* **Downstream Experiments Across Various Tasks:**
<p align="center">
  <img src="figs/comparison.jpg" width="600">
</p>


---

## 📂 Repository Structure

The codebase is organized as follows to facilitate reproducibility:

* `dataset/`: Data loaders for handling unordered image sets and raw clinical reports.
* `dinov3/`: Vision backbone architecture and configurations.
* `util/`: Utility functions, including metrics, logging, and distributed training tools.
* `endovlm.py`: Core architecture implementation (AGSP, PSAA, SC-MAE).
* `engine_pretrain.py`: Training, evaluation, and logging loops for the pre-training phase.
* `main_pretrain.py`: Main entry script for distributed pre-training.
* `main_pretrain.sh`: Shell script containing hyperparameters to launch the training pipeline.
* `llm_extraction/`: LLM-based annotation pipeline for extracting structured anatomy/pathology labels from raw clinical reports.

---

## 🛠️ Installation

1. Clone this anonymous repository:
```bash
git clone <anonymous_repo_url>
cd EndoVLM

```

2. Create a conda environment and install dependencies:


```bash
conda create -n endovlm python=3.11
conda activate endovlm

# Install core dependencies
pip install torch transformers open-clip-torch timm

```

---

## 🗄️ Data Preparation

EndoVLM is pre-trained on unordered image sets and corresponding clinical reports. To reproduce our pipeline, please organize your dataset in the following structure, or adjust the parsers in `dataset/` accordingly:

```text
data/
├── images/
│   ├── patient_000000001/
│   │   ├── image_0001.jpg
│   │   ├── image_0002.jpg
│   │   └── ...
│   └── ...
└── reports/
    └── clinical_reports.json  # Contains patient-level reports and taxonomy labels

```

*(Note: Due to patient privacy and ethical guidelines, the 348K pre-training dataset cannot be released at this stage. We provide toy examples in the `data/` folder for pipeline verification.)*

---

## 🚀 Pre-training

We provide a streamlined shell script to launch the distributed pre-training process. Hyperparameters (e.g., learning rate, batch size, temperature, top-K for AGSP) can be modified directly inside the script.

To start pre-training:

```bash
bash main_pretrain.sh

```

The script calls `main_pretrain.py`, which initializes the model from `endovlm.py` and executes the optimization loop defined in `engine_pretrain.py`.

---

## 🏷️ LLM-based Data Annotation (`llm_extraction/`)

The structured anatomy and pathology labels used for pre-training are generated via LLM-based annotation. The script `llm_extraction/extract_anato_patho.py` contains all four annotation tasks with their full prompts (in English, assuming English-language input reports):

### Quick Start

```bash
# Install dependencies
pip install openai tqdm

# Set your DashScope (Alibaba Cloud) API key
export DASHSCOPE_API_KEY="your-api-key"

# Run
python llm_extraction/extract_anatomy_pathology.py \
    --base_url your_api_base_url \
    --input data/reports.jsonl \
    --output data/reports_annotated.jsonl

```

---

## 📈 Evaluation & Inference

To extract visual-linguistic features or reproduce the zero-shot evaluation, you can initialize the model and load our pre-trained weights (weights will be made publicly available upon acceptance).

Example inference snippet:

```python
import torch
from endovlm import build_endovlm

# Build model and load pre-trained weights
model = build_endovlm(model_type='vit_base_patch16', pretrain_path='xxx')
model.eval()

# Extract representations
image_features,_,_ = model.forward_encoder(image_set)
text_features = model.forward_text(["This is an image of Esophagus."])

```
