<img width="8215" height="9813" alt="图1 SARBench-VLM数据集架构V2 0" src="https://github.com/user-attachments/assets/4a2cf033-3e04-4bfc-bd57-1a86ff001cad" /># SAREval: A Multi-Dimensional Benchmark for SAR Image Understanding VLMs

<div align="center">
  <p>🔭 A Comprehensive Benchmark for Evaluating Vision-Language Models on Synthetic Aperture Radar (SAR) Imagery</p>
  <a href="https://github.com/Dilys2022/SAREval"><img src="https://img.shields.io/badge/GitHub-SAREval-blue.svg" alt="GitHub"></a>
  <a href="https://evalscope.readthedocs.io"><img src="https://img.shields.io/badge/EvalScope-Compatible-green.svg" alt="EvalScope Compatible"></a>
</div>

## 📖 Project Overview
SAREval is the first dedicated benchmark for evaluating Visual Language Models (VLMs) on Synthetic Aperture Radar (SAR) image understanding. It addresses the mismatch of existing remote sensing benchmarks with SAR’s unique mechanisms (scattering, polarization, geometric distortion) by focusing on three core capabilities: **Perception**, **Reasoning**, and **Robustness**.  
### Data Classification
| Core Dimension | Subtask Category | Sample Size | Key Use Case |
|----------------|------------------|-------------|--------------|
| Perception     | Image-level/Object-level/Pixel-level Tasks | 11,309 | Feature extraction, object localization |
| Reasoning      | Geometric/Attribute/Imaging Parameter Tasks | 2,643 | Physical attribute derivation |
| Robustness     | Speckle/Clutter Interference Tasks | 998 | Anti-interference evaluation |

### Key Features
- **20+ Diverse Tasks**: Covers core capabilities including perception (e.g., aircraft classification, ship detection), reasoning (e.g., imaging parameter estimation), and robustness (e.g., noise resistance).
- **High-Quality Annotations**: Expert-verified labels with multiple prompt templates per sample, ensuring reliable evaluation.
- **Easy-to-Use**: Well-documented folder structure, clear annotation formats, and ready-to-run evaluation scripts via Evalscope framework.
- **Open Access**: Full dataset (images + annotations) available for download, supporting reproducible research.

## 📁 Dataset Structure
Upon downloading and unzipping, the SAREval dataset follows a **task-centric hierarchical structure** for intuitive navigation:

```
SAREval/
├── images/                          # Directory for all SAR image files (JPG/PNG)
│   ├── AircraftClassificationDetection/  # Aircraft classification task images
│   ├── AircraftCounting/                 # Aircraft counting task images
│   ├── BridgeCounting/                   # Bridge counting task images
│   └── ... (20+ task-specific subfolders)
├── LMUData/                         # Unified annotation summaries (TSV format)
│   ├── AircraftClassificationDetection.tsv
│   ├── AircraftCounting.tsv
│   └── ... (one TSV per task)
├── Aircraft_Classification_Detection.json  # Task-specific JSON annotations
├── Aircraft_Counting.json                  # Task-specific JSON annotations
├── Bridge_Counting.json                    # Task-specific JSON annotations
└── README.txt                       # Quick start guide
```

- **Image Storage**: Each task has a dedicated subfolder under `images/` containing matching image files.
- **Annotation Mapping**: JSON files directly correspond to task folders (detailed annotations), while TSV files in `LMUData/` provide condensed summaries.

## 📋 Annotation Format
Each task-specific JSON file (e.g., `Aircraft_Classification_Detection.json`) uses a structured format with human-readable fields. Below is a complete example:

```json
{
    "image_path": "aircraft_1.png",        # Relative path to the image (matches task folder)
    "ground_truth": "Airbus_A220",         # Ground-truth label (target category/value)
    "ground_truth_option": "C",            # Correct option letter (for multiple-choice tasks)
    "options_list": [                      # Option list (multiple-choice tasks only)
        "Boeing737",
        "Other",
        "Airbus_A220",
        "Boeing747"
    ],
    "options": "A. Boeing737   B. Other   C. Airbus_A220   D. Boeing747",  # Formatted options
    "prompts": [                           # Multiple prompt templates for diverse inputs
        "What type of aircraft is visible in this image?",
        "Which model does the identified aircraft belong to?"
    ],
    "task": "Aircraft Type Classification",# Task name
    "image_name": "aircraft_1.png",        # Image filename (consistent with image_path)
    "question_id": 0,                      # Unique question ID
    "cls_description": "High Difficulty"   # Sample difficulty (High/Medium/Low)
}
```

### Field Explanations
| Field                | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `image_path`         | Relative path to the image (maps to `images/[TaskFolder]/[image_path]`).    |
| `ground_truth`       | Task-specific truth value (category label, coordinates, or numerical value).|
| `ground_truth_option`| Correct option letter (A/B/C/D) for multiple-choice tasks.                  |
| `prompts`            | Multiple question templates to test model generalization.                   |
| `cls_description`    | Difficulty label for stratified evaluation.                                 |

## 📥 Data Download
The full SAREval dataset (images + annotations) is available via Baidu Net Disk:

- **Dataset Package**: [SAREval_Full_Dataset.zip](https://pan.baidu.com/s/119HoKeb8135KQBwFuY-9yQ?pwd=4jwu)
- **Extraction Code**: `4jwu`

## 🚀 Quick Start
### 1. Environment Setup
SAREval uses the **Evalscope framework** for seamless model evaluation. Install dependencies first:

```bash
# Python 3.10 is recommended
conda create -n evalscope python=3.10
conda activate evalscope
pip install evalscope
pip install 'evalscope[vlmeval]'
```

### 2. Model Deployment
Load and preprocess data for model input (PyTorch-compatible):

```bash
# Install vLLM
pip install vllm -U
# Deploy Model Service (llava-1.5-7b-hf)
VLLM_USE_MODELSCOPE=True CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model llava-hf/llava-1.5-7b-hf --port 8000 --trust-remote-code --max_model_len 4096 --served-model-name llava-1.5-7b-hf
# Deploy Model Service (deepseek-vl2-tiny)
# VLLM_USE_MODELSCOPE=True  CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model deepseek-ai/deepseek-vl2-tiny --port 8000 --trust-remote-code --max_model_len 4096 --served-model-name deepseek-vl2-tiny --hf-overrides '{"architectures": ["DeepseekVLV2ForCausalLM"]}'
# ...
```

### 3. Model Evaluation with Evalscope
Evaluate VLMs using a YAML configuration file (e.g., `eval_config.yaml`):

```yaml
eval_backend: VLMEvalKit
eval_config:
  model: 
    - type: llava-1.5-7b-hf  # Support LLaVA, Qwen-VL, InternVL, etc.
      name: CustomAPIModel
      api_base: http://localhost:8000/v1/chat/completions  # Local model API
      key: EMPTY  # Set to "EMPTY" for local deployment
      temperature: 0.0  # Deterministic results
      img_size: -1  # Auto-adapt image size
  data:
    - AircraftClassificationDetection  # Target task name
  mode: all  # Evaluate all samples
  limit: 10  # Optional: limit sample count (remove for full evaluation)
  reuse: false
  work_dir: outputs  # Save results here
  nproc: 1
```

Run evaluation with:
```bash
evalscope eval --config eval_config.yaml
```

## 📊 Evaluation Metrics
SAREval uses task-specific metrics for comprehensive assessment:
- **Multiple-Choice**: Accuracy
- **Visual Grounding**:  IoU@0.25, IoU@0.5  
- **Description Tasks (e.g., scene captioning)**: BLEU, ROUGE-L, BERTScore，LLM-as-Judgement


## 🎯 Contributing
We welcome contributions to SAREval! Feel free to submit PRs for bug fixes, new tasks, or improved scripts. For major changes, please open an issue first to discuss your ideas.
