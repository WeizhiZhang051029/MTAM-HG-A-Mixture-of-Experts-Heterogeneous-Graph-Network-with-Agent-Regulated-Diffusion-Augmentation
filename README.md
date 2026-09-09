# 🔩 MTAM-HG: A Mixture-of-Experts Heterogeneous Graph Network with Agent-Regulated Diffusion Augmentation for Strip Yield Strength Prediction

<p align="center">
  <b>Mechanism-Prior Diffusion Augmentation · CBTG-Agent · Mixture-of-experts · Heterogeneous Graph Network </b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/MTAM--HG-Yield%20Strength%20Prediction-blue">
  <img src="https://img.shields.io/badge/MP--TabDiff-Mechanism--Prior%20Augmentation-green">
  <img src="https://img.shields.io/badge/CBTG--Agent-Dynamic%20Sample%20Regulation-purple">
  <img src="https://img.shields.io/badge/MoE--IPOHGN-Heterogeneous%20Graph%20MoE-orange">
</p>

<p align="center">
  <a href="https://github.com/WeizhiZhang051029/MTAM-HG-A-Mixture-of-Experts-Heterogeneous-Graph-Network-with-Agent-Regulated-Diffusion-Augmentation">
    <b>Project Page</b>
  </a>
    |  
  <a href="#citation">
    <b>Paper</b>
  </a>
</p>

## 📌 Overview

This repository provides the official implementation of **MTAM-HG: A Mixture-of-Experts Heterogeneous Graph Network with Agent-Regulated Diffusion Augmentation for Strip Yield Strength Prediction**.

<p align="center">
  <img src="images/framework.jpg" width="100%">
</p>

<p align="center">
  <em>Overall framework of MTAM-HG for data augmentation and strip yield strength prediction in continuous annealing production lines.</em>
</p>

Yield strength is a key quality indicator in continuous annealing production lines (CAPLs), but its accurate prediction remains challenging when production records are limited, strength distributions are long-tailed, and process variables are strongly coupled. Existing data-driven approaches also rarely incorporate process-mechanism constraints or explicitly account for differences among operating conditions.

To address these challenges, **MTAM-HG** integrates mechanism-prior diffusion augmentation, feedback-driven sample regulation, and heterogeneous graph mixture-of-experts prediction within a unified framework.

The MTAM-HG framework comprises two main modules:

* **Data augmentation:** MP-TabDiff embeds furnace-temperature trajectories, production-window constraints, and an empirical yield-strength prior into tabular diffusion to generate process-consistent samples. CBTG-Agent dynamically perceives operating conditions and training states, and regulates synthetic samples through iterative decision-making, feedback, selection, and reweighting.

* **MoE-IPOHGN prediction:** MoE-IPOHGN represents CAPL variables within an implicit process-order heterogeneous graph and captures condition-dependent variable interactions. A Hard Sparse Gate (HSG) adaptively activates specialized experts for different operating conditions, enabling sample-dependent yield-strength prediction.
  
Experiments on real CAPL production data demonstrate that MTAM-HG improves prediction accuracy and cross-condition stability over competitive baselines, while maintaining reliable performance under data scarcity.

---

## 🔥 Highlights

* Yield strength prediction in the continuous annealing line is investigated.
* A novel MTAM-HG framework is proposed for prediction under data scarcity.
* MP-TabDiff and CBTG-Agent are designed for mechanism-prior diffusion augmentation.
* A MoE heterogeneous graph network (MoE-IPOHGN) predicts yield strength.
* Experiments on real industrial data verify the effectiveness of MTAM-HG.

---

## 🧩 Framework

The training workflow of MTAM-HG is organized as follows:

```text
Real CAPL production data
        |
        v
Training / validation / test partition
        |
        v
MP-TabDiff training on real training data
        |
        v
Mechanism-prior synthetic sample generation
        |
        v
CBTG-Agent dynamic sample regulation
        |
        v
Selected and reweighted synthetic samples
        |
        v
MoE-IPOHGN synthetic-data pretraining
        |
        v
Real-domain LoRA calibration
        |
        v
Validation-based model selection
        |
        v
Strip yield strength prediction
```

The test set is isolated throughout model development and is used only for final evaluation.

---

## 📊 Experimental Protocol

Experiments use **600 real CAPL production records** with **21 process variables** and yield strength as the prediction target. The raw industrial data cannot be publicly released due to confidentiality.

For each of **10 independent runs**, the data are stratified by yield strength and split into training/validation/test sets at **70%/15%/15%**, with the run seed controlling both data partitioning and model initialization.

All preprocessing, clustering, and synthetic-data generation are fitted only on the corresponding real training set. CBTG-Agent uses training-set feedback, while validation is used for model selection and early stopping; the test set is reserved for final evaluation.

Results are reported as **mean ± standard deviation**. Statistical significance is assessed using a **two-sided paired Wilcoxon signed-rank test with Holm correction** (\(p_{\mathrm{adj}}<0.05\)).

---

## ⚙️ Configuration

The manuscript configuration is provided in:

```text
configs/mtam_hg.yaml
```

Representative settings include:

| Component                    | Setting         |
| ---------------------------- | --------------- |
| Real-data split              | 70% / 15% / 15% |
| Independent runs             | 10              |
| Operating-condition clusters | 5               |
| Synthetic candidates         | 5,000           |
| MP-TabDiff diffusion steps   | 50              |
| MP-TabDiff fine-tuning steps | 500             |
| CBTG-Agent refresh interval  | 5 epochs        |
| Synthetic retention ratio    | 60%             |
| Number of HG experts         | 4               |
| Active experts               | Top-2           |
| Synthetic pretraining        | 100 epochs      |
| Real-domain calibration      | Up to 50 epochs; early-stopping patience: 5 |
| Optimizer                    | AdamW           |

Detailed architecture, optimization, routing, Agent, and LoRA parameters are specified in the configuration file.

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/WeizhiZhang051029/MTAM-HG-A-Mixture-of-Experts-Heterogeneous-Graph-Network-with-Agent-Regulated-Diffusion-Augmentation.git
cd MTAM-HG-A-Mixture-of-Experts-Heterogeneous-Graph-Network-with-Agent-Regulated-Diffusion-Augmentation
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the package:

```bash
python -m pip install --upgrade pip
pip install -e .
```

The default main experiment requires **Linux and an NVIDIA GPU with CUDA support**. Install a PyTorch build compatible with the CUDA environment. The environment-activation commands above are platform-specific examples; they do not imply that the default training configuration supports Windows or macOS.

---

## 🚀 Running

### Complete MTAM-HG Experiment

To reproduce the complete ten-run experiment:

```bash
python run_experiment.py \
  --data_path data/CAPL.xlsx \
  --config configs/mtam_hg.yaml \
  --seeds 42 43 44 45 46 47 48 49 50 51 \
  --tabdiff_num_samples 5000
```

The pipeline sequentially:

1. partitions and preprocesses the real CAPL data;
2. trains MP-TabDiff on the training partition;
3. generates mechanism-constrained candidate samples;
4. regulates synthetic samples using CBTG-Agent;
5. pretrains MoE-IPOHGN on the selected synthetic samples;
6. performs real-domain calibration;
7. selects the model according to validation performance;
8. evaluates the final model on the held-out test set.

### Check the Experiment Configuration

```bash
python run_experiment.py --dry_run
```

This command checks the experiment arguments and prints the per-run commands without training the model.

---

## 📁 Repository Structure

```text
MTAM-HG-A-Mixture-of-Experts-Heterogeneous-Graph-Network-with-Agent-Regulated-Diffusion-Augmentation/
├── configs/
│   └── mtam_hg.yaml
├── generation/
│   ├── __init__.py
│   ├── prepare.py
│   ├── train.py
│   ├── sample.py
│   ├── postprocess.py
│   └── tabdiff.py
├── images/
│   ├── framework.jpg
│   ├── mp-tabdiff.jpg
│   ├── cbtg-agent.jpg
│   └── moe-ipohgn.jpg
├── models/
│   ├── __init__.py
│   ├── graph_structure.py
│   ├── ipohgn.py
│   ├── mr_lora.py
│   └── mtam_hg.py
├── third_party/
│   └── TabDiff/
│       ├── LICENSE
│       ├── main.py
│       ├── process_dataset.py
│       ├── utils_train.py
│       ├── tabdiff.yaml
│       ├── src/
│       └── tabdiff/
├── training/
│   ├── __init__.py
│   ├── batch_metadata.py
│   ├── cbtg.py
│   ├── clusters.py
│   ├── compilation.py
│   ├── evaluation.py
│   ├── quality_agent.py
│   ├── step_logging.py
│   └── synthetic_types.py
├── utils/
│   ├── __init__.py
│   ├── graph.py
│   ├── logger.py
│   ├── seed.py
│   └── tensor_logging.py
├── .gitattributes
├── .gitignore
├── CITATION.cff
├── config.py
├── config_loader.py
├── dataset.py
├── evaluate.py
├── losses.py
├── metrics.py
├── pipeline.py
├── protocol.py
├── protocol_integrity.py
├── run_experiment.py
├── train.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

The main components are organized as follows:

- `configs/mtam_hg.yaml`: configuration for the main MTAM-HG experiment.
- `run_experiment.py`: experiment execution and aggregation of results across runs.
- `generation/`: MP-TabDiff data preparation, training, sampling, and mechanism-based postprocessing.
- `models/`: heterogeneous graph construction, IPOHGN experts, mixture-of-experts routing, and MR-LoRA adaptation.
- `training/`: CBTG-Agent regulation, working-condition clustering, synthetic pretraining, and supporting training routines.
- `dataset.py`: data loading, partitioning, and preprocessing.
- `pipeline.py`, `train.py`, and `evaluate.py`: workflow orchestration, model training, and prediction evaluation.
- `config.py`, `config_loader.py`, and `protocol.py`: model settings, configuration loading, and experiment defaults.
- `losses.py` and `metrics.py`: training objectives and regression evaluation metrics.
- `protocol_integrity.py`: data-partition and synthetic-data provenance validation.
- `utils/`: graph utilities, random-seed initialization, and logging.
- `third_party/TabDiff/`: adapted TabDiff implementation with its original license.
---

## 📦 Outputs

Experiment outputs are written under:

```text
outputs/
```

Each independent run stores the corresponding:

* model checkpoints;
* prediction results;
* evaluation metrics;
* expert-routing statistics;
* CBTG-Agent sample-selection records;
* training logs;
* preprocessing and experiment metadata.

With the default configuration, the main experiment outputs are organized as follows (representative files shown):

```text
outputs/mtam_hg/
└── <experiment_timestamp>/
    ├── experiment_summary.json
    ├── seed_42/
    │   └── <run_timestamp>/
    │       ├── checkpoints/
    │       │   └── best_model.pth
    │       ├── logs/
    │       │   └── train_log.csv
    │       └── results/
    │           ├── metrics.json
    │           ├── predictions.csv
    │           └── ...
    ├── seed_43/
    │   └── ...
    └── ...
```

Per-run results and the mean and sample standard deviation across runs are stored in `experiment_summary.json`; the summary statistics are available under its `aggregate` field.

Generated checkpoints, synthetic samples, predictions, preprocessing statistics, and industrial data are excluded from version control.

---

## 📏 Evaluation Metrics

Yield strength prediction is evaluated using four regression metrics:

* **Root Mean Squared Error (RMSE)**
* **Mean Absolute Error (MAE)**
* **Mean Absolute Percentage Error (MAPE)**
* **Coefficient of Determination (R²)**

Lower RMSE, MAE, and MAPE values indicate smaller prediction errors, while a higher R² indicates stronger agreement between predicted and measured yield strength.

---

## 📰 News

* **July 2026** — MTAM-HG framework completed.
* **July 2026** — Manuscript completed and submitted.
* **August 2026** — Source code released.


---

## 🙏 Acknowledgements

This project builds upon **PyTorch**, **scikit-learn**, and the open-source **TabDiff** implementation.

We thank the open-source community for the tools and resources that support research in tabular diffusion modeling, heterogeneous graph learning, mixture-of-experts architectures, and parameter-efficient adaptation.

The original TabDiff copyright and license are provided in:

```text
third_party/TabDiff/LICENSE
```

---

## 📖 Citation

If you find this repository useful in your research, please consider citing our paper:

```bibtex
@article{zhang2026mtamhg,
  title   = {MTAM-HG: A Mixture-of-Experts Heterogeneous Graph Network with Agent-Regulated Diffusion Augmentation for Strip Yield Strength Prediction},
  journal = {Expert Systems with Applications},
  year    = {2026},
}
```

The citation information will be updated after the paper is officially published.

---

## 📬 Contact

For questions regarding the implementation, experimental configuration, or reproducibility of MTAM-HG, please open an issue in this repository.
