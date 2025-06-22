# 📊 Graph-Classification

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch--Geometric-2.0%2B-green.svg)](https://pytorch-geometric.readthedocs.io/)

Graph-Classification is a modular and extendable implementation of node and edge classification tasks using Graph Neural Networks (GNNs), including **GCN**, **GraphSAGE**, and **GAT**. It utilizes the [OGB](https://ogb.stanford.edu/) benchmark datasets, especially `ogbn-products`, and is built with [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io/).

---

## 📁 Project Structure

```
Graph-Classification/
│
├── All Models/                 # All GNN model implementations
│   ├── gcn.py
│   ├── sage.py
│   └── gat.py
│
├── Node/                       # Node classification pipeline
│   ├── train_node.py
│   └── eval_node.py
│
├── Edge/                       # Edge classification pipeline
│   ├── train_edge.py
│   └── eval_edge.py
│
├── Graph_Prediction.ipynb      # Interactive Jupyter notebook
├── MyDoc_Report.pdf            # Detailed technical report
├── WhatToDo.pdf                # Task planning and roadmap
├── GCN Vs. SAGE in Edge Prd.png # Comparison chart (GCN vs SAGE)
├── Hardware-Limitation.png     # Hardware resource chart
└── README.md                   # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/M-Amin-Kiani/Graph-Classification.git
cd Graph-Classification
```

### 2. Install Dependencies

Make sure Python ≥ 3.8 is installed.

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-geometric ogb scikit-learn numpy tqdm
```

---

## 📦 Dataset

We use the [OGBN-Products](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products) dataset from the Open Graph Benchmark (OGB). It is automatically downloaded during the first run.

---

## 🚀 Usage

### ▶️ Node Classification

```bash
# Training
python Node/train_node.py --model gcn --epochs 100 --batch-size 1024

# Evaluation
python Node/eval_node.py --model gcn
```

### 🔁 Edge Classification

```bash
# Training
python Edge/train_edge.py --model sage --epochs 50 --batch-size 2048

# Evaluation
python Edge/eval_edge.py --model sage
```

### 📓 Jupyter Notebook

To experiment interactively:

```bash
jupyter notebook Graph_Prediction.ipynb
```

---

## 🔧 Hyperparameters

You can configure training via command-line arguments:

| Argument       | Description                         | Default   |
|----------------|-------------------------------------|-----------|
| `--model`      | Model type: `gcn`, `sage`, `gat`    | `gcn`     |
| `--epochs`     | Number of training epochs           | `100`     |
| `--batch-size` | Batch size for mini-batching        | `1024`    |
| `--lr`         | Learning rate                       | `0.01`    |
| `--dropout`    | Dropout rate                        | `0.5`     |

---

## 📈 Sample Results

| Model     | Task               | Accuracy |
|-----------|--------------------|----------|
| GCN       | Node classification| 78.3%    |
| GraphSAGE | Edge classification| 85.1%    |

![GCN vs SAGE](GCN%20Vs.%20SAGE%20in%20Edge%20Prd.png)

---

## 📌 Features

- Modular model design (easy to plug in custom GNNs)
- Both node and edge classification
- Visualization and evaluation tools
- Works with OGB datasets
- GPU compatible

---

## 🙌 Contribution

Feel free to open issues or pull requests!  
For major changes, please open an issue first to discuss what you would like to change.

---

## 📃 License

This project is licensed under the UI License - see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

**Author:** M. Amin Kiani  
**Email:** amin.kiani82@gmail.com  
**LinkedIn:** [linkedin.com/in/amin-kiani](https://linkedin.com/in/amin-kiani) *(optional)*

---

> If you find this project helpful, please consider giving it a ⭐ on GitHub!
