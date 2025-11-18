# Active Learning for Illicit Transaction Detection on the Elliptic Bitcoin Dataset

This repository contains an end-to-end framework for detecting illicit cryptocurrency transactions using a combination 
of Graph Neural Networks (GNNs) and Active Learning (AL). The project evaluates how graph structure, temporal dynamics, 
and selective label acquisition influence model performance under severe label scarcity, using the Elliptic Bitcoin 
transaction dataset.
## Project Overview

The **[Elliptic dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)** 
 provides a large-scale Bitcoin transaction graph with **203k transactions**,  
**234k directed edges**, and **166 node features**. Only **2%** of nodes are labeled as illicit, creating a highly imbalanced and label-scarce environment.

We study:

- Whether **GNNs** improve detection over feature-only baselines.
- Whether **Active Learning** can reduce the labeling cost needed to reach high accuracy.
- How both components interact under real-world AML constraints.

We evaluate:

- **Models:** MLP, GCN, EvolveGCN-O, DySAT  
- **AL methods:** Random, Entropy, CMCS(certainty-oriented minority class sampling), Sequential (temporal-aware)
- **Graph variants:** Directed vs. Undirected, Local vs. Full feature sets

## Main Contributions

- Build a full dynamic-graph learning pipeline over the Elliptic dataset.
- Integrate multiple GNN architectures, including temporal ones (EvolveGCN, DySAT).
- Implement a complete Active Learning framework tailored to temporal, imbalanced graphs.
- Analyze convergence, label-efficiency, minority sampling behavior, and feature vs. structure importance.
- Provide a fully reproducible code base with clean configuration files.

## 📂 Repository Structure

```
Crypto_Fraud
├── code/
│   ├── active_learning.py/
│   ├── data.py/
│   ├── models.py/
│   ├── run_experiments.py
│   ├── training.py/
│   │── visual.py
│
│
├── configs/
├── results/
├── vizualizations/
├── README.md
└── requirements.txt
```

## Models

### **1. MLP (Baseline)**
A fully-connected neural network that treats each transaction independently.
- Uses only the node features (no graph structure).
- Strong baseline since Elliptic features are heavily engineered.
- Helps isolate the contribution of graph connectivity.



### **2. GCN (Graph Convolutional Network)**
A static GNN based on neighborhood aggregation.
- Each node updates its embedding by aggregating information from its neighbors.
- Captures local structural patterns such as suspicious clusters or dense fund-flow regions.



### **3. EvolveGCN-O**
A temporal extension of GCN for dynamic graphs.
- Uses a recurrent mechanism (GRU) to **evolve the GCN weights over time**.
- Learns how transaction behavior changes across the 49 time steps.
- Does not rely on fixed node embeddings, making it suitable for evolving financial networks.



### **4. DySAT (Dynamic Self-Attention Network)**
A dynamic GNN using dual attention mechanisms.
- **Structural attention:** learns which neighbors are most informative at each time step.
- **Temporal attention:** learns which past snapshots are relevant for the current state.
- Captures long-range dependencies and evolving laundering patterns.
- More expressive than simple GCN aggregation, especially in temporal settings.

---
## 🔍 Active Learning Framework

Loop:
1. Train  
2. Score  
3. Acquire  
4. Expand labeled set

## Active Learning Strategies

### **1. Random Sampling (Baseline)**
Selects unlabeled nodes uniformly at random.
- Serves as a control baseline.
- Useful for measuring whether more sophisticated strategies actually provide value.
- Ensures unbiased but inefficient coverage of the data.

---

### **2. Entropy Sampling**
Selects nodes with the highest predictive uncertainty.
- Measures uncertainty as the entropy of the predicted probability distribution.
- Focuses on samples near the decision boundary.
- Often accelerates learning when the model’s confidence is meaningful.

---

### **3. CMCS  (Certainty Minority-Oriented Sampling)**
A class-aware strategy designed for highly imbalanced datasets like Elliptic.
- Prioritizes nodes the model predicts as illicit (minority class).
- Addresses the tendency of uncertainty-based methods to oversample majority (licit) nodes.
- Aims to increase the proportion of illicit samples in the labeled pool, improving minority-class F1.

---

### **4. Sequential (Temporal-Aware) Sampling**
Selects nodes based on chronological order in the Bitcoin transaction graph.
- Mimics real-world AML workflows where future data is unavailable.
- Ensures the model learns only from information available “up to this time”.
- Useful for dynamic or streaming scenarios.

---

## 🧪 Experimental Setup

- 203,769 transactions  
- 49 time steps  
- Labels: 21% licit, 2% illicit, 77% unknown  
- Chronological split: Train 1–34, Val 35–41, Test 42–49  

Metrics:
- F1-illicit  
- AUPRC  
- Performance vs labeling budget

## 🧾 How to Run

```
pip install -r requirements.txt
python code/run_experiments.py --config configs/config_active.yaml
```

## 📈 Key Findings

- MLP strongest baseline  
- GNNs underperform due to weak graph signal  
- AL reaches passive performance with fewer labels  
- CMCS + Sequential achieve highest illicit ratios  

## 🔮 Future Work

- Dynamic AL  
- GAT/HGT models  
- Temporal rollouts  
