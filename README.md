# ShadowWeave: Uncovering Hidden APT via Graph Neural Networks in Knowledge Graph

> This repository contains the source code and experimental data for the paper *"ShadowWeave: Uncovering Hidden APT via Graph Neural Networks in Knowledge Graph"* (ICDE), enabling reproduction of the experiments presented in the paper.

## 📋 Paper Overview

**ShadowWeave** is a Graph Neural Network (GNN)-based inference engine for relation reasoning in APT (Advanced Persistent Threat) knowledge graphs. It unifies explicit and implicit relation processing, supporting cross-stage threat path inference and unknown relation completion.

### Key Contributions

- A GNN-compatible framework optimizing APT knowledge graph representations
- A unified reasoning engine for both explicit and implicit relation processing
- Significant performance improvements on real-world datasets, supporting five critical cybersecurity tasks

## 🗂️ Dataset

The experiments utilize two types of datasets:

### 1. Explicit Relations Dataset
- **Source**: APT Knowledge Graph (based on STIX 2.1, MITRE ATT&CK, VirusTotal, etc.)
- **Content**: 10 entity types, 11 relation types, 47,091 triples
- **Split**: 80% training, 10% validation, 10% test

### 2. Implicit Relations Dataset
- **Source**: VirusTotal
- **Target Relation**: `Indicator → Uses → TTPs`
- **Content**: 10,127 Indicators, 293 TTPs, 148,231 triples

All datasets are available in the `data/` directory of this repository.

## 📊 Experimental Results

### Explicit Relation Prediction (Overall)

| Model | Hit@1 | Hit@3 | MRR |
|-------|-------|-------|-----|
| ShadowWeave | 0.9941 | 0.9998 | 0.9967 |
| RGCN | 0.9821 | 0.9978 | 0.9898 |
| HeteroGraphSAGE | 0.9924 | 0.9998 | 0.9958 |
| CompGCN | 0.9845 | 0.9981 | 0.9902 |
| HAN | 0.7521 | 0.8934 | 0.8123 |
| RGAT | 0.8234 | 0.9215 | 0.8657 |
| HGT | 0.7891 | 0.9012 | 0.8345 |

### Implicit Relation Discovery (Indicator→Uses→TTPs)

| Model | Accuracy |
|-------|----------|
| ShadowWeave | 0.9472 |
| HeteroGraphSAGE | 0.9270 |
| RGCN | 0.5337 |
| CompGCN | 0.5903 |
| HAN | 0.5895 |
| RGAT | 0.6672 |
| HGT | 0.5649 |

### Cybersecurity Task Support (Average Accuracy)

| Task | ShadowWeave | HAN | RGCN | HeteroGraphSAGE |
|------|-------------|-----|------|-----------------|
| APT Group Profiling | 0.9863 | 0.7521 | 0.9012 | 0.9547 |
| Malware Propagation Analysis | 0.9988 | 0.6987 | 0.9123 | 0.9789 |
| Attack Target Prediction | 0.9401 | 0.6345 | 0.8821 | 0.9234 |
| Attack Technique Inference | 0.9577 | 0.6892 | 0.8945 | 0.9412 |
| APT Group & Malware Attribution | 1.0000 | 0.6875 | 0.9159 | 0.8654 |
| **Average** | **0.9766** | **0.6924** | **0.9012** | **0.9327** |

### Low-frequency Relationship Performance (Class Weighting Ablation)

| Power Value | Low-frequency Hit@1 | Low-frequency MRR | Overall Hit@1 | Overall MRR |
|-------------|---------------------|-------------------|---------------|-------------|
| No Weighting | 0.2800 | 0.5124 | 0.9926 | 0.9957 |
| power=0.5 | 0.7200 (+157.14%) | 0.8389 (+63.64%) | 0.9943 (+0.17%) | 0.9969 (+0.12%) |
| power=1.0 (Default) | 0.8200 (+192.86%) | 0.8933 (+74.32%) | 0.9941 (+0.15%) | 0.9967 (+0.10%) |
| power=1.5 | 0.9000 (+221.43%) | 0.9400 (+83.45%) | 0.9511 (-4.11%) | 0.9667 (-2.92%) |
| power=2.0 | 0.2400 (-14.29%) | 0.4509 (-12.00%) | 0.4515 (-54.51%) | 0.5789 (-41.88%) |

### Feature Encoding Ablation Study

| Variant | Explicit Hit@1 | Explicit MRR | Implicit Accuracy |
|---------|----------------|--------------|-------------------|
| SW-Cat (Categorical only) | 0.9919 | 0.9954 | 0.7441 |
| SW-CatNum (+ Numerical) | 0.9928 | 0.9961 | 0.8757 |
| SW-CNT50 (+ Text dim=50) | 0.9921 | 0.9956 | 0.7501 |
| SW-CNT100 (+ Text dim=100) | 0.9925 | 0.9959 | 0.8331 |
| SW-CNT200 (+ Text dim=200) | 0.9941 | 0.9967 | 0.9472 |
| SW-CNT300 (+ Text dim=300) | 0.9926 | 0.9958 | 0.9600 |

### Multi-hop Aggregation Ablation Study

| Variant | Explicit Hit@1 | Explicit MRR | Implicit Accuracy |
|---------|----------------|--------------|-------------------|
| SW-1hop | 0.9924 | 0.9958 | 0.9270 |
| SW-2hop | 0.9931 | 0.9962 | 0.9352 |
| SW-3hop (Default) | 0.9941 | 0.9967 | 0.9472 |
