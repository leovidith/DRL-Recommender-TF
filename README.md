# Deep Reinforcement Learning for Recommendation (TensorFlow)

A high-fidelity TensorFlow implementation of a recommender system powered by Deep Deterministic Policy Gradient (DDPG), based on
**Liu et al., "Deep Reinforcement Learning-based Recommendation with Explicit User-Item Interactions Modeling" (arXiv:1810.12027)**
This version integrates performance-oriented modifications and state representation learning.

---

## Dataset

* [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)
  Extract using:

```bash
unzip ./ml-1m.zip
```

---

## Core Contributions

* State-aware user modeling with learnable embedding layers
* Enhanced policy learning via Prioritized Experience Replay (PER)
* Mitigated Q-value overestimation to stabilize training
* Time-windowed embeddings to remove data leakage
* Custom training and evaluation flow tailored for reproducibility and performance

---

## Results

Evaluation scores on held-out interactions:

* Precision\@5: 0.479 | nDCG\@5: 0.471
* Precision\@10: 0.444 | nDCG\@10: 0.429

[Experiment Report (Korean)](https://www.notion.so/DRR-8e910fc598d242968bd371b27ac20e01)

---

## Usage

### Train

```bash
python train.py
```

### Evaluate

Ensure saved models exist. Launch:

```bash
jupyter notebook
```

Run `evaluation.ipynb` for metrics.

---

## Requirements

```bash
tensorflow==2.17.0  
scikit-learn==1.4.2  
matplotlib==3.8.4
```

## Summary

This repository reconstructs and evolves a DRL-based recommendation framework with precision-focused state modeling, policy refinement, and principled evaluation—suitable for academic benchmarking or experimental pipelines in applied reinforcement learning.
