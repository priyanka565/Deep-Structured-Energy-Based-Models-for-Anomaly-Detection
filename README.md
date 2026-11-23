# Deep Structured Energy-Based Models (DSEBM) for Anomaly Detection – PyTorch

A PyTorch implementation of the Deep Structured Energy-Based Model (DSEBM) for unsupervised anomaly detection in network traffic, trained and evaluated on the KDDCup99 (10%) intrusion detection dataset.

The model is trained on normal traffic only and learns an energy function through denoising score matching. At test time, samples are flagged as anomalies using two criteria:

- **DSEBM-e:** high energy score
- **DSEBM-r:** high reconstruction error

## Credits

- **Method:** Zhai, S., Cheng, Y., Lu, W., & Zhang, Z. (2016). *Deep Structured Energy Based Models for Anomaly Detection.* ICML 2016. [arXiv:1605.07717](https://arxiv.org/abs/1605.07717)
- **Original implementation (TensorFlow 1):** [zehuichen123/DSEBM](https://github.com/zehuichen123/DSEBM). This project is a PyTorch port of that implementation. The data loader (`data/kddcup10.py`) and the preprocessed dataset are taken from it.
- **Dataset:** [KDD Cup 1999 Data](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html), UCI Machine Learning Repository

## My contributions

- **Ported the model from TensorFlow 1 to PyTorch.** I rebuilt the network as `nn.Module` classes and replaced the TF1 graph, session and placeholder setup with PyTorch's eager execution.
- **Implemented the energy gradient with `torch.autograd.grad`** (`create_graph=True`), so that the score-matching loss can be backpropagated through the gradient of the energy function.
- **Added Apple Silicon GPU support (MPS)**, with automatic fallback to CPU.
- **Fixed a preprocessing bug from the original code.** The training data was normalized into an unused variable (`data.data`), so the model was trained on unnormalized data. Min-max normalization is now correctly applied to `data.train_data`.
- **Simplified the evaluation pipeline.** Scores and labels are collected in plain lists instead of the original concatenate-and-slice approach.

## Results (KDDCup99 10%, 50 epochs)

| Score type | Precision | Recall | F1-score |
|---|---|---|---|
| DSEBM-e (energy) | 0.xx | 0.xx | 0.xx |
| DSEBM-r (reconstruction) | 0.xx | 0.xx | 0.xx |

Anomalies are the top 20% of samples by score, matching the outlier ratio of the dataset.

## Project structure

```
├── dsebm.py                       # DSEBM model, training and evaluation (PyTorch)
├── run.py                         # Entry point: load data, normalize, train, evaluate
├── data/kddcup10.py               # KDDCup99 loader and train/test split
└── kddcup99-10.data.pp.csv.gz     # Preprocessed KDDCup99 10% dataset (120 features + label)
```

## Usage

```bash
pip install torch numpy scikit-learn
python run.py
```

Settings (learning rate, batch size, epochs) are defined in `run.py`.

## Tech stack

Python · PyTorch · NumPy · scikit-learn
