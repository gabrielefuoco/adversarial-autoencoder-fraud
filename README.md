# Adversarial Autoencoder for Financial Fraud Detection

![Latent space of the standard autoencoder](Relazione%20Finale/assets/tsne_standard_ae.png)

A reproduction and extension of an adversarial-autoencoder approach to financial anomaly detection: the latent space learned by an autoencoder is used as feature space for supervised classifiers, and every claim is measured against a baseline. The headline finding is that the adversarial regularisation does not pay for itself on this dataset.

## Table of contents

- [Results at a glance](#results-at-a-glance)
- [How it works](#how-it-works)
- [Decisions](#decisions)
- [Repository map](#repository-map)
- [Running the experiments](#running-the-experiments)
- [Limitations](#limitations)
- [Documentation](#documentation)

## Results at a glance

Multi-class AUC on the same latent dimensionality ($d=10$), classifiers trained on the autoencoder's latent vectors:

| Classifier | Adversarial AE ($z_{AAE}$) | Standard AE ($z_{AE}$) | Delta |
| :--- | :---: | :---: | :---: |
| **Random Forest** | 0.9687 | **0.9841** | +1.54% for the standard AE |
| **SVM (RBF kernel)** | 0.9950 | **0.9962** | +0.12% for the standard AE |

The ablation keeps encoder, decoder, layers, neurons and activations identical and removes only the discriminator and the adversarial phase, so the difference is attributable to the latent regularisation alone. Forcing the latent space towards a Gaussian mixture made classification slightly worse, not better: on this dataset the network's own topology separates fraud better than an imposed one.

## How it works

### The dataset

A derivative of **PaySim**, a simulator of mobile-money transactions: **532,909 transactions** with a **1–2% fraud** rate and **618 features**. The hard part is the mix: high-cardinality categorical codes (general ledger account `HKONT`, posting key `BSCHL`, profit centre `PRCTR`) next to continuous amounts (`DMBTR`, `WRBTR`). Encoding those columns naively produces unmanageable sparse vectors, so the preprocessing is where the work is.

### The latent representation

Three networks in PyTorch — encoder, decoder, discriminator. The encoder maps a transaction to a **10-dimensional latent vector**; the decoder reconstructs the input; the discriminator pushes the aggregated latent distribution towards a Gaussian mixture, so the latent code is both reconstructive and structurally regularised. The latent vectors then become the **features** for SVM and Random Forest, which is the integration proposed by the original paper.

### The metric

**MAUC** — Hand & Till's multi-class extension of the AUC (2001), pairwise averaging of the probabilistic AUC approximation — implemented in `MAUC.py`. The choice matters: on a dataset that is 98–99% licit, accuracy is meaningless (a classifier that always answers "regular" scores above 0.99). The AUC measures whether frauds are *ranked* above licit transactions, which is what a review queue needs.

### Robustness to imbalance

A stress test sweeps the fraud ratio in the training set:

- **below 1%** the model holds, with slightly higher variance;
- **around 2%** performance peaks — a small contamination appears to help the discriminator draw the prior's boundaries, acting as an implicit regulariser;
- **above 5%** it collapses, and at **10%** the MAUC of linear classifiers falls below **0.85**.

The cause is **pollution**: once anomalies are frequent enough, the autoencoder learns them as normal and reconstructs them with low error, so the reconstruction score loses its discriminative power. Operationally this method is valid only while frauds stay rare; a sudden *fraud spike* can make the system blind, which is why a production deployment needs distribution-drift monitoring beside the model.

![Effect of pollution: MAUC against fraud ratio](Relazione%20Finale/assets/imbalance_pollution.png)

## Decisions

**Latent vectors as features, not as an anomaly score.** The original design scores transactions by reconstruction error. Using the latent code as a feature space lets a supervised classifier exploit the labels that already exist, and makes the comparison between architectures measurable with the same metric.

**Standard AE as the ablation, not PCA.** The paper compares the AAE against classical methods, not against an unregularised autoencoder of the same shape. Without that control, a difference in performance cannot be attributed to the regularisation.

**GPU stack: RAPIDS cuML instead of scikit-learn.** An RBF SVM on CPU took hours on 500k high-dimensional samples, which made hyperparameter tuning impossible, and t-SNE at $O(N \log N)$ forced aggressive downsampling that destroyed topology. `cuml.svm.SVC` and `cuml.manifold.TSNE` brought it to seconds, orders of magnitude faster, and made the full dataset analysable. A compatibility wrapper was needed to keep `OneVsRestClassifier` and the existing evaluation pipeline working.

**A pinned Docker image instead of a documented environment.** `rapidsai/base:23.10-cuda11.8-py3.9` plus PyTorch cu118, so results do not depend on the host's drivers. The trade-off is a heavy image; the gain is that an experiment can be repeated bit-perfect on another machine.

## Repository map

```
Encoder.py, Decoder.py, Discriminator.py   the three networks
validation_V0.py                           trains the autoencoder
generateDatasets.py                        materialises datasets from the latent spaces
MAUC.py                                    multi-class AUC (Hand & Till, 2001)
MAUC_Exp.py                                evaluation with latent features
MAUC_rawExp.py                             evaluation with the original 618 features
analyseResults.py                          aggregation and plots of the results
classification_eval.py                     classifier-side evaluation
project_extension/                         standard-AE ablation and comparative evaluation
Relazione Finale/                          the full report, one chapter per folder
Progetto ML.md                             working notes and experiment log
```

## Running the experiments

```bash
docker build -t fraud-detection-rapids .
docker run --gpus all -it --rm -v "$PWD":/workspace fraud-detection-rapids
```

Then, inside the container, in this order: train the autoencoder, generate the datasets from the latent spaces, and run the evaluations (`MAUC_Exp.py` for latent features, `MAUC_rawExp.py` for the original ones).

**The dataset is not in this repository.** It has to be obtained and placed where the scripts expect it, which is the main obstacle to reproducing the numbers from a clean clone.

## Limitations

| Item | Impact | Status |
|---|---|---|
| The dataset is not versioned | a clean clone cannot run anything until the CSV is supplied | open, by choice (size) |
| No automated tests | the pipeline is validated by re-reading results, not by a suite | open |
| The AAE loses against the standard AE | the adversarial regularisation is not justified on this data | measured, reported as a negative result |
| Collapse above 5% fraud ratio | the method is blind to fraud spikes | measured, needs drift monitoring in production |
| Synthetic data (PaySim) | the numbers do not transfer to a real ledger without re-validation | accepted |

## Documentation

The complete report lives in [`Relazione Finale/`](Relazione%20Finale), one chapter per folder:

- **Architectural analysis and theory** — the financial anomaly-detection problem, autoencoders, adversarial autoencoders
- **The original baseline** — the reference paper, the dataset, the original pipeline, reproduction of its results
- **Extension and methodology** — stack modernisation, data-pipeline memory engineering, the imbalance stress test, training stability, the ablation design
- **Results and discussion** — MAUC comparison, t-SNE analysis, training stability, robustness to imbalance
- **Conclusions and appendices**
