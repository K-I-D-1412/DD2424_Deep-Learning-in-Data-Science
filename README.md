# DD2424 Deep Learning in Data Science

> **Note**: This project was developed as part of the *Deep Learning in Data Science* course (DD2424) at KTH Royal Institute of Technology.
> 
> **Code of Honour**: If there are similar questions or labs or projects in the future, it is the responsibility of KTH students not to copy or modify these codes, or other files because it is against the [KTH EECS Code of Honour](https://www.kth.se/en/eecs/utbildning/hederskodex). The owner of this repository doesn't take any commitment for other's faults.

---

## 📌 Project Overview

This repository contains a progressive series of image classifiers built **entirely from scratch** using NumPy, applied to the **CIFAR-10** dataset. Starting from a single-layer linear network and evolving to a two-layer neural network with advanced optimization, each stage introduces new techniques while preserving mathematical rigor — all gradients are hand-derived and analytically verified.

### Performance Summary

| Model | Architecture | Optimizer | Key Techniques | Test Accuracy |
| :--- | :--- | :--- | :--- | :---: |
| Baseline (Softmax) | Single-layer | SGD | — | 39.30% |
| Bonus 1 (Softmax) | Single-layer | SGD | Full data, Augmentation, Grid Search | 42.02% |
| Bonus 2 (BCE) | Single-layer | SGD | Sigmoid + BCE Loss, Full data, Augmentation | 41.92% |
| Assignment 2 Baseline | Two-layer (m=50) | SGD + CLR | Cyclical LR, Coarse-to-Fine Search | 51.18% |
| Assignment 2 Bonus (CLR) | Two-layer (m=200) | SGD + CLR | Dual Augmentation, Network Scaling | **56.51%** |
| Assignment 2 Bonus (Adam) | Two-layer (m=200) | Adam | Adam Optimizer, Dual Augmentation | 54.18% |
| Assignment 3 Initial ConvNet | Patchify ConvNet (f=4, nf=10, nh=50) | SGD + CLR | Efficient patch matrix convolution, hand-derived gradients | 56.21% |
| Assignment 3 Best Short Run | Patchify ConvNet (f=8, nf=40, nh=50) | SGD + CLR | Filter-size comparison, cached MX representation | 58.71% |
| Assignment 3 Best Long Run | Patchify ConvNet (f=4, nf=40, nh=50) | SGD + Increasing CLR | Wider convolution layer, increasing cycle lengths | 64.64% |
| Assignment 3 Large ConvNet | Patchify ConvNet (f=4, nf=40, nh=300) | SGD + Increasing CLR | Large network, L2 regularization | **66.10%** |
| Assignment 3 Large ConvNet + LS | Patchify ConvNet (f=4, nf=40, nh=300) | SGD + Increasing CLR | Label smoothing ε=0.1, reduced test loss | 66.02% |
| Assignment 3 Bonus (Flip) | Patchify ConvNet (f=4, nf=40, nh=300) | SGD + Increasing CLR | Random horizontal flip augmentation | 69.50% |
| Assignment 3 Bonus (Flip, 5 Cycles) | Patchify ConvNet (f=4, nf=40, nh=300) | SGD + Increasing CLR | Random horizontal flip, longer training | **69.80%** |

---

## 📁 Code Files Reference

| File | Description |
| :--- | :--- |
| [`image_classifier.py`](image_classifier.py) | **Assignment 1 — Baseline.** Single-layer Softmax classifier. Implements core pipeline: data loading, forward pass, cross-entropy + L2 loss, hand-derived analytical gradients, and mini-batch SGD. Runs 4 experiments with different hyperparameter configurations. |
| [`bonus_image_classifier.py`](bonus_image_classifier.py) | **Assignment 1 — Bonus 1.** Enhanced single-layer Softmax classifier. Adds full dataset utilization (49k training), on-the-fly random horizontal flipping, automated grid search, and optional step decay for learning rate scheduling. |
| [`BCE_image_classifier.py`](BCE_image_classifier.py) | **Assignment 1 — Bonus 2.** Single-layer classifier refactored with Sigmoid activation + Multiple Binary Cross-Entropy (BCE) loss. Key change: gradient scaled by 1/K, requiring a 10x larger learning rate. Also generates confidence histograms for qualitative comparison with Softmax. |
| [`two_layer_image_classifier.py`](two_layer_image_classifier.py) | **Assignment 2 — Core.** Two-layer neural network (input → ReLU hidden → Softmax output). Introduces Cyclical Learning Rates (CLR), coarse-to-fine random search for λ on log-scale, and He initialization (`1/√d`). |
| [`bonus_two_layer_image_classifier.py`](bonus_two_layer_image_classifier.py) | **Assignment 2 — Bonus.** Extended two-layer network with: network scaling (m=200), dual data augmentation (flipping + spatial translation ±3px), Inverted Dropout, and Adam optimizer support. Supports `optimizer='sgd'` and `optimizer='adam'` modes via a unified `MiniBatchGD` interface. |
| [`torch_gradient_computations.py`](torch_gradient_computations.py) | **Utility.** Uses PyTorch's autograd to independently compute gradients for verification against hand-derived analytical gradients (max error ~10⁻⁸). |
| [`assignment3_convnet.py`](assignment3_convnet.py) | **Assignment 3 — Patchify ConvNet.** Three-layer image classifier with an initial non-overlapping patch-based convolution layer. Implements slow convolution checks, efficient `MX @ Fs_flat` / `np.einsum` convolution, hand-derived forward/backward passes, L2 regularization, convolution bias, label smoothing, increasing cyclical learning rates, cached patch matrices, and experiment plotting utilities. |
| [`assignment3_bonus_convnet.py`](assignment3_bonus_convnet.py) | **Assignment 3 — Bonus.** Separate bonus implementation based on the verified Assignment 3 ConvNet. Adds dynamic random horizontal flip augmentation for training mini-batches, rebuilds `MX_batch` on the fly, and includes bonus experiments for label smoothing, L2 regularization search, longer training, and maximum learning-rate comparison. |
---

## 🛠️ Part 1: Basic Framework & Baseline (Exercise 1)

### Implementation Details
The foundational framework (`image_classifier.py`) includes the following core components:
* **Data Pipeline:** Loading CIFAR-10 batches, transforming labels to one-hot encoding, and pre-processing data (zero-mean normalization based strictly on training set statistics).
* **Forward Pass:** Linear transformation ($s = Wx + b$) followed by a Softmax activation function.
* **Cost Function:** Computes the Cross-Entropy Loss combined with an L2 Regularization term to prevent over-fitting.
* **Backward Pass:** Hand-derived analytical gradients for the weight matrix ($W$) and bias vector ($b$). *Note: The analytical gradients were rigorously tested against PyTorch's automatic differentiation, achieving a maximum absolute error at the $10^{-8}$ scale.*
* **Training Loop:** Mini-batch Gradient Descent.

### Baseline Experimental Results
The baseline network was trained on a single batch (10,000 images) for 40 epochs across four different hyperparameter configurations:

| Experiment | L2 Reg ($\lambda$) | Learning Rate ($\eta$) | Batch Size | Final Test Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Exp 1** | 0 | 0.1 | 100 | 29.49% |
| **Exp 2** | 0 | 0.001 | 100 | 39.21% |
| **Exp 3** | 0.1 | 0.001 | 100 | **39.30%** |
| **Exp 4** | 1.0 | 0.001 | 100 | 37.55% |

### Analysis & Visualizations

* **The Importance of a Correct Learning Rate ($\eta$):** A high learning rate ($\eta = 0.1$ in Exp 1) caused severe divergence, as seen below. The loss spikes uncontrollably.
  
  **Figure 1: High Learning Rate Divergence (Exp 1)**
  ![High Learning Rate Loss Plot](images/loss_lam0_eta0.1_batch100.png)

* **Baseline Generalization:** Experiment 3 ($\lambda=0.1$) yielded the best baseline generalization. However, even with mild regularization, a slight gap between validation and training loss begins to appear around epoch 30, suggesting the onset of over-fitting.
  
  **Figure 2: Baseline Best Practice (Exp 3)**
  ![Baseline Convergence Loss Plot](images/loss_lam0.1_eta0.001_batch100.png)

* **Weight Templates:** The visualization of the $W$ matrix (Figure 3) reveals that a single-layer network learns colors and vague contours rather than distinct shapes. You can see a fuzzy horse shape (green torso on green background) and a centered automobile contour.
  
  **Figure 3: Baseline Learnt Weight Templates (Exp 3)**
  ![Baseline Fuzzy Weights Visualization](images/weights_lam0.1_eta0.001_batch100.png)

---

## 🚀 Part 2: Performance Improvements (Exercise 2.1 / Bonus 1)

To maximize the performance of a simple linear classifier, several advanced techniques were implemented in `bonus_image_classifier.py`:

### Enhancements Implemented
1.  **Full Dataset Utilization:** Concatenated all 5 CIFAR-10 training batches, utilizing 49,000 images for training and 1,000 for validation.
2.  **Data Augmentation:** Implemented an on-the-fly random horizontal flip with a 50% probability during training to force the network to learn translation-invariant features.
3.  **Automated Grid Search:** Built a grid search mechanism to systematically find the optimal combination of $\lambda$, $\eta$, and batch size.
4.  **Step Decay (Learning Rate):** Explored reducing the learning rate by a factor of 10 every 10 epochs (ultimately disabled for the linear model as it caused premature loss plateauing). 

### Final Results & Analysis
Through Grid Search, the **optimal configuration** was found to be: 
`n_batch = 100`, `eta = 0.001`, and `lam = 0.0`. 

Using this configuration, the network underwent a final deep training phase of 100 epochs, achieving a **final test accuracy of 42.02%**.

**Key Takeaways:**
* **Eradication of Over-fitting:** The combination of Data Augmentation and a massive increase in training data proved incredibly effective. As shown in **Figure 4**, the validation loss flattened and remained entirely stable from epoch 40 to 100 without diverging upwards, proving that over-fitting was completely mitigated.
* **Symmetrical Weight Templates:** Due to the 50% horizontal flipping, the learned weight templates (e.g., "horse" and "automobile") became highly symmetrical (Figure 5). The network successfully learned generalized, centered features rather than memorizing orientation.

  **Figure 4: 100-Epoch Final Loss Curve (Softmax)**
  ![Bonus 1 Final Deep Training Loss Plot](images/Bonus1_Final_Loss.png)

  **Figure 5: 100-Epoch Final Learnt Weight Templates (Softmax)**
  ![Bonus 1 Final Symmetric Weights Visualization](images/Bonus1_Final_Weights.png)

---

## 🧠 Part 3: Multiple Binary Cross-Entropy Loss (Exercise 2.2 / Bonus 2)

To understand the underlying mathematical structure of different loss functions, the network was entirely refactored in `BCE_image_classifier.py` to use a **Sigmoid** activation function paired with a **Multiple Binary Cross-Entropy (BCE)** loss, instead of Softmax and Cross-Entropy.

### Mathematical & Architectural Shift
* **Assumption Change:** While Softmax forces a mutually exclusive probability distribution (single-label), Sigmoid treats the prediction of each of the 10 classes as an independent binary classification problem (multi-label).
* **Analytical Gradient:** The gradient of the Multiple BCE loss with respect to the scores $s$ was hand-derived as $\frac{\partial l}{\partial s} = \frac{1}{K}(p - y)$. This is structurally identical to the Softmax gradient but is scaled down by a factor of $1/K$ (where $K=10$).

### Final Results & Analysis
Because the gradient is scaled down by $1/10$, the network required a proportionally larger learning rate. Grid search identified the **optimal configuration** as:
`n_batch = 100`, `eta = 0.01`, and `lam = 0.001`.

After training for 100 epochs with data augmentation, the model achieved a **final test accuracy of 41.92%**.

**Key Takeaways:**
* **Comparable but Marginally Lower Performance:** The accuracy (41.92%) is highly comparable but very slightly lower than the Softmax counterpart (42.02%). This mathematically aligns with the nature of the CIFAR-10 dataset, where labels are strictly mutually exclusive. Softmax exploits this mutual exclusivity perfectly, while Sigmoid makes a weaker assumption. 
* **Model Capacity Limit:** As seen in Figure 6, the training and validation curves overlap almost perfectly. The lack of a gap confirms zero over-fitting; the model is simply operating at its maximum mathematical capacity for a zero-hidden-layer architecture.

  **Figure 6: 100-Epoch Final Loss Curve (Multiple BCE)**
  ![BCE Loss Plot](images/BCE_Loss.png)

  **Figure 7: Learnt Weight Templates (Multiple BCE)**
  ![BCE Weights Visualization](images/BCE_Weights.png)

### Confidence Histogram Analysis
To evaluate the qualitative difference in prediction confidence between the two architectures, histograms of the predicted probability for the **ground truth class** were generated for both correctly and incorrectly classified test examples.

**Figure 8: Confidence Distribution (Softmax vs. Multiple BCE)**
![Softmax Confidence Histogram](images/Softmax_Histogram.png)
![BCE Confidence Histogram](images/BCE_Histogram.png)

**Qualitative Differences:**
1. **Softmax ("Winner-Takes-All"):** In the Softmax model, probabilities are forced to sum to 1. This creates a highly competitive distribution. Correct predictions (green) frequently reach high absolute probabilities (0.6 to 1.0). Conversely, when the network is incorrect (red), it assigns near-zero probability to the ground truth class because it is highly confident in a wrong class.
2. **Multiple BCE / Sigmoid ("Independent Evaluation"):** The Sigmoid model treats each class independently. Noticeably, the entire distribution shifts significantly to the left. Even for correctly classified examples, the absolute probability assigned to the ground truth class peaks around 0.3 and rarely exceeds 0.6. The network doesn't need to push the absolute value to 1.0; it only needs the ground truth class score to be *relatively* higher than the other 9 independent classes to make a correct prediction.

---

## 🚀 Part 4: Two-Layer Neural Network & Cyclical Learning Rates (Assignment 2)

Building upon the single-layer baseline, the project was expanded to implement a **Two-Layer Neural Network** with a ReLU activation function in the hidden layer. This phase introduced advanced training methodologies, heavily focusing on learning rate scheduling and hyperparameter optimization.

### 1. Architecture & Sanity Check
Before training on the full dataset, the analytical gradients (derived by hand and vectorized in NumPy) were strictly verified. By temporarily disabling L2 regularization ($\lambda = 0$) and training on a micro-batch of 100 images, the network successfully overfitted the data, driving the loss to near zero. This confirmed the mathematical correctness of the backpropagation implementation.

**Figure 9: Sanity Check (Overfitting 100 Images)**
![Sanity Check Loss](images/Assignment2/sanity_check_loss.png)

### 2. Cyclical Learning Rates (CLR)
To eliminate the need for exhaustive learning rate tuning and to prevent the network from getting stuck in saddle points or local minima, **Cyclical Learning Rates (CLR)** were implemented. The learning rate oscillates in a triangular schedule between a defined $\eta_{min}$ and $\eta_{max}$.

**Figure 10: Training Metrics over One Cycle vs. Three Cycles**
![CLR One Cycle](images/Assignment2/Figure3_One_Cycle.png)
![CLR Three Cycles](images/Assignment2/Figure4_Three_Cycles.png)
*Observation: The cyclical nature of the learning rate creates distinct "waves" in the cost and accuracy plots. Each cycle allows the network to jump out of suboptimal minima, progressively refining the generalization.*

### 3. Hyperparameter Search & Baseline Final Training
A **Coarse-to-Fine Random Search** was conducted on a logarithmic scale to find the optimal L2 regularization penalty ($\lambda$). 
* **Baseline Configuration:** Hidden nodes $m=50$, trained for 3 full cycles.
* **Optimal $\lambda$:** $\approx 0.000094$
* **Result:** The baseline two-layer network achieved a final test accuracy of **51.18%**.

**Figure 11: Final Baseline Training (m=50)**
![Baseline Final Training](images/Assignment2/Figure5_Final_Training.png)

---

## 🏆 Part 5: Advanced Optimizations & Network Scaling (Bonus)

To push the limits of the two-layer architecture, several advanced techniques were systematically tested and combined.

### 1. Network Expansion & Dual Data Augmentation
The network capacity was significantly increased by quadrupling the hidden nodes from $m=50$ to $m=200$. To prevent this highly parameterized network from overfitting, a robust dual data augmentation strategy was introduced:
* **Random Horizontal Flipping (Mirroring):** Pre-computed pixel-index permutation for efficient vectorized flipping at 50% probability.
* **Random Spatial Translations:** Each image randomly shifted by ($\pm 3$ pixels) in both x and y directions with zero-padding.

**Result:** Training this expanded architecture for 5 cycles yielded a massive performance jump, achieving a final test accuracy of **56.51%** (an absolute improvement of +5.33% over the baseline). The aggressive data augmentation acted as an exceptionally strong regularizer, keeping the validation curves tightly tracking the training curves.

**Figure 12: Expanded Network Training (m=200, 5 Cycles, Augmented)**
![M200 Final Training](images/Assignment2/Figure5_Final_Training_m200_5cycles.png)

*Note on Dropout:* An experiment was conducted using Inverted Dropout in the hidden layer. However, combining Dropout with aggressive data transformations resulted in **over-regularization** (underfitting) for a network of this capacity.

### 2. Optimizer Showdown: Adam vs. SGD with CLR
As a final experiment, the cyclical learning rate schedule was replaced with the **Adam Optimizer** using a fixed base learning rate ($\eta = 5 \times 10^{-4}$). The Adam optimizer combines the benefits of Momentum and RMSprop with bias-corrected first and second moment estimates.

**Result:** Adam achieved a final test accuracy of **54.18%**. 

**Figure 13: Adam Optimizer Training Curves**
![Adam Training](images/Assignment2/Figure6_Final_Training_Adam.png)

**Key Takeaway (Adam vs. CLR):**
While the Adam optimizer provided remarkably smooth curves and rapid initial convergence, it slightly underperformed compared to SGD with Cyclical Learning Rates (56.51%). This experiment practically highlights a known deep learning phenomenon: the periodic large learning rate spikes in CLR act as a powerful *implicit regularizer*, forcing the network to settle into wider, more robust global minima, whereas Adam can sometimes converge too quickly into sharper local minima.

---

## 🧩 Part 6: Patchify Convolutional Network (Assignment 3)

Assignment 3 extends the previous NumPy classifiers by replacing the raw flattened input layer with a lightweight patch-based convolution layer. The model is still implemented fully from scratch, including forward propagation, backpropagation, gradient checks, training loops, learning-rate schedules, and experiment management.

### 1. Efficient Patch-Based Convolution

The input CIFAR-10 image is first reshaped from a flattened vector into a `32 × 32 × 3` image. For a filter width `f`, each image is divided into non-overlapping patches. These patches are flattened into a matrix `MX`, where each row corresponds to one image patch.

The convolution filters are also flattened into `Fs_flat`, so the convolution can be computed efficiently as a matrix multiplication. The final implementation uses:

```python
np.einsum("ijn,jl->iln", MX, Fs_flat, optimize=True)
```

This removes the explicit loop over images and gives an efficient batch convolution implementation.

The implementation was verified against the provided `debug_info.npz` file. The slow convolution, patch matrix construction, flattened filters, matrix multiplication convolution, `einsum` convolution, forward pass, and backward pass all matched the reference outputs with maximum absolute difference `0.000000000000e+00`.

### 2. Training Infrastructure

The final `assignment3_convnet.py` script includes:

* cached patch matrices under `assignment3_cache/`;
* experiment summaries and trained parameters under `assignment3_outputs/`;
* report figures under `assignment3_report_assets/`;
* standard cyclical learning rates;
* increasing-cycle-length learning rates;
* L2 regularization on `Fs_flat`, `W1`, and `W2`;
* convolution bias `b0`;
* optional label smoothing;
* command-line experiment actions via `--action`.

### 3. Short Training Runs

Four architectures were trained to compare different filter widths while keeping the flattened convolution output dimension approximately comparable.

| Architecture | f | nf | nh | Training Time | Validation Accuracy | Test Accuracy |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Architecture 1 | 2 | 3 | 50 | 31.54s | 49.50% | 51.16% |
| Architecture 2 | 4 | 10 | 50 | 25.56s | 57.00% | 56.21% |
| Architecture 3 | 8 | 40 | 50 | 25.49s | 59.70% | 58.71% |
| Architecture 4 | 16 | 160 | 50 | 29.43s | 57.40% | 56.56% |

Architecture 3 achieved the best short-run performance, suggesting that an intermediate filter size can provide a good balance between local feature extraction and representation capacity.

**Figure 14: Assignment 3 Short-Run Test Accuracy**

![Assignment 3 Short-Run Test Accuracy](assignment3_report_assets/ex3_short_runs_test_accuracy.png)

**Figure 15: Assignment 3 Short-Run Training Time**

![Assignment 3 Short-Run Training Time](assignment3_report_assets/ex3_short_runs_training_time.png)

### 4. Longer Training with Increasing Cycle Lengths

The cyclical learning rate schedule was extended so that the half-cycle length doubles after each cycle:

* Cycle 1: `step_size = 800`
* Cycle 2: `step_size = 1600`
* Cycle 3: `step_size = 3200`

This gives `11200` total update steps for three cycles.

| Model | f | nf | nh | Training Time | Validation Accuracy | Test Accuracy |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Long Run 1 | 4 | 10 | 50 | 63.55s | 57.10% | 58.13% |
| Long Run 2 | 8 | 40 | 50 | 53.36s | 62.70% | 61.08% |
| Long Run 3 | 4 | 40 | 50 | 123.69s | 64.40% | 64.64% |

The strongest long-run model was `f=4, nf=40, nh=50`, reaching **64.64%** test accuracy. This suggests that the improvement of the wider models was strongly related to the increased number of convolution filters, not only the larger filter width.

**Figure 16: Long Training Loss Curve (f=4, nf=10)**

![Long Training Loss Curve f4 nf10](assignment3_report_assets/ex3_long_loss_f4_nf10.png)

**Figure 17: Long Training Loss Curve (f=8, nf=40)**

![Long Training Loss Curve f8 nf40](assignment3_report_assets/ex3_long_loss_f8_nf40.png)

**Figure 18: Long Training Loss Curve (f=4, nf=40)**

![Long Training Loss Curve f4 nf40](assignment3_report_assets/ex3_long_loss_f4_nf40.png)

### 5. Larger Network and Label Smoothing

A larger network was trained with:

* `f = 4`
* `nf = 40`
* `nh = 300`
* `lambda = 0.0025`
* `n_cycles = 4`
* `step_size_1 = 800`

Two versions were compared: one without label smoothing and one with label smoothing using `epsilon = 0.1`.

| Setting | Training Time | Train Subset Accuracy | Validation Accuracy | Test Accuracy | Test Loss | Test Cost |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| No Label Smoothing | 392.46s | 99.30% | 67.80% | **66.10%** | 1.0504 | 1.5872 |
| Label Smoothing ε=0.1 | 502.56s | 97.78% | 68.20% | 66.02% | 1.0302 | 1.4123 |

Label smoothing did not significantly improve test accuracy in this run, but it reduced the test loss from `1.0504` to `1.0302` and lowered the test cost from `1.5872` to `1.4123`. This suggests that label smoothing mainly reduced overconfident predictions and improved the loss behavior rather than changing the final classification accuracy.

**Figure 19: Exercise 4 Loss Curves without Label Smoothing**

![Exercise 4 Loss Curves without Label Smoothing](assignment3_report_assets/ex4_loss_no_label_smoothing.png)

**Figure 20: Exercise 4 Loss Curves with Label Smoothing**

![Exercise 4 Loss Curves with Label Smoothing](assignment3_report_assets/ex4_loss_label_smoothing.png)

---

## 🏆 Part 7: Bonus Experiments — Data Augmentation for Patchify ConvNet

The Assignment 3 bonus experiments were implemented in a separate file, `assignment3_bonus_convnet.py`, to keep the main verified implementation independent from the bonus modifications.

The main idea was to test whether data augmentation could further improve the large patchify ConvNet. Since the best non-augmented large model still showed a large gap between training and test accuracy, random horizontal flipping was added as a dynamic training-time augmentation.

### 1. Dynamic Random Horizontal Flip

In the main Assignment 3 experiments, the training patch matrix `MX_train` was precomputed and cached. However, random horizontal flipping changes the image content at every mini-batch, so the augmented training batches cannot reuse a fixed cached `MX_train`.

Therefore, the bonus training loop uses the following dynamic pipeline for every mini-batch:

```text
X_batch -> random horizontal flip -> image tensor -> MX_batch -> forward/backward pass
```

Validation and test sets are not augmented, and their cached `MX` matrices are still used for deterministic evaluation.

### 2. Bonus Experiment Results

The large ConvNet architecture was kept the same as in Exercise 4:

* `f = 4`
* `nf = 40`
* `nh = 300`
* `lambda = 0.0025`
* `step_size_1 = 800`
* `flip_probability = 0.5`

Several bonus variants were tested.

| Experiment | Train Subset Accuracy | Validation Accuracy | Test Accuracy | Test Loss |
| :--- | ---: | ---: | ---: | ---: |
| Large ConvNet without augmentation | 99.30% | 67.80% | 66.10% | 1.0504 |
| Large ConvNet with label smoothing | 97.78% | 68.20% | 66.02% | 1.0302 |
| Random horizontal flip, 4 cycles | 88.86% | 70.10% | 69.50% | 0.8903 |
| Random horizontal flip + label smoothing | 84.26% | 70.60% | 68.78% | 0.9630 |
| Random horizontal flip, 5 cycles | 93.32% | 70.60% | **69.80%** | **0.8850** |

Random horizontal flipping gave the largest improvement. It increased the test accuracy from `66.10%` to `69.50%` with 4 cycles. Extending the flip-augmented training to 5 cycles further improved the test accuracy to `69.80%`.

### 3. Regularization and Learning Rate Search

Additional experiments were run to investigate whether the flip-augmented model could be improved further. Lower L2 regularization values caused stronger overfitting, while a larger value also reduced accuracy. Combining flip augmentation with label smoothing also did not improve the final test accuracy.

Different maximum learning rates were also tested for the 5-cycle flip setup.

| Experiment | eta_max | Validation Accuracy | Test Accuracy | Test Loss |
| :--- | ---: | ---: | ---: | ---: |
| Flip, 5 cycles | 0.100 | 70.60% | **69.80%** | **0.8850** |
| Flip, 5 cycles | 0.075 | **71.40%** | 69.72% | 0.8992 |
| Flip, 5 cycles | 0.050 | 68.00% | 68.94% | 0.9124 |

Although `eta_max = 0.075` achieved the highest validation accuracy, it did not achieve the best test accuracy. The best final test accuracy was obtained with `eta_max = 0.1`.

### 4. Bonus Visualizations

**Figure 21: Best Bonus Flip-Augmentation Loss Curves**

![Best Bonus Flip-Augmentation Loss Curves](assignment3_report_assets/bonus_flip_loss_curves.png)

**Figure 22: Compact Bonus Accuracy Comparison**

![Compact Bonus Accuracy Comparison](assignment3_report_assets/bonus_comparison_accuracy.png)

**Figure 23: Final Bonus Test Accuracy Comparison**

![Final Bonus Test Accuracy Comparison](assignment3_report_assets/bonus_final_accuracy_comparison.png)

**Figure 24: Final Bonus Test Loss Comparison**

![Final Bonus Test Loss Comparison](assignment3_report_assets/bonus_final_loss_comparison.png)



### 5. Speed Comparison with PyTorch CPU Conv2d

I also compared the training speed of the NumPy patchify ConvNet with a PyTorch CPU implementation using `torch.nn.functional.conv2d` and autograd. Each architecture was trained for 200 update steps.

| f | nf | NumPy Time | PyTorch CPU Time | PyTorch / NumPy |
|---:|---:|---:|---:|---:|
| 2 | 3 | 2.27s | 0.54s | 0.24× |
| 4 | 10 | 1.06s | 0.47s | 0.44× |
| 8 | 40 | 1.10s | 0.70s | 0.63× |
| 16 | 160 | 1.39s | 3.15s | 2.27× |

PyTorch CPU was faster for small and medium filter widths, while the NumPy implementation was faster for the largest filter width. This suggests that PyTorch's general convolution implementation is efficient for typical convolution sizes, but the custom NumPy patch-matrix implementation can be competitive for this assignment's special non-overlapping large-filter case.

![NumPy vs PyTorch CPU Training Speed](assignment3_report_assets/bonus_speed_comparison_times.png)

![Relative Speed PyTorch CPU vs NumPy](assignment3_report_assets/bonus_speed_comparison_ratio.png)

### 6. Bonus Takeaway

The best bonus model was the 5-cycle random horizontal flip model, which achieved **69.80%** test accuracy. This is an absolute improvement of **3.70 percentage points** over the large non-augmented ConvNet from Exercise 4.

The experiments show that data augmentation was more effective than label smoothing or further L2 tuning for this patchify ConvNet. The main drawback is computational cost, since the augmented patch matrix must be rebuilt dynamically for every mini-batch.

The bonus experiments show that data augmentation gave the largest accuracy improvement, while the speed comparison showed that the custom NumPy implementation can be competitive with PyTorch CPU for large non-overlapping filters.

---

## 🔧 Setup & Reproduction

### Prerequisites
* Python 3.x
* NumPy, Matplotlib, Pickle
* (Optional) PyTorch — only for gradient verification via `torch_gradient_computations.py`

### Dataset
Download the [CIFAR-10 Python version](https://www.cs.toronto.edu/~kriz/cifar.html) and place the extracted `cifar-10-batches-py` folder under `Datasets/`.

```
DD2424_Assignment1/
├── Datasets/
│   └── cifar-10-batches-py/
│       ├── data_batch_1
│       ├── ...
│       └── test_batch
├── images/
│   ├── Assignment2/
│   └── ...
├── assignment3_cache/                 # Generated MX cache, not tracked by Git
├── assignment3_outputs/               # Experiment summaries and generated parameters
├── assignment3_report_assets/         # Assignment 3 figures used in README/report
├── image_classifier.py
├── bonus_image_classifier.py
├── BCE_image_classifier.py
├── two_layer_image_classifier.py
├── bonus_two_layer_image_classifier.py
├── assignment3_convnet.py
├── assignment3_bonus_convnet.py
├── debug_info.npz
├── torch_gradient_computations.py
└── README.md
```

### Running
```bash
# Assignment 1 — Baseline (4 experiments)
python image_classifier.py

# Assignment 1 — Bonus 1 (Grid Search + Final Training)
python bonus_image_classifier.py

# Assignment 1 — Bonus 2 (BCE Loss)
python BCE_image_classifier.py

# Assignment 2 — Core (CLR + Coarse-to-Fine Search)
python two_layer_image_classifier.py

# Assignment 2 — Bonus (Adam / Augmentation / Scaling)
python bonus_two_layer_image_classifier.py

# Assignment 3 — Debug checks
python assignment3_convnet.py --action debug

# Assignment 3 — Plot existing experiment figures
python assignment3_convnet.py --action plot-short
python assignment3_convnet.py --action plot-long
python assignment3_convnet.py --action plot-ex4

# Assignment 3 — Exercise 4 experiments
python assignment3_convnet.py --action ex4-no-ls
python assignment3_convnet.py --action ex4-ls

# Assignment 3 — Bonus debug check
python assignment3_bonus_convnet.py --action debug

# Assignment 3 — Bonus sanity check for random horizontal flip
python assignment3_bonus_convnet.py --action bonus-sanity

# Assignment 3 — Bonus experiments
python assignment3_bonus_convnet.py --action bonus-flip
python assignment3_bonus_convnet.py --action bonus-flip-5cycles

# Assignment 3 — Bonus plots
python assignment3_bonus_convnet.py --action plot-bonus
```