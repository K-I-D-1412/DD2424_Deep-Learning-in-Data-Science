# DD2424 Deep Learning in Data Science

> **Note**: This project was developed as part of the *Deep Learning in Data Science* course (DD2424) at KTH Royal Institute of Technology.
> 
> **Code of Honour**: If there are similar questions or labs or projects in the future, it is the responsibility of KTH students not to copy or modify these codes, or other files because it is against the [KTH EECS Code of Honour](https://www.kth.se/en/eecs/utbildning/hederskodex). The owner of this repository doesn't take any commitment for other's faults.

## 📌 Project Overview
This repository contains the implementation of a **Single-Layer Neural Network** built entirely from scratch using NumPy. The objective of this project is to classify images from the CIFAR-10 dataset. 

The project demonstrates a fundamental understanding of neural network architecture, including forward propagation, hand-derived analytical gradient computation (backward pass), loss functions (Cross-Entropy with L2 Regularization), and the Mini-batch Gradient Descent optimization algorithm. 

The repository is structured into two main parts:
1. `image_classifier.py`: The baseline single-layer network framework.
2. `bonus_classifier.py`: An enhanced version implementing advanced training techniques to push the limits of a zero-hidden-layer architecture.

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

**Analysis & Visualizations:**

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

To maximize the performance of a simple linear classifier, several advanced techniques were implemented in `bonus_classifier.py`:

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