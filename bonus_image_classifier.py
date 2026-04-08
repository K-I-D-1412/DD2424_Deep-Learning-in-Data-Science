import numpy as np
import pickle
import matplotlib.pyplot as plt
import copy
import os

def LoadBatch(filename):
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    
    # 1. For X, extract image data, convert to float32 and divide by 255.0 for normalization
    X = dict[b'data'].astype(np.float32) / 255.0
    # X is (10000, 3072), each column is an image, so transpose the matrix
    X = X.T 
    
    # 2. For y, extract the label list
    y = dict[b'labels']
    
    # 3. For Y, generate One-hot encoding matrix 
    # K=10 classes, n=len(y) samples
    Y = np.zeros((10, len(y)), dtype=np.float32)
    for i, label in enumerate(y):
        Y[label, i] = 1.0 # Set the corresponding class row to 1.0
        
    return X, Y, y

def LoadAllData(dir_path):
    X_list, Y_list, y_list = [], [], []
    
    for i in range(1, 6):
        filename = os.path.join(dir_path, f"data_batch_{i}")
        X, Y, y = LoadBatch(filename)
        X_list.append(X)
        Y_list.append(Y)
        y_list.extend(y)
        
    # Concatenate matrices along the column direction (axis=1)
    # X_all shape will become (3072, 50000)
    X_all = np.concatenate(X_list, axis=1)
    Y_all = np.concatenate(Y_list, axis=1)
    
    # Split validation set (last 1000 images)
    val_size = 1000
    
    X_train = X_all[:, :-val_size]
    Y_train = Y_all[:, :-val_size]
    y_train = y_list[:-val_size]
    
    X_val = X_all[:, -val_size:]
    Y_val = Y_all[:, -val_size:]
    y_val = y_list[-val_size:]
    
    return X_train, Y_train, y_train, X_val, Y_val, y_val

def PreProcess(X_train, X_val, X_test):
    X_mean = np.mean(X_train, axis=1, keepdims=True)
    X_std = np.std(X_train, axis=1, keepdims=True)
    
    # Use the statistic from the training set to normalize all three sets
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    
    return X_train, X_val, X_test

def InitParameters(d, K=10, seed=42):
    rng = np.random.default_rng(seed)
    W = 0.01 * rng.standard_normal(size=(K, d)).astype(np.float32)
    b = np.zeros((K, 1)).astype(np.float32)
    
    return {'W': W, 'b': b}
    
def ApplyNetwork(X, network):
    W = network['W']
    b = network['b']
    
    s = np.dot(W, X) + b
    
    # Softmax function
    s_max = np.max(s, axis=0, keepdims=True)
    e_s = np.exp(s - s_max) # The softmax function is invariant to the shift s -> s - c for any constant c, avoiding overflow
    
    P = e_s / np.sum(e_s, axis=0, keepdims=True)
    
    return P

def ComputeLoss(P, y, network, lam):
    W = network['W']
    n = P.shape[1]
    
    # Use numpy array indexing to get the correct probabilities
    y_arr = np.array(y)
    correct_prob = P[y_arr, np.arange(n)]
    
    # Compute the cross-entropy loss
    loss_cross = np.mean(-np.log(correct_prob + 1e-10)) # Add small epsilon to prevent log(0)
    
    # Compute the regularization loss
    loss_reg = lam * np.sum(W * W)
    
    # Total loss
    J = loss_cross + loss_reg
    
    return J

def ComputeAccuracy(P, y):
    y_pred = np.argmax(P, axis=0)
    accuracy = np.mean(y_pred == y)
    
    return accuracy

def BackwardPass(X, Y, P, network, lam):
    W = network['W']
    n = X.shape[1]
    
    # Gradient of loss w.r.t. s
    G = P - Y # We have proved in lecture that dJ/ds = P - Y
    
    # Gradient of loss w.r.t. W
    grad_W = (1 / n) * np.dot(G, X.T) + 2 * lam * W
    
    # Gradient of loss w.r.t. b
    grad_b = (1 / n) * np.sum(G, axis=1, keepdims=True)
    
    return {'W': grad_W, 'b': grad_b}

def MiniBatchGD(X_train, Y_train, y_train, X_val, Y_val, y_val, GDparams, init_net, lam, augment_data=False, step_decay=False):
    n = X_train.shape[1]
    n_batch = GDparams['n_batch']
    n_epochs = GDparams['n_epochs']
    eta = GDparams['eta']

    network = copy.deepcopy(init_net)

    # Initialize the loss history, used to plot the learning curve
    train_loss_history = []
    val_loss_history = []

    # Pre-compute flip indices if data augmentation is enabled
    if augment_data:
        flip_indices = GenerateFlipIndices()
    
    for epoch in range(n_epochs):

        if step_decay and epoch > 0 and epoch % 10 == 0:
            eta = eta * 0.1
            print(f"Decline learning rate, current Epoch {epoch+1}, eta: {eta}")

        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            
            X_batch = X_train[:, j_start:j_end]
            Y_batch = Y_train[:, j_start:j_end]

            # Apply data augmentation (random horizontal flip) if enabled
            if augment_data:
                X_batch = FlipImages(X_batch, flip_indices)
            
            # Compute the forward pass
            P_batch = ApplyNetwork(X_batch, network)
            
            # Compute backward pass
            grads = BackwardPass(X_batch, Y_batch, P_batch, network, lam)
            
            # Update the parameters
            network['W'] -= eta * grads['W']
            network['b'] -= eta * grads['b']
            
        # Calculate the cost
        P_train = ApplyNetwork(X_train, network)
        train_cost = ComputeLoss(P_train, y_train, network, lam)
        train_loss_history.append(train_cost)

        P_val = ApplyNetwork(X_val, network)
        val_cost = ComputeLoss(P_val, y_val, network, lam)
        val_loss_history.append(val_cost)

        print(f"Epoch {epoch+1:02d}/{n_epochs} | Train Cost: {train_cost:.4f} | Val Cost: {val_cost:.4f}")

    return network, train_loss_history, val_loss_history
    
def PlotLoss(train_loss, val_loss, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Training loss', color='green')
    plt.plot(val_loss, label='Validation loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(filename)
    plt.close()

def VisualizeWeights(W, filename):
    # Reshape the weight matrix W into a 3D tensor of shape (32, 32, 3, 10)
    Ws = W.T.reshape((32, 32, 3, 10), order='F')
    # Transpose the tensor to shape (32, 32, 3, 10)
    W_im = np.transpose(Ws, (1, 0, 2, 3))
    
    # CIFAR-10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    fig, axs = plt.subplots(1, 10, figsize=(15, 3))
    for i in range(10):
        w_im = W_im[:, :, :, i]
        # Normalize to 0~1 for matplotlib display
        w_im_norm = (w_im - np.min(w_im)) / (np.max(w_im) - np.min(w_im))
        
        axs[i].imshow(w_im_norm)
        axs[i].axis('off')
        axs[i].set_title(classes[i], fontsize=10)
        
    plt.suptitle("Learnt Weight Matrices as Images", fontsize=14)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def PlotConfidenceHistograms(P, y, title, filename):
    y_arr = np.array(y)
    n = P.shape[1]
    
    # 1. Extract the predicted probabilities for the ground truth class
    gt_probs = P[y_arr, np.arange(n)]
    
    # 2. Determine which samples were predicted correctly and which were predicted incorrectly
    predictions = np.argmax(P, axis=0)
    correct_mask = (predictions == y_arr)
    
    # 3. Separate the probabilities
    probs_correct = gt_probs[correct_mask]
    probs_incorrect = gt_probs[~correct_mask]
    
    # 4. Plot overlapping histograms
    plt.figure(figsize=(10, 5))
    plt.hist(probs_correct, bins=30, alpha=0.7, color='green', label='Correctly Classified')
    plt.hist(probs_incorrect, bins=30, alpha=0.7, color='red', label='Incorrectly Classified')
    
    plt.xlabel('Probability of Ground Truth Class')
    plt.ylabel('Number of Examples')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(filename)
    plt.close()

def GenerateFlipIndices():
    aa = np.arange(32).reshape((32, 1))
    bb = np.arange(31, -1, -1).reshape((32, 1))
    vv = np.tile(32 * aa, (1, 32))
    ind_flip = vv.reshape((1024, 1)) + np.tile(bb, (32, 1))
    
    # Combine RGB three channels index
    inds_flip = np.vstack((ind_flip, 1024 + ind_flip, 2048 + ind_flip))
    # Flatten into a one-dimensional array for matrix indexing
    return inds_flip.flatten()

def FlipImages(X_batch, flip_indices):
    n_images = X_batch.shape[1]
    # Generate a boolean array with approximately 50% True values
    flip_mask = np.random.rand(n_images) < 0.5
    
    # Deep copy the data to avoid modifying the original dataset
    X_flipped = np.copy(X_batch)
    
    # Use numpy advanced indexing:
    # X_batch[flip_indices, :] will shuffle the rows of the image into flipped order
    # [:, flip_mask] ensures that we only assign the flipped data to the selected 50% of images
    X_flipped[:, flip_mask] = X_batch[flip_indices, :][:, flip_mask]
    
    return X_flipped

if __name__ == "__main__":
    import os
    import json
    
    # ---------------------------------------------------------
    # Options:
    # Set to True to perform Grid Search and save the best parameters.
    # Set to False to skip Grid Search and load the saved parameters.
    # ---------------------------------------------------------
    PERFORM_GRID_SEARCH = True
    params_file = "bonus_best_params.json"
    
    print("[Bonus 1] Loading the full dataset...")
    dataset_dir = "Datasets/cifar-10-batches-py"
    X_train, Y_train, y_train, X_val, Y_val, y_val = LoadAllData(dataset_dir)
    X_test, Y_test, y_test = LoadBatch(os.path.join(dataset_dir, "test_batch"))
    
    X_train_norm, X_val_norm, X_test_norm = PreProcess(X_train, X_val, X_test)
    
    best_params = {}
    
    if PERFORM_GRID_SEARCH or not os.path.exists(params_file):
        if not PERFORM_GRID_SEARCH:
            print(f"Warning: {params_file} not found. Falling back to Grid Search.")
            
        print("\nPreparing for grid search...")
        # Define candidate hyperparameter ranges for grid search.
        grid_lams = [0.0, 0.001, 0.01]
        grid_etas = [0.01, 0.001]
        grid_batches = [100, 500]
        
        # Since we evaluate multiple combinations on the full 50,000 dataset,
        # 15 epochs per combination is sufficient to identify the best hyperparameters.
        search_epochs = 15
        
        total_experiments = len(grid_batches) * len(grid_etas) * len(grid_lams)
        
        best_val_acc = -1.0
        best_network = None
        
        print(f"\nStarting Grid Search with {total_experiments} combinations...")
        experiment_count = 1
        
        for batch_size in grid_batches:
            for eta in grid_etas:
                for lam in grid_lams:
                    print(f"\n▶ [Combination {experiment_count}/{total_experiments}] batch={batch_size}, eta={eta}, lam={lam}")
                    
                    # Network must be re-initialized for each experiment!
                    network = InitParameters(d=X_train_norm.shape[0])
                    GDparams = {'n_batch': batch_size, 'eta': eta, 'n_epochs': search_epochs}
                    
                    # Train (enable data augmentation, disable step decay)
                    trained_net, _, _ = MiniBatchGD(
                        X_train_norm, Y_train, y_train, X_val_norm, Y_val, y_val, 
                        GDparams, network, lam, augment_data=True, step_decay=False
                    )
                    
                    # Evaluate the score of the current combination using the Validation Set
                    P_val = ApplyNetwork(X_val_norm, trained_net)
                    val_acc = ComputeAccuracy(P_val, y_val)
                    print(f"Current combination Validation Accuracy: {val_acc * 100:.2f}%")
                    
                    # Save the parameters if it breaks the highest score record
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_params = {'n_batch': batch_size, 'eta': eta, 'lam': lam}
                        best_network = copy.deepcopy(trained_net)
                    
                    experiment_count += 1
                    
        print(f"\nGrid Search completed!")
        print(f"Best configuration found: {best_params}")
        print(f"Highest Validation Accuracy: {best_val_acc * 100:.2f}%")
        
        # Save best parameters to a local file
        with open(params_file, 'w') as f:
            json.dump(best_params, f, indent=4)
        print(f"[*] Saved the best parameters to {params_file}")
        
    else:
        # Load the saved parameters
        print(f"\n[*] Skipping grid search. Loading parameters from {params_file}...")
        with open(params_file, 'r') as f:
            best_params = json.load(f)
        print(f"Loaded configuration: {best_params}")

    print("\n=======================================================")
    print("Starting final deep training with the best configuration...")
    print("=======================================================")
    
    # Re-initialize a fresh network for the final training
    final_network = InitParameters(d=X_train_norm.shape[0])
    
    # Set hyper-parameters for the final deep run
    final_GDparams = {
        'n_batch': best_params['n_batch'], 
        'eta': best_params['eta'], 
        'n_epochs': 50  # Increased to 50 epochs for the final test
    }
    
    # Run MiniBatchGD with step_decay enabled
    final_trained_net, train_history, val_history = MiniBatchGD(
        X_train_norm, Y_train, y_train, X_val_norm, Y_val, y_val, 
        final_GDparams, final_network, best_params['lam'], 
        augment_data=True, 
        step_decay=False
    )
    
    # Evaluate the final model on the test set
    P_test = ApplyNetwork(X_test_norm, final_trained_net)
    final_test_acc = ComputeAccuracy(P_test, y_test)
    PlotConfidenceHistograms(P_test, y_test, "Softmax Confidence Histogram", "images/Softmax_Histogram.png")
    print(f"\n Final deep training completed!")
    print(f"Using the best configuration, final Test Accuracy: {final_test_acc * 100:.2f}%")
    
    os.makedirs("images", exist_ok=True)
    PlotLoss(train_history, val_history, "images/Bonus1_Final_Loss.png")
    VisualizeWeights(final_trained_net['W'], "images/Bonus1_Final_Weights.png")
    print("Saved final training loss curves and weight visualizations to the 'images' directory.")