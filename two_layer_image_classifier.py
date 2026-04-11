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

def LoadAllData(dir_path, val_size=5000):
    X_list, Y_list, y_list = [], [], []
    
    for i in range(1, 6):
        filename = os.path.join(dir_path, f"data_batch_{i}")
        X, Y, y = LoadBatch(filename)
        X_list.append(X)
        Y_list.append(Y)
        y_list.extend(y)
        
    X_all = np.concatenate(X_list, axis=1)
    Y_all = np.concatenate(Y_list, axis=1)
    
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

def InitParameters(d, m=50, K=10, seed=42):
    rng = np.random.default_rng(seed)

    # Layer 1: From input to hidden layer
    # Standard deviation set to 1/sqrt(d)
    W1 = rng.standard_normal(size=(m, d)).astype(np.float32) / np.sqrt(d)
    b1 = np.zeros((m, 1)).astype(np.float32)

    # Layer 2: From hidden layer to output layer
    # Standard deviation set to 1/sqrt(m)
    W2 = rng.standard_normal(size=(K, m)).astype(np.float32) / np.sqrt(m)
    b2 = np.zeros((K, 1)).astype(np.float32)
    
    return {'W': [W1, W2], 'b': [b1, b2]}
    
def ApplyNetwork(X, network):
    W1, W2 = network['W'][0], network['W'][1]
    b1, b2 = network['b'][0], network['b'][1]
    
    # 1. Layer 1 linear calculation
    s1 = np.dot(W1, X) + b1
    
    # 2. ReLU activation function (negative numbers become 0, positive numbers remain unchanged)
    h = np.maximum(0, s1)
    
    # 3. Layer 2 linear calculation
    s = np.dot(W2, h) + b2
    
    # 4. Softmax calculation
    s_max = np.max(s, axis=0, keepdims=True)
    e_s = np.exp(s - s_max) # prevent overflow
    P = e_s / np.sum(e_s, axis=0, keepdims=True)
    
    fp_data = {'X': X, 's1': s1, 'h': h}
    
    return P, fp_data

def ComputeLoss(P, y, network, lam):
    W1, W2 = network['W'][0], network['W'][1]
    n = P.shape[1]
    
    # Use numpy array indexing to get the correct probabilities
    y_arr = np.array(y)
    correct_prob = P[y_arr, np.arange(n)]
    
    # Compute the cross-entropy loss
    loss_cross = np.mean(-np.log(correct_prob + 1e-10)) # Add small epsilon to prevent log(0)
    
    # Compute the regularization loss
    loss_reg = lam * (np.sum(W1 * W1) + np.sum(W2 * W2))
    
    # Cost
    J = loss_cross + loss_reg
    
    return J, loss_cross  # Return Cost and Loss

def ComputeAccuracy(P, y):
    y_pred = np.argmax(P, axis=0)
    accuracy = np.mean(y_pred == y)
    
    return accuracy

def ComputeLearningRate(t, eta_min, eta_max, n_s):
    # Compute the current epoch number
    l = t // (2 * n_s)

    # Determine if it is in the rising or falling phase and apply the corresponding formula
    if t >= 2 * l * n_s and t <= (2 * l + 1) * n_s:
        # Rising phase
        eta_t = eta_min + ((t - 2 * l * n_s) / n_s) * (eta_max - eta_min)
    else:
        # Falling phase
        eta_t = eta_max - ((t - (2 * l + 1) * n_s) / n_s) * (eta_max - eta_min)
        
    return eta_t
    

def BackwardPass(X, Y, P, fp_data,network, lam):
    W1, W2 = network['W'][0], network['W'][1]
    h = fp_data['h']
    s1 = fp_data['s1']
    
    n = X.shape[1] # number of samples

    # layer 2:
    # Gradient of loss w.r.t. s
    G = P - Y # We have proved in lecture that dJ/ds = P - Y
    
    # Gradient of loss w.r.t. W
    grad_W2 = (1 / n) * np.dot(G, h.T) + 2 * lam * W2
    
    # Gradient of loss w.r.t. b
    grad_b2 = (1 / n) * np.sum(G, axis=1, keepdims=True)

    # layer 1:
    # Gradient of loss w.r.t. h
    G_hidden = np.dot(W2.T, G)
    
    # Gradient of loss w.r.t. s1
    G_hidden[s1 <= 0] = 0

    # Gradient of loss w.r.t. W1
    grad_W1 = (1 / n) * np.dot(G_hidden, X.T) + 2 * lam * W1
    
    # Gradient of loss w.r.t. b1
    grad_b1 = (1 / n) * np.sum(G_hidden, axis=1, keepdims=True)
    
    return {'W': [grad_W1, grad_W2], 'b': [grad_b1, grad_b2]}

def MiniBatchGD(X_train, Y_train, y_train, X_val, Y_val, y_val, GDparams, init_net, lam):
    n = X_train.shape[1]
    n_batch = GDparams['n_batch']
    n_epochs = GDparams['n_epochs']
    eta_min = GDparams['eta_min']
    eta_max = GDparams['eta_max']
    n_s = GDparams['n_s']

    network = copy.deepcopy(init_net)

    # Initialize the loss history, used to plot the learning curve
    train_loss_history = []
    val_loss_history = []
    
    # Initialize 6 lists to record historical data
    train_cost_history, val_cost_history = [], []
    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []
    
    t = 0 # Global update step counter t

    for epoch in range(n_epochs):
        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            
            X_batch = X_train[:, j_start:j_end]
            Y_batch = Y_train[:, j_start:j_end]
            
            # Compute the forward pass
            P_batch, fp_data = ApplyNetwork(X_batch, network)
            
            # Compute backward pass
            grads = BackwardPass(X_batch, Y_batch, P_batch, fp_data, network, lam)

            eta_t = ComputeLearningRate(t, eta_min, eta_max, n_s)
            
            # Update the parameters
            network['W'][0] -= eta_t * grads['W'][0]
            network['b'][0] -= eta_t * grads['b'][0]
            network['W'][1] -= eta_t * grads['W'][1]
            network['b'][1] -= eta_t * grads['b'][1]

            t += 1
            
        # Calculate the cost
        P_train, _ = ApplyNetwork(X_train, network)
        train_cost, train_loss = ComputeLoss(P_train, y_train, network, lam)
        train_acc = ComputeAccuracy(P_train, y_train)
        
        train_cost_history.append(train_cost)
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)

        P_val, _ = ApplyNetwork(X_val, network)
        val_cost, val_loss = ComputeLoss(P_val, y_val, network, lam)
        val_acc = ComputeAccuracy(P_val, y_val)
        
        val_cost_history.append(val_cost)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(f"Epoch {epoch+1:02d} | Train Cost: {train_cost:.4f} | Val Acc: {val_acc*100:.2f}%")

    return network, train_cost_history, val_cost_history, train_loss_history, val_loss_history, train_acc_history, val_acc_history
    
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

if __name__ == "__main__":
    dataset_dir = "Datasets/cifar-10-batches-py"
    X_test, Y_test, y_test = LoadBatch(os.path.join(dataset_dir, "test_batch"))
    
    # 1: Fine Search (validation set 5000)
    print("\n=== PART A: Fine Search ===")
    X_train_f, Y_train_f, y_train_f, X_val_f, Y_val_f, y_val_f = LoadAllData(dataset_dir, val_size=5000)
    X_train_f_n, X_val_f_n, _ = PreProcess(X_train_f, X_val_f, X_test)
    
    rng = np.random.default_rng(99)
    # Search range: 1e-5 to 1e-4
    l_min, l_max = -5, -4 
    n_s_fine = 2 * (X_train_f_n.shape[1] // 100) # 900
    
    best_val_acc = 0
    best_lam = 0
    
    for i in range(4): 
        l = l_min + (l_max - l_min) * rng.random()
        lam = 10 ** l
        print(f"Testing lambda = {lam:.6f}...")
        
        network = InitParameters(d=X_train_f_n.shape[0])
        GDparams_fine = {'n_batch': 100, 'eta_min': 1e-5, 'eta_max': 1e-1, 'n_s': n_s_fine, 'n_epochs': 8}
        
        _, _, _, _, _, _, val_acc_hist = MiniBatchGD(
            X_train_f_n, Y_train_f, y_train_f, X_val_f_n, Y_val_f, y_val_f, 
            GDparams_fine, network, lam
        )
        
        max_acc = max(val_acc_hist)
        if max_acc > best_val_acc:
            best_val_acc = max_acc
            best_lam = lam

    print(f"\n Fine Search Winner: lambda = {best_lam:.6f} (Validation Acc: {best_val_acc*100:.2f}%)")
    
    # 2: Final Training (validation set 1000)
    print("\n=== PART B: Final Training (3 Cycles) ===")
    X_train_final, Y_train_final, y_train_final, X_val_final, Y_val_final, y_val_final = LoadAllData(dataset_dir, val_size=1000)
    X_train_final_n, X_val_final_n, X_test_n = PreProcess(X_train_final, X_val_final, X_test)
    
    print(f"Final Training Set Size: {X_train_final_n.shape[1]}")
    
    n_s_final = 2 * (X_train_final_n.shape[1] // 100) # 980
    n_epochs_final = 12 # 3 cycles * 4 epochs/cycle = 12 epochs
    
    final_network = InitParameters(d=X_train_final_n.shape[0])
    GDparams_final = {'n_batch': 100, 'eta_min': 1e-5, 'eta_max': 1e-1, 'n_s': n_s_final, 'n_epochs': n_epochs_final}
    
    trained_net_final, train_cost, val_cost, train_loss, val_loss, train_acc, val_acc = MiniBatchGD(
        X_train_final_n, Y_train_final, y_train_final, 
        X_val_final_n, Y_val_final, y_val_final, 
        GDparams_final, final_network, best_lam
    )
    
    P_test, _ = ApplyNetwork(X_test_n, trained_net_final)
    final_test_acc = ComputeAccuracy(P_test, y_test)
    
    os.makedirs("images/Assignment2", exist_ok=True)
    epochs_range = range(1, n_epochs_final + 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    axs[0].plot(epochs_range, train_loss, label='Training loss', color='green')
    axs[0].plot(epochs_range, val_loss, label='Validation loss', color='red')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Final Loss Plot')
    axs[0].legend()
    
    axs[1].plot(epochs_range, train_acc, label='Training accuracy', color='green')
    axs[1].plot(epochs_range, val_acc, label='Validation accuracy', color='red')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Final Accuracy Plot')
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig("images/Assignment2/Figure5_Final_Training.png")
    plt.close()
    
    print("\n==========================================")
    print(" ASSIGNMENT 2 CORE EXERCISES COMPLETED!")
    print(f"Final Test Accuracy: {final_test_acc * 100:.2f}%")
    print("==========================================")