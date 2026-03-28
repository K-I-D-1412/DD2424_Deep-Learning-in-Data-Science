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

def MiniBatchGD(X_train, Y_train, y_train, X_val, Y_val, y_val, GDparams, init_net, lam):
    n = X_train.shape[1]
    n_batch = GDparams['n_batch']
    n_epochs = GDparams['n_epochs']
    eta = GDparams['eta']

    network = copy.deepcopy(init_net)

    # Initialize the loss history, used to plot the learning curve
    train_loss_history = []
    val_loss_history = []
    
    for epoch in range(n_epochs):
        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            
            X_batch = X_train[:, j_start:j_end]
            Y_batch = Y_train[:, j_start:j_end]
            
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

if __name__ == "__main__":
    # 1. Load all the data
    X_train, Y_train, y_train = LoadBatch("Datasets/cifar-10-batches-py/data_batch_1")
    X_val, Y_val, y_val = LoadBatch("Datasets/cifar-10-batches-py/data_batch_2")
    X_test, Y_test, y_test = LoadBatch("Datasets/cifar-10-batches-py/test_batch")
    
    # 2. Preprocess
    X_train_norm, X_val_norm, X_test_norm = PreProcess(X_train, X_val, X_test)
    
    experiments = [
        {'lam': 0,   'n_epochs': 40, 'n_batch': 100, 'eta': 0.1},
        {'lam': 0,   'n_epochs': 40, 'n_batch': 100, 'eta': 0.001},
        {'lam': 0.1, 'n_epochs': 40, 'n_batch': 100, 'eta': 0.001},
        {'lam': 1,   'n_epochs': 40, 'n_batch': 100, 'eta': 0.001}
    ]

    os.makedirs("images", exist_ok=True)

    for i, exp in enumerate(experiments):
        print(f"\n--- Running Experiment {i+1}: lambda={exp['lam']}, eta={exp['eta']}, batch={exp['n_batch']} ---")
        
        # 3. Initialize network
        network = InitParameters(d=X_train_norm.shape[0])
        
        # 4. Set training parameters
        GDparams = {
            'n_batch': exp['n_batch'],
            'eta': exp['eta'],
            'n_epochs': exp['n_epochs']
        }
        lam = exp['lam']
        
        print("--- Starting training ---")
        # 5. Start training
        trained_net, train_history, val_history = MiniBatchGD(
            X_train_norm, Y_train, y_train, 
            X_val_norm, Y_val, y_val, 
            GDparams, network, lam
        )
        
        # 6. Compute the final accuracy on the test set
        P_test = ApplyNetwork(X_test_norm, trained_net)
        test_acc = ComputeAccuracy(P_test, y_test)

        # 7. Plot the loss curve
        loss_filename = f"images/loss_lam{lam}_eta{GDparams['eta']}_batch{GDparams['n_batch']}.png"
        PlotLoss(train_history, val_history, loss_filename)
        
        # 8. Visualize the learned weight templates
        weights_filename = f"images/weights_lam{lam}_eta{GDparams['eta']}_batch{GDparams['n_batch']}.png"
        VisualizeWeights(trained_net['W'], weights_filename)
        
        print(f"Experiment {i+1} completed! Final accuracy on the test set: {test_acc * 100:.2f}%. Saved plots.")