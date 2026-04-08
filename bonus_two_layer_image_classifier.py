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

def InitAdamStates(network):
    m = {
        'W': [np.zeros_like(network['W'][0]), np.zeros_like(network['W'][1])],
        'b': [np.zeros_like(network['b'][0]), np.zeros_like(network['b'][1])]
    }
    v = {
        'W': [np.zeros_like(network['W'][0]), np.zeros_like(network['W'][1])],
        'b': [np.zeros_like(network['b'][0]), np.zeros_like(network['b'][1])]
    }
    return m, v
    
def ApplyNetwork(X, network, keep_prob=1.0):
    W1, W2 = network['W'][0], network['W'][1]
    b1, b2 = network['b'][0], network['b'][1]
    
    # 1. Layer 1 linear calculation
    s1 = np.dot(W1, X) + b1
    
    # 2. ReLU activation function (negative numbers become 0, positive numbers remain unchanged)
    h = np.maximum(0, s1)
    
    # Dropout
    if keep_prob < 1.0:
        # Generate mask U: generate 1 with probability keep_prob, otherwise 0. Then divide by keep_prob for inverse scaling
        U = (np.random.rand(*h.shape) < keep_prob).astype(np.float32) / keep_prob
        h = h * U  # Apply mask, turn off some neurons
    else:
        U = None # Testing/evaluation phase, do not use Dropout
    
    # 3. Layer 2 linear calculation
    s = np.dot(W2, h) + b2
    
    # 4. Softmax calculation
    s_max = np.max(s, axis=0, keepdims=True)
    e_s = np.exp(s - s_max) # prevent overflow
    P = e_s / np.sum(e_s, axis=0, keepdims=True)
    
    # Must save the Dropout mask U for backpropagation
    fp_data = {'X': X, 's1': s1, 'h': h, 'U': U}
    
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

def TranslateImages(X_batch, max_trans=3):

    n_images = X_batch.shape[1]
    
    # 1. Transform (3072, n_images) to (n_images, 3, 32, 32) for spatial slicing
    X_reshaped = X_batch.T.reshape(n_images, 3, 32, 32)
    X_trans_reshaped = np.zeros_like(X_reshaped) # 背景默认补零
    
    for i in range(n_images):
        # Randomly generate x and y direction translation amounts (-3 to 3)
        tx = np.random.randint(-max_trans, max_trans + 1)
        ty = np.random.randint(-max_trans, max_trans + 1)
        
        # If there is no translation, copy directly
        if tx == 0 and ty == 0:
            X_trans_reshaped[i] = X_reshaped[i]
            continue
            
        # Calculate the slice range of the source image
        x_start_src = max(0, -tx)
        x_end_src = min(32, 32 - tx)
        y_start_src = max(0, -ty)
        y_end_src = min(32, 32 - ty)
        
        # Calculate the slice range of the destination image
        x_start_dst = max(0, tx)
        x_end_dst = min(32, 32 + tx)
        y_start_dst = max(0, ty)
        y_end_dst = min(32, 32 + ty)
        
        # Paste the effective part of the original image to the corresponding position of the new canvas
        X_trans_reshaped[i, :, y_start_dst:y_end_dst, x_start_dst:x_end_dst] = \
            X_reshaped[i, :, y_start_src:y_end_src, x_start_src:x_end_src]
            
    # 2. Flatten the processed images back to (3072, n_images)
    X_trans = X_trans_reshaped.reshape(n_images, 3072).T
    
    return X_trans

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
    U = fp_data['U']

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
    
    # Dropout: turn off the gradient of the closed neurons
    if U is not None:
        G_hidden = G_hidden * U
    
    # Gradient of loss w.r.t. s1
    G_hidden[s1 <= 0] = 0

    # Gradient of loss w.r.t. W1
    grad_W1 = (1 / n) * np.dot(G_hidden, X.T) + 2 * lam * W1
    
    # Gradient of loss w.r.t. b1
    grad_b1 = (1 / n) * np.sum(G_hidden, axis=1, keepdims=True)
    
    return {'W': [grad_W1, grad_W2], 'b': [grad_b1, grad_b2]}

def MiniBatchGD(X_train, Y_train, y_train, X_val, Y_val, y_val, GDparams, init_net, lam, augment_data=False, keep_prob=1.0, optimizer='sgd'):
    n = X_train.shape[1]
    n_batch = GDparams['n_batch']
    n_epochs = GDparams['n_epochs']
    
    if optimizer == 'sgd':
        eta_min = GDparams['eta_min']
        eta_max = GDparams['eta_max']
        n_s = GDparams['n_s']
    elif optimizer == 'adam':
        eta = GDparams['eta'] # Adam Use fixed learning rate
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        m_state, v_state = InitAdamStates(init_net)

    network = copy.deepcopy(init_net)

    # Initialize the loss history, used to plot the learning curve
    train_loss_history = []
    val_loss_history = []
    
    # Initialize 6 lists to record historical data
    train_cost_history, val_cost_history = [], []
    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []
    
    t = 0 # Global update step counter t

    if augment_data:
        flip_indices = GenerateFlipIndices()

    for epoch in range(n_epochs):
        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            
            X_batch = X_train[:, j_start:j_end]
            Y_batch = Y_train[:, j_start:j_end]
            
            if augment_data:
                X_batch = FlipImages(X_batch, flip_indices)
                X_batch = TranslateImages(X_batch, max_trans=3)
            
            # Compute the forward pass
            P_batch, fp_data = ApplyNetwork(X_batch, network, keep_prob=keep_prob)
            
            # Compute backward pass
            grads = BackwardPass(X_batch, Y_batch, P_batch, fp_data, network, lam)

            if optimizer == 'sgd':
                # SGD + CLR Update Logic
                eta_t = ComputeLearningRate(t, eta_min, eta_max, n_s)
                for l in range(2):
                    network['W'][l] -= eta_t * grads['W'][l]
                    network['b'][l] -= eta_t * grads['b'][l]
            
            elif optimizer == 'adam':
                # Adam Update Logic
                t += 1  # Adam bias correction needs t starting from 1
                for l in range(2): # Iterate through two layers
                    for param in ['W', 'b']: # Iterate through weights and biases
                        grad = grads[param][l]
                        
                        # 1. Update biased first-order moment estimate (Momentum)
                        m_state[param][l] = beta1 * m_state[param][l] + (1 - beta1) * grad
                        # 2. Update biased second-order moment estimate (RMSprop)
                        v_state[param][l] = beta2 * v_state[param][l] + (1 - beta2) * (grad ** 2)
                        
                        # 3. Compute bias-corrected estimates
                        m_hat = m_state[param][l] / (1 - beta1 ** t)
                        v_hat = v_state[param][l] / (1 - beta2 ** t)
                        
                        # 4. Update network parameters
                        network[param][l] -= eta * m_hat / (np.sqrt(v_hat) + epsilon)
            
            if optimizer == 'sgd':
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
    
    # ---------------------------------------------------------
    # 🚀 PART A: Fine Search for Adam (验证集 5000, m=200)
    # ---------------------------------------------------------
    print("\n=== PART A: Fine Search (Adam Optimizer, m=200, Augmentation=True) ===")
    X_train_f, Y_train_f, y_train_f, X_val_f, Y_val_f, y_val_f = LoadAllData(dataset_dir, val_size=5000)
    X_train_f_n, X_val_f_n, _ = PreProcess(X_train_f, X_val_f, X_test)
    
    rng = np.random.default_rng(99)
    # Adam 的正则化需求通常比较小，我们依然在 1e-5 到 1e-4 之间搜索
    l_min, l_max = -5, -4 
    
    best_val_acc = 0
    best_lam = 0
    
    for i in range(4): # 测 4 次
        l = l_min + (l_max - l_min) * rng.random()
        lam = 10 ** l
        print(f"Testing lambda = {lam:.6f} with Adam...")
        
        network = InitParameters(d=X_train_f_n.shape[0], m=200)
        
        # === 修改 1: Adam 不需要 CLR 参数，只需要固定 eta ===
        GDparams_fine = {'n_batch': 100, 'eta': 5e-4, 'n_epochs': 8}
        
        # === 修改 2: 加上 optimizer='adam', 确保 keep_prob=1.0 (关闭 Dropout) ===
        _, _, _, _, _, _, val_acc_hist = MiniBatchGD(
            X_train_f_n, Y_train_f, y_train_f, X_val_f_n, Y_val_f, y_val_f, 
            GDparams_fine, network, lam, 
            augment_data=True, keep_prob=1.0, optimizer='adam'
        )
        
        max_acc = max(val_acc_hist)
        if max_acc > best_val_acc:
            best_val_acc = max_acc
            best_lam = lam

    print(f"\n🏆 Adam Fine Search Winner: lambda = {best_lam:.6f} (Validation Acc: {best_val_acc*100:.2f}%)")
    
    # ---------------------------------------------------------
    # 🚀 PART B: Final Training with Adam
    # ---------------------------------------------------------
    print("\n=== PART B: Final Training (Adam, 20 Epochs, m=200) ===")
    X_train_final, Y_train_final, y_train_final, X_val_final, Y_val_final, y_val_final = LoadAllData(dataset_dir, val_size=1000)
    X_train_final_n, X_val_final_n, X_test_n = PreProcess(X_train_final, X_val_final, X_test)
    
    print(f"Final Training Set Size: {X_train_final_n.shape[1]}")
    
    n_epochs_final = 20 # Adam 收敛很快，20 个 Epochs 应该足够看出威力了
    
    final_network = InitParameters(d=X_train_final_n.shape[0], m=200)
    
    # 同样只传固定学习率 eta
    GDparams_final = {'n_batch': 100, 'eta': 5e-4, 'n_epochs': n_epochs_final}
    
    # 开启 Adam 优化器进行终极训练
    trained_net_final, train_cost, val_cost, train_loss, val_loss, train_acc, val_acc = MiniBatchGD(
        X_train_final_n, Y_train_final, y_train_final, 
        X_val_final_n, Y_val_final, y_val_final, 
        GDparams_final, final_network, best_lam, 
        augment_data=True, keep_prob=1.0, optimizer='adam'
    )
    
    # 测试集表现
    P_test, _ = ApplyNetwork(X_test_n, trained_net_final)
    final_test_acc = ComputeAccuracy(P_test, y_test)
    
    # === 图表标题和保存的文件名更新为 Adam ===
    os.makedirs("images/Assignment2", exist_ok=True)
    epochs_range = range(1, n_epochs_final + 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    axs[0].plot(epochs_range, train_loss, label='Training loss', color='green')
    axs[0].plot(epochs_range, val_loss, label='Validation loss', color='red')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Final Loss Plot (m=200, Adam)')
    axs[0].legend()
    
    axs[1].plot(epochs_range, train_acc, label='Training accuracy', color='green')
    axs[1].plot(epochs_range, val_acc, label='Validation accuracy', color='red')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Final Accuracy Plot (m=200, Adam)')
    axs[1].legend()
    
    plt.tight_layout()
    plt.savefig("images/Assignment2/Figure6_Final_Training_Adam.png")
    plt.close()
    
    print("\n==========================================")
    print("🎓 ADAM OPTIMIZER EXERCISE COMPLETED!")
    print(f"Network Configuration: m=200, Augmentation=True, Optimizer=Adam")
    print(f"🚀 Final Test Accuracy: {final_test_acc * 100:.2f}%")
    print("Images saved as 'Figure6_Final_Training_Adam.png'.")
    print("==========================================")