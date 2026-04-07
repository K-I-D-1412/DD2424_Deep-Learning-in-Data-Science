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
    
    # Total loss
    J = loss_cross + loss_reg
    
    return J

def ComputeAccuracy(P, y):
    y_pred = np.argmax(P, axis=0)
    accuracy = np.mean(y_pred == y)
    
    return accuracy

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
            P_batch, fp_data = ApplyNetwork(X_batch, network)
            
            # Compute backward pass
            grads = BackwardPass(X_batch, Y_batch, P_batch, fp_data, network, lam)
            
            # Update the parameters
            network['W'][0] -= eta * grads['W'][0]
            network['b'][0] -= eta * grads['b'][0]
            network['W'][1] -= eta * grads['W'][1]
            network['b'][1] -= eta * grads['b'][1]
            
        # Calculate the cost
        P_train, _ = ApplyNetwork(X_train, network)
        train_cost = ComputeLoss(P_train, y_train, network, lam)
        train_loss_history.append(train_cost)

        P_val, _ = ApplyNetwork(X_val, network)
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
    # 1. 加载数据
    print("Loading data...")
    X_train, Y_train, y_train = LoadBatch("Datasets/cifar-10-batches-py/data_batch_1")
    X_val, Y_val, y_val = LoadBatch("Datasets/cifar-10-batches-py/data_batch_2")
    X_test, Y_test, y_test = LoadBatch("Datasets/cifar-10-batches-py/test_batch")
    
    # 2. 数据预处理 (注意这里使用小写 m 表示隐藏层节点数)
    X_train_norm, X_val_norm, X_test_norm = PreProcess(X_train, X_val, X_test)
    
    # ---------------------------------------------------------
    # 🚀 SANITY CHECK: 100 张图片的过拟合测试
    # ---------------------------------------------------------
    print("\n--- Starting Sanity Check (Overfitting on 100 images) ---")
    
    # 截取前 100 张图片作为我们的小型训练集
    n_small = 100
    X_small = X_train_norm[:, :n_small]
    Y_small = Y_train[:, :n_small]
    y_small = y_train[:n_small]
    
    # 初始化网络 (d = 3072)
    network = InitParameters(d=X_small.shape[0])
    
    # 设置测试专用的超参数
    GDparams_sanity = {
        'n_batch': 100,      # 把这 100 张图作为一个完整的 Batch 一次性吃进去
        'eta': 0.1,          # 给一个相对较大的学习率，让它快速下降
        'n_epochs': 200      # 跑 200 圈，确保它有时间“死记硬背”
    }
    lam_sanity = 0           # 核心：关闭正则化！
    
    # 开始训练！(为了不报错，验证集我们也直接塞这 100 张图进去，反正我们只看训练集表现)
    trained_net, train_history, val_history = MiniBatchGD(
        X_small, Y_small, y_small, 
        X_small, Y_small, y_small, 
        GDparams_sanity, network, lam_sanity
    )
    
    # 画出 Loss 曲线
    os.makedirs("images/Assignment2", exist_ok=True)
    PlotLoss(train_history, val_history, "images/Assignment2/sanity_check_loss.png")
    
    print("\nSanity Check Completed!")
    print(f"Final Train Cost after 200 epochs: {train_history[-1]:.4f}")
    print("Please check 'images/Assignment2/sanity_check_loss.png'. The green line should drop close to 0.")