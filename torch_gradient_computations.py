import torch
import numpy as np

def ComputeGradsWithTorch(X, y, network_params):
    
    # 强制将 numpy 数组转换为 float32 类型的 tensor，避免类型不匹配
    Xt = torch.tensor(X, dtype=torch.float32)

    L = len(network_params['W'])

    # will be computing the gradient w.r.t. these parameters    
    W = [None] * L
    b = [None] * L    
    for i in range(len(network_params['W'])):
        W[i] = torch.tensor(network_params['W'][i], dtype=torch.float32, requires_grad=True)
        b[i] = torch.tensor(network_params['b'][i], dtype=torch.float32, requires_grad=True)        

    ## give informative names to these torch classes        
    apply_relu = torch.nn.ReLU()
    apply_softmax = torch.nn.Softmax(dim=0)

    #### BEGIN your code ###########################
    
    # Eq (1): s1 = W1 * X + b1
    s1 = torch.mm(W[0], Xt) + b[0]
    
    # Eq (2): h = max(0, s1)
    h = apply_relu(s1)
    
    # Eq (3): s = W2 * h + b2
    scores = torch.mm(W[1], h) + b[1]

    #### END of your code ###########################            

    # apply SoftMax to each column of scores     
    P = apply_softmax(scores)
    
    # compute the loss
    n = X.shape[1]
    loss = torch.mean(-torch.log(P[y, np.arange(n)]))
    
    # compute the backward pass relative to the loss and the named parameters 
    loss.backward()

    # extract the computed gradients and make them numpy arrays 
    grads = {}
    grads['W'] = [None] * L
    grads['b'] = [None] * L
    for i in range(L):
        grads['W'][i] = W[i].grad.numpy()
        grads['b'][i] = b[i].grad.numpy()

    return grads