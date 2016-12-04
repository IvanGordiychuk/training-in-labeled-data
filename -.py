def stochastic_gradient_step(X, y, w, train_ind, eta=0.01):
    grad0 = X[train_ind][0]*(np.dot(w,X[train_ind])-y[train_ind])
    grad1 = X[train_ind][1]*(np.dot(w,X[train_ind])-y[train_ind])
    grad2 = X[train_ind][2]*(np.dot(w,X[train_ind])-y[train_ind])
    grad3 = X[train_ind][3]*(np.dot(w,X[train_ind])-y[train_ind])
    return w - (eta * np.array((grad0, grad1, grad2, grad3))).reshape(4)

def stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e4,
                                min_weight_dist=1e-8, seed=42, verbose=False):
    weight_dist = np.inf
    w = w_init
    errors = []
    iter_num = 0
    np.random.seed(seed)
        
    while weight_dist > min_weight_dist and iter_num < max_iter:
        random_ind = np.random.randint(X.shape[0])
        w2 = stochastic_gradient_step(X, y, w, random_ind, eta=0.01)
        weight_dist = np.linalg.norm(w2-w)
        iter_num+=1
        w=w2
        errors.append(np.linalg.norm(w-norm_eq_weights.reshape(4)))
    return w,errors