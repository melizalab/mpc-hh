import numpy as np
from sklearn.cluster import KMeans

'''
It may be asked, "why make an RBFN from scratch instead of using PyTorch (or something similar)?"
When I started this project, the do-mpc package could not use PyTorch models so I had to use only numpy.
During the revision stage, do-mpc did update to allow PyTorch, but (at least at the time) it was slow when
making predictions with PyTorch models.
'''


def time_delay_embedding(x, D_e, tau):
    '''
    Returns a time-delay embedded matrix (TDE) of input vector x with embedding dimension D_e and time delay tau
    Can return errors if impossible values of D_e and tau are specified
    '''
    TDE = np.zeros((len(x) - tau * (D_e - 1), D_e))  # dimensions of TDE
    # find intervals of input vector x that construct the TDE one column at a time
    for i in range(D_e):
        LB = len(x) - i * tau - TDE.shape[0]
        UB = len(x) - i * tau
        TDE[:, i] = x[LB:UB]
    return TDE

def RBF(state, centers, scale):
    diff = state - centers
    distance = np.sum(diff ** 2, axis=1)
    return np.exp(-scale * distance)


def choose_centers(TDE, n_centers, method='kmeans',random_state=123):
    '''
    n_centers: number of centers to choose (i.e. number of RBFs in network)
    method: how to choose centers ('kmeans' or 'sample')
    '''
    if method == 'sample':
        indxs = np.random.choice(TDE.shape[0], n_centers, replace=False)
        centers = TDE[indxs]
    elif method == 'kmeans':
        centers = KMeans(n_clusters=n_centers,random_state=random_state,n_init='auto').fit(TDE).cluster_centers_
    elif method == 'upsample':
        cutoff_value = -20
        non_spiking = np.where(TDE[:,0]<cutoff_value)[0]
        spiking = np.where(TDE[:,0]>cutoff_value)[0]
        non_spiking_indxs = np.random.choice(non_spiking,int(.5*n_centers),replace = False)
        spiking_indxs = np.random.choice(spiking,int(.5*n_centers),replace = False)
        centers = np.concatenate([TDE[spiking_indxs],TDE[non_spiking_indxs]])
    return centers

def regression_matrices(v, I, D_e, tau, n_centers, scale, center_method='kmeans'):
    '''
    This function returns Y and X matrices used for estimate RBF-DDF parameters.
    By default it does not return the time-delay embedded matrix (TDE) of inputs v and I nor the centers used in the RBFs,
    however this can be changed by setting the retrun_TDE and return_centers arguments to True respectively.

    Arguments
    v: observed state time series that you want to predict
    I: external forcing to state variable v
    D_e: embedding dimension of TDE
    tau: time delay
    n_centers: number of centers to use in RBF network
    scale: scale of the RBFs
    center_method: method used to pick centers from data ('kmeans' or 'sample')
    '''
    # Create TDE matrix
    TDE = time_delay_embedding(v, D_e, tau)

    # Y is the difference between V(n) and V(n-1)
    Y = TDE[:, 0][1:] - TDE[:, 0][:-1]

    # Get centers for RBFs, returned object has dimensions n_centers x D_e
    centers = choose_centers(TDE, n_centers, center_method)

    # X has as many rows as Y and has n_centers+1 columns with the extra column for input scaling parameter alpha
    X = np.zeros((len(Y), n_centers + 1))
    for i in range(X.shape[0]):
        state = TDE[i]
        X[i, :-1] = RBF(state, centers, scale)
        # for j in range(n_centers):
        #    state = TDE[i]
        #    X[i,j] = RBF(state,centers[j,:],scale)
    # Get input I at proper time indicies for building X matrix
    I_offset = len(I) - X.shape[0]
    I_np1 = I[I_offset:]  # I(n+1)
    I_n = I[I_offset - 1:-1]  # I(n)
    # Add to final column of X
    X[:, -1] = I_np1 + I_n
    return Y, X, TDE, centers


def S_n(v, D_e, tau):
    '''
    For a given input vector v, it returns a time-delay embedding vector S(n) with dimension D_e and delay tau.

    This function assumes that the tau_a used in each embedding follows the standard practice of
    tau_a = (a-1)*tau, a = 1,2,...,D_e

    Its intended use is to give an embedding of the last element in the input v
    (i.e. if v has length 4 it returns S(4))
    '''
    time_indxs = []
    for a in range(1, D_e + 1):
        time_indxs.append((a - 1) * tau)
    # Flip vector to be in alignment of how X design matrix was constructed
    return np.flip([v[n] for n in time_indxs])


def predict_future(v, I, tau, D_e, centers, weights,scale):
    '''
    Set up prediction alogrithm
    w: RBF network weights
    alpha: coefficient for sum of inputs I(n+1) and I(n)
    I_sum: I(n+1)+I(n)
    RETURNS:
    v_hat: vector of predicted states v, initialized with observations of v
    '''
    v_hat = v[:tau * (D_e)]
    w = weights[:-1]
    alpha = weights[-1]
    n_centers = len(centers)
    I_sum = I[tau * D_e:len(v)] + I[tau * D_e - 1:len(v) - 1]

    # Predict for time length such that the length of v_hat equals v

    for i in range(len(v) - tau * D_e):
        offset = tau * D_e
        state_embedding = S_n(v_hat[i:tau * D_e + i], D_e, tau)
        Psi = RBF(state_embedding, centers,scale)
        v_predict = v_hat[-1] + np.dot(w, Psi) + alpha * I_sum[i]
        v_hat = np.append(v_hat, [v_predict], axis=0)
    return v_hat


def single_step_prediction(v_n,v_d1,i_now,i_future,centers,weights,alpha,scale):
    '''
    Single Time Step Prediction of DDF Model
    '''
    state_embedding = np.array([v_n,v_d1])
    Psi = RBF(state_embedding,centers,scale)
    v_hat = v_n + np.dot(weights,Psi)+alpha*(i_now+i_future)
    return v_hat

