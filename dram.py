import numpy as np
import torch
import torch.nn as nn
from  torch.distributions import Dirichlet
from sklearn.base import BaseEstimator
from scipy.io import loadmat
from time import time
from utils import Rho


class DRAMLN(BaseEstimator):
    '''
    Parameters::
        n_models: int, default=3
            Number of Dirichlet components.
        lam: float, default=1e-5
            Strength of L2 norm regularization.
        max_iter: int, default=10
            Maximum iteration of EM.
        n_samples: int, default=20
            Number of samples for approximating the target prior.
        verbose: int, default=0
            How many intermediate results will be printed
        max_iter_lbfgs: int, default=10
            Maximum iteration of LBFGS used in the M-step.
        lr_lbfgs: int, default=1
            Predefined learning rate of LBFGS used in the M-step.
        validate: tuple, default=None
            Validation set like (feature_matrix, ranking_matrix)
        random_state: int, default=123
            Random seed for controlling reproducibility.
    --------------------------------------------------------------------
    Methods::
        fit(X, Ranks, M): training the model
            X: ndarray of shape (n_samples, n_features)
            Ranks: a list of relevant label indexes, 
                e.g., [argsort(d)[len(d[d==0]):] for d in label_dist_array]
            M: number of labels
        predict(X): predicting label distributions for X
    '''

    def __init__(self, n_models=3, lam=1e-5, max_iter=10, n_samples=20, 
                verbose=0, max_iter_lbfgs=10, lr_lbfgs=1,
                validate=None, random_state=123):
        self.n_models = n_models
        self.lam = lam
        self.max_iter = max_iter
        self.n_samples = n_samples
        self.lr_lbfgs = lr_lbfgs
        self.max_iter_lbfgs = max_iter_lbfgs
        self.verbose = verbose
        self.validate = validate
        self.random_state = random_state

    def _zsample_generator(self, rank_matrix, n_labels, num):
        final = []
        samples = loadmat('samples.mat')
        for rank in rank_matrix:
            if len(rank) == 1:
                vec = np.ones((num, 1))
            else:
                vecs = samples['%ditems' % len(rank)]
                vec = vecs[np.random.randint(0, vecs.shape[0], size=num)]
            temp = np.zeros((num, n_labels))
            for i, ind in enumerate(rank):
                temp[:, ind] = vec[:, i]
            final.append(temp)
        return np.stack(final)

    def predict(self, X):
        with torch.no_grad():
            torch.manual_seed(self.random_state)
            if not isinstance(X, torch.Tensor):
                X = torch.FloatTensor(X)
            posterior = self.post_fn(X) + 1e-6  # shape=(N, K)
            locs = self.loc_fn(X).view(X.shape[0], self.n_models, -1) # shape=(N, K, M)
            Zhat = locs / locs.sum(2, keepdims=True)
            Zhat = (Zhat * posterior.unsqueeze(2)).sum(1)
        return Zhat.numpy()

    def fit(self, X, Ranks, M):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        K, L = self.n_models, self.n_samples
        X = torch.FloatTensor(X)
        self.loc_fn = nn.Sequential(nn.Linear(X.shape[1], M*K), nn.Softplus())
        self.post_fn = nn.Sequential(nn.Linear(X.shape[1], K), nn.Softmax(dim=1))
        params = list(self.loc_fn.parameters()) + list(self.post_fn.parameters())
        for p in params:
            nn.init.normal_(p, mean=0.0, std=0.1)
        gammas = torch.softmax(torch.rand((X.shape[0], L, K)), dim=-1)   # shape=(N, L, K)
        
        # Expectation Maximization
        for em in range(self.max_iter):
            start = time()
            
            # Generate label distribution samples
            Zsamples = torch.clip(torch.FloatTensor(self._zsample_generator(Ranks, M, L)), 1e-4, 1-1e-4) # shape=(N, L, M)
            Zsamples /= Zsamples.sum(-1, keepdims=True)
            
            # M-step
            optimizer = torch.optim.LBFGS(params, lr=self.lr_lbfgs, max_iter=self.max_iter_lbfgs, history_size=5,
                                    tolerance_change=1e-5, tolerance_grad=1e-6, line_search_fn='strong_wolfe')
            def closure():
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                concentration = self.loc_fn(X).view(-1, 1, K, M) + 1e-6    # shape=(N, 1, K, M)
                likelihood = Dirichlet(concentration).log_prob(Zsamples.unsqueeze(-2)) # shape=(N, L, K)
                elbo = (((self.post_fn(X) + 1e-6).unsqueeze(1).log() + likelihood) * gammas).mean(1)
                regularization = 0
                for p in self.loc_fn.parameters():
                    regularization += self.lam * p.pow(2).sum()
                loss = (regularization - elbo).sum()
                if loss.requires_grad:
                    loss.backward()
                return loss
            loss = optimizer.step(closure)
            
            # E-step
            with torch.no_grad():
                posterior = self.post_fn(X) # shape=(N, K)
                concentration = self.loc_fn(X).view(-1, 1, K, M) + 1e-6 # shape=(N, 1, M, K)
                likelihood = Dirichlet(concentration).log_prob(Zsamples.unsqueeze(-2)) # shape=(N, L, K)
                gammas = ((1e-6+posterior).log().unsqueeze(1) + likelihood)
                gammas = torch.softmax(gammas, dim=2)
                unit_loss = loss/(X.shape[0]*M)
            end = time()
            
            # print
            training_Rho = Rho(self.predict(X), Zsamples[:,0,:].numpy())
            if self.validate is None:
                if (self.verbose > 0) and (em % (self.max_iter // self.verbose) == 0):
                    print("Iteration: %2d, time lapse: %.2fs, spearmanr: %.3f, unit loss: %.3f" % 
                        (em+1, (end-start), training_Rho, unit_loss))
            else:
                Xv, Dv = self.validate
                print("Iteration: %2d, time lapse: %.2fs, training/validating spearmanr: %.3f / %.3f, unit loss: %.3f" % 
                        (em+1, (end-start), training_Rho, Rho(self.predict(Xv), Dv), unit_loss))
        return self