import numpy as np
import pandas as pd
from utils import *
from sklearn.model_selection import train_test_split
from dram import DRAMLN

def select_instance(X, D):
    '''
        Discard the instances with tie labels
    '''
    selection = []
    for i, d in enumerate(D):
        _d = d[d!=0]
        if np.unique(_d).size == _d.size:
            selection.append(i)
    selection = np.array(selection)
    X, D = X[selection], D[selection]
    return X, D

def dist2rank(D):
    return [np.argsort(d)[d[d==0].size:].tolist() for d in D]

if __name__ == '__main__':
    dataset = 'Movie'
    config = np.load('config.npy', allow_pickle=True).item()
    
    print("Loading the optimal hyperparameters tuned on the machine\n\"Intel(R) Xeon(R) Silver 4110 CPU @ 2.10GHz, 64.0 GB RAM\"...")
    measures = ['Cheb', 'Canber', 'Cosine', 'Rho']
    result = pd.DataFrame(columns=measures)
    X, D = load(dataset)
    X, D = select_instance(X, D)
    for seed in range(10):
        print("Training on the dataset partition with random seed %d..." % seed)
        Xr, Xs, Dr, Ds = train_test_split(X, D, test_size=0.3, random_state=seed)
        Rr = dist2rank(Dr)
        param = config[dataset][seed]
        param['verbose'] = 3 # 3 immediate results will be printed
        model = DRAMLN(**param).fit(Xr, Rr, Dr.shape[1])
        result.loc[seed] = report(model.predict(Xs), Ds, out_form='pandas', ds=dataset).loc[measures]
    mean = result.mean(0)
    std = result.std(0)
    print('----------------------------------')
    print("Performance of DRAM on %s dataset:" % dataset)
    for mea in ['Cheb', 'Canber', 'Cosine', 'Rho']:
        print("%s (mean±std): %.3f±%.3f" % (mea, mean[mea], std[mea]))