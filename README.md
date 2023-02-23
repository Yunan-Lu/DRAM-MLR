# DRAM
Implementation of "Predicting Label Distribution from Multi-label Ranking"

DRAM: a framework for predicting label **D**istribution from multi-label **RA**nking via conditional Dirichlet **M**ixtures

## Environment
python=3.7.6, numpy=1.21.6, pandas=1.3.5, scikit-learn=0.24.2, scipy=1.7.3, pytorch=1.13.0+cpu

## Reproducing
Change the directory to this project and run the following command in terminal.
```python
python demo.py
```

## Usage
Here is a simple example of using DRAM-LN.
```python
from dram import DRAMLN

# train DRAM-LN
dramln = DRAMLN(validate=None).fit(X, R, M) # X: feature matrix; R: rankings, e.g., [[0, 2, 1], [3, 1], ...]; M: number of labels

# train DRAM-LN
dramln = DRAMLN(validate=(Xvalidate, Rvalidate)).fit(X, R, M)

# Predict label distributions for Xtest
Dhat = dramln.predict(Xtest)
```

## Paper
```latex
@inproceedings{Lu2022DRAM,
	title={Predicting Label Distribution from Multi-label Ranking},
	author={Yunan Lu and Xiuyi Jia},
	booktitle={Advances in Neural Information Processing Systems},
	year={2022}
}
```
