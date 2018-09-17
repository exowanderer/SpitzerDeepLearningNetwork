from tpot import TPOTRegressor

from functools import partial

import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# from matplotlib import pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn.externals        import joblib
from sklearn.metrics          import r2_score

from time import time


n_skip = 100 # testing on smaller data set
features = pd.read_csv('pmap_raw_16features.csv')[::n_skip]
labels = pd.read_csv('pmap_raw_labels_and_errors.csv')[::n_skip]

#Split training, testing, and validation data
training_indices, validation_indices = train_test_split(idx, test_size=0.20)

#Let Genetic Programming find best ML model and hyperparameters
tpot = TPOTRegressor(generations=10, verbosity=2, n_jobs=-1)

start = time()
tpot.fit(features.iloc[training_indices].values, labels.iloc[training_indices].values)
print('Full TPOT regressor operation took {:.1f} minutes'.format((time() - start)/60))

#Score the accuracy

print('Best pipeline test accuracy: {:.3f}'.format(
  tpot.score(features.iloc[validation_indices].values, labels.iloc[validation_indices].values)))

#Export the generated code
tpot.export('spitzer_calibration_tpot_best_pipeline.py')
