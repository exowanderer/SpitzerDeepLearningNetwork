from tpot import TPOTClassifier

import pandas as pd
import numpy as np
# import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)

import pdb

import warnings
warnings.filterwarnings("ignore")

import gc

from sys import argv

# from matplotlib import pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import StandardScaler, MinMaxScaler, minmax_scale
from sklearn.ensemble         import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.decomposition    import PCA, FastICA
from sklearn.externals        import joblib
from sklearn.metrics          import r2_score

from glob                     import glob

# plt.rcParams['figure.dpi'] = 300

# from corner import corner

from time import time
start0 = time()

import pandas as pd

def setup_features(dataRaw, notFeatures=[], transformer=PCA(), feature_scaler=StandardScaler(), label_scaler=None, verbose=False, returnAll=None):
    """Example function with types documented in the docstring.

        For production level usage: All scaling and transformations must be done 
            with respect to the calibration data distributions
        
        Args:
            features  (nD-array): Array of input raw features.
            labels    (1D-array): The second parameter.
            transformer    (int): The first parameter.
            label_scaler   (str): The second parameter.
            feature_scaler (str): The second parameter.
        Returns:
            features_scaled_transformed, labels_scaled

        .. _PEP 484:
            https://github.com/ExoWanderer/

    """
    if isinstance(dataRaw,str):
        dataRaw       = pd.read_csv(filename)
        
    dataColNames  = dataRaw.columns
    PLDpixels     = {}
    for key in dataRaw.columns.values:
        if 'pix' in key:
            PLDpixels[key] = dataRaw[key]
    
    PLDpixels     = pd.DataFrame(PLDpixels)
    # PLDpixels     = pd.DataFrame({key:dataRaw[key] for key in dataRaw.columns.values if 'pix' in key})
    PLDnorm       = np.sum(np.array(PLDpixels),axis=1)
    PLDpixels     = (PLDpixels.T / PLDnorm).T
    
    inputData = dataRaw.copy()
    for key in dataRaw.columns: 
        if key in PLDpixels.columns:
            inputData[key] = PLDpixels[key]
    
    testPLD = np.array(pd.DataFrame({key:inputData[key] for key in inputData.columns.values if 'pix' in key}))
    if verbose: 
        assert(not sum(abs(testPLD - np.array(PLDpixels))).all())
        print('Confirmed that PLD Pixels have been Normalized to Spec')
    
    feature_columns = inputData.drop(notFeatures,axis=1).columns.values
    features        = inputData.drop(notFeatures,axis=1).values
    labels          = inputData['flux'].values
    print(features.shape)
    # **PCA Preconditioned Random Forest Approach**
    if verbose: print('Performing PCA', end=" ")
    if verbose: start = time()
    
    labels_scaled     = label_scaler.fit_transform(labels[:,None]).ravel() if label_scaler   is not None else labels
    features_scaled   = feature_scaler.fit_transform(features)             if feature_scaler is not None else features
    features_trnsfrmd = transformer.fit_transform(features_scaled)         if transformer    is not None else features_scaled
    
    if verbose: print('took {} seconds'.format(time() - start))
    
    if returnAll == True:
        return features_trnsfrmd, labels_scaled, dataRaw, transformer, label_scaler, feature_scaler
    
    if returnAll == 'features':
        return features_trnsfrmd
    
    if returnAll == 'labels':
        return labels_scaled
    
    if returnAll == 'both with raw data':
        features_trnsfrmd, labels_scaled, dataRaw
    
    return features_trnsfrmd, labels_scaled

def predict_with_scaled_transformer(dataRaw, notFeatures=None, transformer=None, feature_scaler=None, label_scaler=None, verbose=False):
    """Example function with types documented in the docstring.

        For production level usage: All scaling and transformations must be done 
            with respect to the calibration data distributions
        
        Args:
            features  (nD-array): Array of input raw features.
            labels    (1D-array): The second parameter.
            transformer    (int): The first parameter.
            label_scaler   (str): The second parameter.
            feature_scaler (str): The second parameter.
        Returns:
            features_scaled_transformed, labels_scaled

        .. _PEP 484:
            https://github.com/ExoWanderer/

    """
    if isinstance(dataRaw,str):
        dataRaw       = pd.read_csv(filename)
    
    dataColNames  = dataRaw.columns
    PLDpixels     = {}
    for key in dataRaw.columns.values:
        if 'pix' in key:
            PLDpixels[key] = dataRaw[key]
    
    PLDpixels     = pd.DataFrame(PLDpixels)
    
    PLDnorm       = np.sum(np.array(PLDpixels),axis=1)
    PLDpixels     = (PLDpixels.T / PLDnorm).T
    
    inputData = dataRaw.copy()
    for key in dataRaw.columns: 
        if key in PLDpixels.columns:
            inputData[key] = PLDpixels[key]
    
    if verbose: 
        testPLD = np.array(pd.DataFrame({key:inputData[key] for key in inputData.columns.values if 'pix' in key}))
        assert(not sum(abs(testPLD - np.array(PLDpixels))).all())
        print('Confirmed that PLD Pixels have been Normalized to Spec')
    
    feature_columns = inputData.drop(notFeatures,axis=1).columns.values
    features        = inputData.drop(notFeatures,axis=1).values
    labels          = inputData['flux'].values
    
    # **PCA Preconditioned Random Forest Approach**
    if verbose: print('Performing PCA')
    
    labels_scaled     = label_scaler.transform(labels[:,None]).ravel() if label_scaler   is not None else labels
    features_scaled   = feature_scaler.transform(features)             if feature_scaler is not None else features
    features_trnsfrmd = transformer.transform(features_scaled)         if transformer    is not None else features_scaled
    
    return features_trnsfrmd, labels_scaled

# ## Load CSVs data
spitzerCalNotFeatures = ['flux', 'fluxerr', 'dn_peak', 'xycov', 't_cernox', 'xerr', 'yerr', 'xfwhm', 'yfwhm']
spitzerCalFilename    ='pmap_ch2_0p1s_x4_rmulti_s3_7.csv'

spitzerCalRawData     = pd.read_csv(spitzerCalFilename)

features_SSscaled, labels_SSscaled = setup_features(dataRaw       = spitzerCalRawData,
                                                    notFeatures   = spitzerCalNotFeatures,
                                                    transformer   = None,
                                                    feature_scaler= StandardScaler(),
                                                    label_scaler  = None,
                                                    verbose       = False,
                                                    returnAll     = None)

idx_shuffle = np.random.permutation(len(spitzerCalRawData))

#clean the data
labels_SSscaled_shuffled    = labels_SSscaled[idx_shuffle]
features_SSscaled_shuffled  = features_SSscaled[idx_shuffle]

idx = np.arange(len(spitzerCalRawData))
#Split training, testing, and validation data
training_indices, validation_indices = training_indices, testing_indices = train_test_split(idx, train_size=0.75, test_size=0.25)


#Let Genetic Programming find best ML model and hyperparameters
tpot = TPOTClassifier(generations=5, verbosity=2)
tpot.fit(features_SSscaled_shuffled[training_indices], labels_SSscaled_shuffled[training_indicss])

#Score the accuracy
tpot.score(features_SSscaled_shuffled[validation_indices].values, labels_SSscaled_shuffled[validation_indices])

#Export the generated code
tpot.export('spitzer_calibration_tpot_best_pipeline.py')

