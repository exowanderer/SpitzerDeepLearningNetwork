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

from time import time
start0 = time()

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA

def predict_with_scaled_transformer(dataRaw, notFeatures=None, transformer=None, label_scaler=None, feature_scaler=None):
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

set_of_save_files  = ['./randForest_STD_approach.save', 
                      './randForest_PCA_approach.save', 
                      './randForest_ICA_approach.save', 
                      './randForest_RFI_approach.save', 
                      './randForest_RFI_PCA_approach.save', 
                      './randForest_RFI_ICA_approach.save']

# *** Load CSVs data ***
spitzerCalNotFeatures = ['flux', 'fluxerr', 'dn_peak', 'xycov', 't_cernox', 'xerr', 'yerr']
# *** For production level usage ***
# All scaling and transformations must be done with respect to the calibration data distributions
#   - That means to use .transform instead of .fit_transform
#   - See `predict_with_scaled_transformer`

rf_savename = 'randForest_STD_approach.save' if len(argv) < 2 else argv[2]

# THIS TAKES A LONG TIME!! (~30 minutes)
randForest  = joblib.load(rf_savename)

# Need to Scale the Labels based off of the calibration distribution
label_sclr    = joblib.load('spitzerCalLabelScaler_fit.save')

# Need to Scale the Features based off of the calibration distribution
feature_sclr  = joblib.load('spitzerCalFeatureScaler_fit.save')

if 'pca' in rf_savename.lower():
    # Need to Transform the Scaled Features based off of the calibration distribution
    pca_trnsfrmr  = joblib.load('spitzerCalFeaturePCA_trnsfrmr.save')

try:
    new_data_filename   = argv[1]
except:
    raise FileError('Usage: python sklearn_RandomForests_predict_new_data.py CSV_FILENAME')

new_data  = pd.read_csv(new_data_filename)

new_NotFeatures = [notFeature for notFeature in spitzerCalNotFeatures if notFeature in new_data.columns]

new_features, new_labels = predict_with_scaled_transformer(new_data, notFeatures=new_NotFeatures, transformer=pca_trnsfrmr, label_scaler=label_sclr, feature_scaler=feature_sclr)

new_rf_predict  = randForest.predict(new_features)

