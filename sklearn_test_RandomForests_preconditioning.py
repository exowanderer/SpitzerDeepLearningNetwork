import pandas as pd
import numpy as np
# import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)

import pdb

import warnings
warnings.filterwarnings("ignore")

import gc

# from matplotlib import pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import StandardScaler, MinMaxScaler, minmax_scale
from sklearn.ensemble         import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.decomposition    import PCA, FastICA
from sklearn.externals        import joblib
from sklearn.metrics          import r2_score

# plt.rcParams['figure.dpi'] = 300

# from corner import corner

from time import time
start0 = time()

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FastICA

def setup_features(dataRaw, notFeatures = [], transformer = PCA(), scaler = StandardScaler, verbose = False, returnAll = None):
    if isinstance(dataRaw,str):
        dataRaw       = pd.read_csv(filename)
    
    feature_scaler = scaler()
    label_scaler   = scaler()
    
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

def predict_with_scaled_transformer(features, labels, transformer, label_scaler, feature_scaler):
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
    '''
        
    
            features
            
    '''
    labels_scaled     = label_scaler.transform(labels[:,None]).ravel()
    features_scaled   = feature_scaler.transform(features)
    features_trnsfrmd = transformer.transform(features_scaled)
    
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

# ## Load CSVs data
spitzerCalNotFeatures = ['flux', 'fluxerr', 'dn_peak', 'xycov', 't_cernox', 'xerr', 'yerr']
spitzerCalFilename    ='pmap_ch2_0p1s_x4_rmulti_s3_7.csv'

spitzerCalRawData     = pd.read_csv(spitzerCalFilename)

features_SSscaled, labels_SSscaled = setup_features(dataRaw    = spitzerCalRawData, 
                                                    notFeatures= spitzerCalNotFeatures, 
                                                    transformer= None, 
                                                    scaler     = StandardScaler(), 
                                                    verbose    = False,
                                                    returnAll  = None)

nTrees = 1000

# **Standard Random Forest Approach**
# for nComps in range(1,spitzerData.shape[1]):
print('Performing STD Random Forest')
randForest_STD = RandomForestRegressor( n_estimators=nTrees, \
                                        n_jobs=-1, \
                                        criterion='mse', \
                                        max_depth=None, \
                                        min_samples_split=2, \
                                        min_samples_leaf=1, \
                                        min_weight_fraction_leaf=0.0, \
                                        max_features='auto', \
                                        max_leaf_nodes=None, \
                                        bootstrap=True, \
                                        oob_score=True, \
                                        random_state=42, \
                                        verbose=0, \
                                        warm_start=True)

start=time()
randForest_STD.fit(features_SSscaled, labels_SSscaled)

# Save for Later
importances = randForest_STD.feature_importances_
np.savetxt('randForest_STD_feature_importances.txt', importances)

randForest_STD_oob = randForest_STD.oob_score_
randForest_STD_pred= randForest_STD.predict(features_SSscaled)
randForest_STD_Rsq = r2_score(labels_SSscaled, randForest_STD_pred)

print('Standard Random Forest:\n\tOOB Score: {:.3f}%\n\tR^2 score: {:.3f}%\n\tRuntime:   {:.3f} seconds'.format(randForest_STD_oob*100, randForest_STD_Rsq*100, time()-start))

joblib.dump(randForest_STD, 'randForest_STD_approach.save')
del randForest_STD, randForest_STD_pred
_ = gc.collect()


# for nComps in range(1,spitzerData.shape[1]):
print('Performing PCA Random Forest')

start = time()
print('Performing PCA', end=" ")
pca_cal_features_SSscaled, labels_SSscaled, spitzerCalRawData, \
    pca_trnsfrmr, label_sclr, feature_sclr = setup_features(dataRaw    = spitzerCalRawData, 
                                                            notFeatures= spitzerCalNotFeatures, 
                                                            transformer= None, 
                                                            scaler     = StandardScaler(), 
                                                            verbose    = False,
                                                            returnAll  = True)

# *** For production level usage ***
# All scaling and transformations must be done with respect to the calibration data distributions
#   - That means to use .transform instead of .fit_transform
#   - See `predict_with_scaled_transformer`

# Need to Scale the Labels based off of the calibration distribution
joblib.dump(label_sclr  , 'spitzerCalLabelScaler_fit.save')
# Need to Scale the Features based off of the calibration distribution
joblib.dump(feature_sclr, 'spitzerCalFeatureScaler_fit.save')
# Need to Transform the Scaled Features based off of the calibration distribution
joblib.dump(pca_trnsfrmr, 'spitzerCalFeaturePCA_trnsfrmr.save')

spitzerCalPCA_fit.save
print(len(pca_cal_feature_set))
print('took {} seconds'.format(time() - start))

randForest_PCA = RandomForestRegressor( n_estimators=nTrees, 
                                        n_jobs=-1, 
                                        criterion='mse', 
                                        max_depth=None, 
                                        min_samples_split=2, 
                                        min_samples_leaf=1, 
                                        min_weight_fraction_leaf=0.0, 
                                        max_features='auto', 
                                        max_leaf_nodes=None, 
                                        bootstrap=True, 
                                        oob_score=True, 
                                        random_state=42, 
                                        verbose=0, 
                                        warm_start=True)

print(pca_cal_feature_set.shape, labels_SSscaled.shape)

start=time()
randForest_PCA.fit(pca_cal_feature_set, labels_SSscaled)

randForest_PCA_oob = randForest_PCA.oob_score_
randForest_PCA_pred= randForest_PCA.predict(pca_cal_feature_set)
randForest_PCA_Rsq = r2_score(labels_SSscaled, randForest_PCA_pred)

print('PCA Pretrained Random Forest:\n\tOOB Score: {:.3f}%\n\tR^2 score: {:.3f}%\n\tRuntime:   {:.3f} seconds'.format(randForest_PCA_oob*100, randForest_PCA_Rsq*100, time()-start))

joblib.dump(randForest_PCA, 'randForest_PCA_approach.save')

del randForest_PCA, randForest_PCA_pred
_ = gc.collect()

# for nComps in range(1,spitzerData.shape[1]):
print('Performing ICA Random Forest')
start = time()
print('Performing ICA', end=" ")
ica_cal_feature_set  = setup_features(dataRaw    = spitzerCalRawData, 
                                      notFeatures= spitzerCalNotFeatures, 
                                      transformer= FastICA(), 
                                      scaler     = StandardScaler(), 
                                      verbose    = False, 
                                      returnAll  = 'simple')

print('took {} seconds'.format(time() - start))

randForest_ICA = RandomForestRegressor( n_estimators=nTrees, 
                                        criterion='mse', 
                                        max_depth=None, 
                                        min_samples_split=2, 
                                        min_samples_leaf=1, 
                                        min_weight_fraction_leaf=0.0, 
                                        max_features='auto', 
                                        max_leaf_nodes=None, 
                                        bootstrap=True, 
                                        oob_score=True, 
                                        n_jobs=-1, 
                                        random_state=42, 
                                        verbose=0, 
                                        warm_start=True)

start=time()
randForest_ICA.fit(ica_cal_feature_set, labels_SSscaled)

randForest_ICA_oob = randForest_ICA.oob_score_
randForest_ICA_pred= randForest_ICA.predict(ica_cal_feature_set)
randForest_ICA_Rsq = r2_score(labels_SSscaled, randForest_ICA_pred)

print('ICA Pretrained Random Forest:\n\tOOB Score: {:.3f}%\n\tR^2 score: {:.3f}%\n\tRuntime:   {:.3f} seconds'.format(randForest_ICA_oob*100, randForest_ICA_Rsq*100, time()-start))

joblib.dump(randForest_ICA, 'randForest_ICA_approach.save')
del randForest_ICA
_ = gc.collect()


# randForest_STD = joblib.load('randForest_standard_approach.save', mmap_mode='r+')

# **Importance Sampling**
print('Computing Importances for RFI Random Forest')

indices     = np.argsort(importances)[::-1]

cumsum = np.cumsum(importances[indices])
nImportantSamples = np.argmax(cumsum >= 0.95) + 1

# **Random Forest Pretrained Random Forest Approach**
rfi_cal_feature_set = features_SSscaled.T[indices][:nImportantSamples].T

# for nComps in range(1,spitzerData.shape[1]):
print('Performing RFI Random Forest')

randForest_RFI = RandomForestRegressor( n_estimators=nTrees, \
                                        n_jobs=-1, \
                                        criterion='mse', \
                                        max_depth=None, \
                                        min_samples_split=2, \
                                        min_samples_leaf=1, \
                                        min_weight_fraction_leaf=0.0, \
                                        max_features='auto', \
                                        max_leaf_nodes=None, \
                                        bootstrap=True, \
                                        oob_score=True, \
                                        random_state=42, \
                                        verbose=0, \
                                        warm_start=True)

start=time()
randForest_RFI.fit(rfi_cal_feature_set, labels_SSscaled)

randForest_RFI_oob = randForest_RFI.oob_score_
randForest_RFI_pred= randForest_RFI.predict(rfi_cal_feature_set)
randForest_RFI_Rsq = r2_score(labels_SSscaled, randForest_RFI_pred)

print('RFI Pretrained Random Forest:\n\tOOB Score: {:.3f}%\n\tR^2 score: {:.3f}%\n\tRuntime:   {:.3f} seconds'.format(randForest_RFI_oob*100, randForest_RFI_Rsq*100, time()-start))

joblib.dump(randForest_RFI, 'randForest_RFI_approach.save')
del randForest_RFI
_ = gc.collect()


# **PCA Pretrained Random Forest Approach**
print('Performing PCA on RFI', end=" ")
start = time()

pca = PCA()
pca_rfi_cal_feature_set = pca.fit_transform(rfi_cal_feature_set)
print('took {} seconds'.format(time() - start))

# **ICA Pretrained Random Forest Approach**
print('Performing ICA on RFI', end=" ")
ica = FastICA()
start = time()

ica_rfi_cal_feature_set = ica.fit_transform(rfi_cal_feature_set)
print('took {} seconds'.format(time() - start))

print('Performing RFI with PCA Random Forest')

randForest_RFI_PCA = RandomForestRegressor( n_estimators=nTrees, 
                                            criterion='mse', 
                                            max_depth=None, 
                                            min_samples_split=2, 
                                            min_samples_leaf=1, 
                                            min_weight_fraction_leaf=0.0, 
                                            max_features='auto', 
                                            max_leaf_nodes=None, 
                                            bootstrap=True
                                            oob_score=True, 
                                            n_jobs=-1, 
                                            random_state=42, 
                                            verbose=0, 
                                            warm_start=True)

start=time()
randForest_RFI_PCA.fit(pca_rfi_cal_feature_set, labels_SSscaled)

randForest_RFI_PCA_oob = randForest_RFI_PCA.oob_score_
randForest_RFI_PCA_pred= randForest_RFI_PCA.predict(pca_rfi_cal_feature_set)
randForest_RFI_PCA_Rsq = r2_score(labels_SSscaled, randForest_RFI_PCA_pred)

print('RFI Pretrained with PCA Random Forest:\n\tOOB Score: {:.3f}%\n\tR^2 score: {:.3f}%\n\tRuntime:   {:.3f} seconds'.format(
    randForest_RFI_PCA_oob*100, randForest_RFI_PCA_Rsq*100, time()-start))

joblib.dump(randForest_RFI_PCA, 'randForest_RFI_PCA_approach.save')

del randForest_RFI_PCA
_ = gc.collect()

print('Performing RFI with ICA Random Forest')

randForest_RFI_ICA = RandomForestRegressor( n_estimators=nTrees, 
                                            criterion='mse', 
                                            max_depth=None, 
                                            min_samples_split=2, 
                                            min_samples_leaf=1, 
                                            min_weight_fraction_leaf=0.0, 
                                            max_features='auto', 
                                            max_leaf_nodes=None, 
                                            bootstrap=True, 
                                            oob_score=True, 
                                            n_jobs=-1, 
                                            random_state=42, 
                                            verbose=0, 
                                            warm_start=True)

start=time()
randForest_RFI_ICA.fit(ica_rfi_cal_feature_set, labels_SSscaled)

randForest_RFI_ICA_oob = randForest_RFI_ICA.oob_score_
randForest_RFI_ICA_pred= randForest_RFI_ICA.predict(ica_rfi_cal_feature_set)
randForest_RFI_ICA_Rsq = r2_score(labels_SSscaled, randForest_RFI_ICA_pred)

print('RFI Pretrained with ICA Random Forest:\n\tOOB Score: {:.3f}%\n\tR^2 score: {:.3f}%\n\tRuntime:   {:.3f} seconds'.format(
    randForest_RFI_ICA_oob*100, randForest_RFI_ICA_Rsq*100, time()-start))

joblib.dump(randForest_RFI_ICA, 'randForest_RFI_ICA_approach.save')

del randForest_RFI_ICA
_ = gc.collect()

pdb.set_trace()