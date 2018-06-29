from argparse import ArgumentParser
ap = ArgumentParser()
ap.add_argument('-ns' , '--n_resamp'    , required=False, type=int , default=1    , help="Number of resamples to perform (GBR=1; No Resamp=0)")
ap.add_argument('-nt' , '--n_trees'     , required=False, type=int , default=100  , help="Number of trees in the forest")
ap.add_argument('-c'  , '--core'        , required=False, type=int , default=0    , help="Which Core to Use GBR only Uses 1 Core at a time.")
ap.add_argument('-std', '--do_std'      , required=False, type=bool, default=False, help="Use Standard Random Forest Regression")
ap.add_argument('-pca', '--do_pca'      , required=False, type=bool, default=False, help="Use Standard Random Forest Regression with PCA preprocessing")# nargs='?', const=True, 
ap.add_argument('-ica', '--do_ica'      , required=False, type=bool, default=False, help="Use Standard Random Forest Regression with ICA preprocessing")
ap.add_argument('-rfi', '--do_rfi'      , required=False, type=bool, default=False, help="Use Standard Random Forest Regression with PCA preprocessing")
ap.add_argument('-gbr', '--do_gbr'      , required=False, type=bool, default=False, help="Use Gradient Boosting Regression with PCA preprocessing")
ap.add_argument('-rs' , '--random_state', required=False, type=bool, default=False, help="Use Gradient Boosting Regression with PCA preprocessing")
ap.add_argument('-pdb', '--pdb_stop'    , required=False, type=bool, default=False, help="Stop the trace at the end with pdb.set_trace()")
ap.add_argument('-nj', '--n_jobs'       , required=False, type=int , default=-1   , help="Number of cores to use Default:-1")
args = vars(ap.parse_args())

do_std  = args['do_std']
do_pca  = args['do_pca']
do_ica  = args['do_ica']
do_rfi  = args['do_rfi']
do_gbr  = args['do_gbr']
pdb_stop= args['pdb_stop']
n_jobs  = args['n_jobs']

if n_jobs == 1: print('WARNING: You are only using 1 core!')

# Check if requested to complete more than one operatiion
#   if so
need_gc = sum([args[key] for key in args.keys() if 'do_' in key]) > 1

importance_filename = 'randForest_STD_feature_importances.txt'
if do_rfi and not len(glob(importance_filename)): do_std = True

import pandas as pd
import numpy as np
# import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)

import pdb

import warnings
warnings.filterwarnings("ignore")

import gc

from argparse import ArgumentParser
# from matplotlib import pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import StandardScaler, MinMaxScaler, minmax_scale
from sklearn.ensemble         import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.decomposition    import PCA, FastICA
from sklearn.externals        import joblib
from sklearn.metrics          import r2_score

from tqdm import tqdm

from glob                     import glob

# plt.rcParams['figure.dpi'] = 300

# from corner import corner

from time import time
start0 = time()

def setup_features(dataRaw, label='flux', notFeatures=[], transformer=PCA(whiten=True), feature_scaler=StandardScaler(), 
                    label_scaler=None, verbose=True, returnAll=None):
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
    # notFeatures = list(notFeatures)
    # notFeatures.append(label) if label not in notFeatures else None
    
    dataRaw   = pd.read_csv(filename) if isinstance(dataRaw,str) else dataRaw
    inputData = dataRaw.copy()
    
    PLDpixels = pd.DataFrame({key:dataRaw[key] for key in dataRaw.columns if 'pix' in key})
    print(''.format(PLDpixels.shape, PLDpixels.columns))
    # PLDpixels     = {}
    # for key in dataRaw.columns.values:
    #     if 'pix' in key:
    #         PLDpixels[key] = dataRaw[key]
    # 
    # PLDpixels     = pd.DataFrame(PLDpixels)
    
    PLDnorm       = np.sum(np.array(PLDpixels),axis=1)
    PLDpixels     = (PLDpixels.T / PLDnorm).T
    
    # Overwrite the PLDpixels entries with the normalized version
    for key in dataRaw.columns: 
        if key in PLDpixels.columns:
            inputData[key] = PLDpixels[key]
    
    # testPLD = np.array(pd.DataFrame({key:inputData[key] for key in inputData.columns.values if 'pix' in key})) if verbose else None
    # assert(not sum(abs(testPLD - np.array(PLDpixels))).all())  if verbose else None
    # print('Confirmed that PLD Pixels have been Normalized to Spec') if verbose else None
    
    labels          = inputData[label].values
    inputData       = inputData.drop(label, axis=1) # remove
    
    feature_columns = inputData.drop(notFeatures,axis=1).columns.values
    features        = inputData.drop(notFeatures,axis=1).values
    
    print('Shape of Features Array is', features.shape) if verbose else None
    
    if verbose: start = time()
    
    labels_scaled     = label_scaler.fit_transform(labels[:,None]).ravel() if label_scaler   is not None else labels
    features_scaled   = feature_scaler.fit_transform(features)             if feature_scaler is not None else features
    features_trnsfrmd = transformer.fit_transform(features_scaled)         if transformer    is not None else features_scaled
    
    print('took {} seconds'.format(time() - start)) if verbose else None
    
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
    dataRaw = pd.read_csv(filename) if isinstance(dataRaw,str) else dataRaw
    
    PLDpixels = pd.DataFrame({key:dataRaw[key] for key in dataRaw.columns if 'pix' in key})
    # PLDpixels     = {}
    # for key in dataRaw.columns.values:
    #     if 'pix' in key:
    #         PLDpixels[key] = dataRaw[key]
    
    # PLDpixels     = pd.DataFrame(PLDpixels)
    
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

files_in_directory = glob('./*')

# nRF_modes       = 6
# perform_rf_mode = np.ones(nRF_modes, dtype=bool)
#
# set_of_save_files  = ['./randForest_STD_approach.save',
#                       './randForest_PCA_approach.save',
#                       './randForest_ICA_approach.save',
#                       './randForest_RFI_approach.save',
#                       './randForest_RFI_PCA_approach.save',
#                       './randForest_RFI_ICA_approach.save']
#
# for k, sfile in enumerate(set_of_save_files):
#     if sfile in files_in_directory:
#         perform_rf_mode[k] = False

# ## Load CSVs data
spitzerCalNotFeatures = ['flux', 'fluxerr', 'dn_peak', 'xycov', 't_cernox', 'xerr', 'yerr', 'sigma_bg_flux']
spitzerCalFilename    ='pmap_ch2_0p1s_x4_rmulti_s3_7.csv'

spitzerCalRawData     = pd.read_csv(spitzerCalFilename)

spitzerCalRawData['fluxerr']        = spitzerCalRawData['fluxerr']      / np.median(spitzerCalRawData['flux'].values)
spitzerCalRawData['bg_flux']        = spitzerCalRawData['bg_flux']      / np.median(spitzerCalRawData['flux'].values)
spitzerCalRawData['sigma_bg_flux']  = spitzerCalRawData['sigma_bg_flux']/ np.median(spitzerCalRawData['flux'].values)
spitzerCalRawData['flux']           = spitzerCalRawData['flux']         / np.median(spitzerCalRawData['flux'].values)

spitzerCalRawData['bmjd_err']       = np.median(0.5*np.diff(spitzerCalRawData['bmjd']))
spitzerCalRawData['np_err']         = np.sqrt(spitzerCalRawData['yerr'])

n_PLD   = 9
n_resamp= args['n_resamp']

resampling_inputs = ['flux', 'xpos', 'ypos', 'xfwhm', 'yfwhm', 'bg_flux', 'bmjd', 'np'] + ['pix{}'.format(k) for k in range(1,10)]
resampling_errors = ['fluxerr', 'xerr', 'yerr', 'xerr', 'yerr', 'sigma_bg_flux', 'bmjd_err', 'np_err'] + ['fluxerr']*n_PLD

spitzerCalResampled = {}
if n_resamp > 0:
    print('Starting Resampling')
    for colname, colerr in tqdm(zip(resampling_inputs, resampling_errors), total=len(resampling_inputs)):
        if 'pix' in colname:
            spitzerCalResampled[colname]  = np.random.normal(spitzerCalRawData[colname], spitzerCalRawData[colname]*spitzerCalRawData['fluxerr'], size=(n_resamp,len(spitzerCalRawData))).flatten()
        else:
            spitzerCalResampled[colname]  = np.random.normal(spitzerCalRawData[colname], spitzerCalRawData[colerr], size=(n_resamp,len(spitzerCalRawData))).flatten()
    
    spitzerCalResampled = pd.DataFrame(spitzerCalResampled)
else:
    print('No Resampling')
    spitzerCalResampled = pd.DataFrame({colname:spitzerCalRawData[colname] for colname, colerr in tqdm(zip(resampling_inputs, resampling_errors), total=len(resampling_inputs))})

features_SSscaled, labels_SSscaled = setup_features(dataRaw       = spitzerCalResampled,
                                                    notFeatures   = [],#spitzerCalNotFeatures,
                                                    transformer   = PCA(whiten=True), # THIS IS PCA-RF -- NOT DEFAULT
                                                    feature_scaler= StandardScaler(),
                                                    label_scaler  = None,
                                                    verbose       = True,
                                                    returnAll     = None)

pca_cal_features_SSscaled = features_SSscaled

nTrees = args['n_trees']

start = time()
print('Grabbing PCA', end=" ")
pca_cal_features_SSscaled, labels_SSscaled, spitzerCalRawData, \
    pca_trnsfrmr, label_sclr, feature_sclr = setup_features(dataRaw       = spitzerCalResampled, 
                                                            notFeatures   = [],#spitzerCalNotFeatures, 
                                                            transformer   = None, 
                                                            feature_scaler= StandardScaler(),
                                                            label_scaler  = None,
                                                            verbose       = True,
                                                            returnAll     = True)

print(len(pca_cal_features_SSscaled))
print('took {} seconds'.format(time() - start))

if 'core' in args.keys():
    core = args['core']
else:
    from glob import glob
    existing_saves = glob('randForest_GBR_PCA_approach_{}trees_{}resamp_*core.save'.format(nTrees, n_resamp))
    
    core_nums = []
    for fname in existing_saves:
        core_nums.append(fname.split('randForest_GBR_PCA_approach_{}trees_{}resamp_'.format(nTrees, n_resamp))[-1].split('core.save')[0])
    
    core = max(core_nums) + 1

label_sclr_save_name    = 'spitzerCalLabelScaler_fit_{}resamp_{}core.save'.format(n_resamp, core)
feature_sclr_save_name  = 'spitzerCalFeatureScaler_fit_{}resamp_{}core.save'.format(n_resamp, core)
pca_trnsfrmr_save_name  = 'spitzerCalFeaturePCA_trnsfrmr_{}resamp_{}core.save'.format(n_resamp, core)

save_calibration_stacks = False
if label_sclr_save_name   not in files_in_directory and label_sclr   is not None: save_calibration_stacks = True
if feature_sclr_save_name not in files_in_directory and feature_sclr is not None: save_calibration_stacks = True
if pca_trnsfrmr_save_name not in files_in_directory and pca_trnsfrmr is not None: save_calibration_stacks = True

if save_calibration_stacks:
    # *** For production level usage ***
    # All scaling and transformations must be done with respect to the calibration data distributions
    #   - That means to use .transform instead of .fit_transform
    #   - See `predict_with_scaled_transformer`
    
    # Need to Scale the Labels based off of the calibration distribution
    joblib.dump(label_sclr  , label_sclr_save_name)
    
    # Need to Scale the Features based off of the calibration distribution
    joblib.dump(feature_sclr, feature_sclr_save_name)
    
    # Need to Transform the Scaled Features based off of the calibration distribution
    joblib.dump(pca_trnsfrmr, pca_trnsfrmr_save_name)

if do_pca:
    print('Performing PCA Random Forest')
    randForest_PCA = RandomForestRegressor( n_estimators=nTrees, 
                                            n_jobs=n_jobs, 
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
                                            verbose=True, 
                                            warm_start=True)
    
    print('Feature Shape: {}\nLabel Shape: {}'.format(pca_cal_features_SSscaled.shape, labels_SSscaled.shape))
    
    start=time()
    randForest_PCA.fit(pca_cal_features_SSscaled, labels_SSscaled)
    
    randForest_PCA_oob = randForest_PCA.oob_score_
    randForest_PCA_pred= randForest_PCA.predict(pca_cal_features_SSscaled)
    randForest_PCA_Rsq = r2_score(labels_SSscaled, randForest_PCA_pred)
    
    print('PCA Pretrained Random Forest:\n\tOOB Score: {:.3f}%\n\tR^2 score: {:.3f}%\n\tRuntime:   {:.3f} seconds'.format(randForest_PCA_oob*100, randForest_PCA_Rsq*100, time()-start))
    
    joblib.dump(randForest_PCA, 'randForest_PCA_approach_{}trees_{}resamp.save'.format(nTrees, n_resamp))

    if need_gc:
        del randForest_PCA, randForest_PCA_pred
        gc.collect();

if do_gbr:
    
    trainX, testX, trainY, testY = train_test_split(pca_cal_features_SSscaled, labels_SSscaled, test_size=0.25)
    
    print('Performing Gradient Boosting Regression with PCA Random Forest and Quantile Loss')
    randForest_PCA_GBR = GradientBoostingRegressor(loss='quantile', 
                                                   learning_rate=0.1, 
                                                   n_estimators=nTrees, 
                                                   # n_jobs=n_jobs,
                                                   # bootstrap=True,
                                                   # oob_score=True, 
                                                   subsample=1.0, 
                                                   criterion='friedman_mse', 
                                                   min_samples_split=2, 
                                                   min_samples_leaf=1, 
                                                   min_weight_fraction_leaf=0.0, 
                                                   max_depth=3,#None, 
                                                   min_impurity_decrease=0.0, 
                                                   min_impurity_split=None, 
                                                   init=None, 
                                                   random_state=42, 
                                                   max_features='auto', 
                                                   alpha=0.9, 
                                                   verbose=True, 
                                                   max_leaf_nodes=None,
                                                   warm_start=True,
                                                   presort='auto')
    
    print(pca_cal_features_SSscaled.shape, labels_SSscaled.shape)
    
    start=time()
    randForest_PCA_GBR.fit(trainX, trainY)
    
    randForest_PCA_GBR_pred_train = randForest_PCA_GBR.predict(trainX)
    randForest_PCA_GBR_pred_test  = randForest_PCA_GBR.predict(testX)
    randForest_PCA_GBR_Rsq_train  = r2_score(trainY, randForest_PCA_GBR_pred_train)
    randForest_PCA_GBR_Rsq_test   = r2_score(testY , randForest_PCA_GBR_pred_test )
    
    print('PCA Pretrained Random Forest:\n\tTrain R^2 Score: {:.3f}%\n\tTest R^2 score: {:.3f}%\n\tRuntime:   {:.3f} seconds'.format(
                    randForest_PCA_GBR_Rsq_train*100, randForest_PCA_GBR_Rsq_test*100, time()-start))
    
    joblib.dump(randForest_PCA_GBR, 'randForest_GBR_PCA_approach_{}trees_{}resamp_{}core.save'.format(nTrees, n_resamp, core))
    
    if need_gc:
        del randForest_PCA, randForest_PCA_pred
        gc.collect();

if do_std:
    # **Standard Random Forest Approach**
    # for nComps in range(1,spitzerData.shape[1]):
    print('Performing STD Random Forest')
    randForest_STD = RandomForestRegressor( n_estimators=nTrees, \
                                            n_jobs=n_jobs, \
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
                                            verbose=True, \
                                            warm_start=True)

    start=time()
    randForest_STD.fit(features_SSscaled, labels_SSscaled)

    # Save for Later
    importances = randForest_STD.feature_importances_
    np.savetxt(importance_filename, importances)
    
    randForest_STD_oob = randForest_STD.oob_score_
    randForest_STD_pred= randForest_STD.predict(features_SSscaled)
    randForest_STD_Rsq = r2_score(labels_SSscaled, randForest_STD_pred)

    print('Standard Random Forest:\n\tOOB Score: {:.3f}%\n\tR^2 score: {:.3f}%\n\tRuntime:   {:.3f} seconds'.format(randForest_STD_oob*100, randForest_STD_Rsq*100, time()-start))

    joblib.dump(randForest_STD, 'randForest_STD_approach_{}trees_{}resamp.save'.format(nTrees, n_resamp))
    
    if need_gc:
        del randForest_STD, randForest_STD_pred
        gc.collect();

if do_ica:
    # for nComps in range(1,spitzerData.shape[1]):
    print('Performing ICA Random Forest')
    start = time()
    print('Performing ICA', end=" ")
    ica_cal_feature_set  = setup_features(dataRaw       = spitzerCalResampled, 
                                          notFeatures   = spitzerCalNotFeatures, 
                                          transformer   = FastICA(), 
                                          feature_scaler= StandardScaler(),
                                          label_scaler  = None,
                                          verbose       = True, 
                                          returnAll     = 'features')
    
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
                                            n_jobs=n_jobs, 
                                            random_state=42, 
                                            verbose=True, 
                                            warm_start=True)
    
    start=time()
    randForest_ICA.fit(ica_cal_feature_set, labels_SSscaled)
    
    randForest_ICA_oob = randForest_ICA.oob_score_
    randForest_ICA_pred= randForest_ICA.predict(ica_cal_feature_set)
    randForest_ICA_Rsq = r2_score(labels_SSscaled, randForest_ICA_pred)
    
    print('ICA Pretrained Random Forest:\n\tOOB Score: {:.3f}%\n\tR^2 score: {:.3f}%\n\tRuntime:   {:.3f} seconds'.format(randForest_ICA_oob*100, randForest_ICA_Rsq*100, time()-start))
    
    joblib.dump(randForest_ICA, 'randForest_ICA_approach_{}trees_{}resamp.save'.format(nTrees, n_resamp))
    
    if need_gc:
        del randForest_ICA, randForest_ICA_oob, randForest_ICA_pred, randForest_ICA_Rsq
        gc.collect();

if do_rfi:
    # **Importance Sampling**
    print('Computing Importances for RFI Random Forest')
    importances = np.loadtxt(importance_filename)
    indices     = np.argsort(importances)[::-1]
    
    cumsum = np.cumsum(importances[indices])
    nImportantSamples = np.argmax(cumsum >= 0.95) + 1
    
    # **Random Forest Pretrained Random Forest Approach**
    rfi_cal_feature_set = features_SSscaled.T[indices][:nImportantSamples].T
    
    # for nComps in range(1,spitzerData.shape[1]):
    print('Performing RFI Random Forest')
    
    randForest_RFI = RandomForestRegressor( n_estimators=nTrees, \
                                            n_jobs=n_jobs, \
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
                                            verbose=True, \
                                            warm_start=True)
    
    start=time()
    randForest_RFI.fit(rfi_cal_feature_set, labels_SSscaled)
    
    randForest_RFI_oob = randForest_RFI.oob_score_
    randForest_RFI_pred= randForest_RFI.predict(rfi_cal_feature_set)
    randForest_RFI_Rsq = r2_score(labels_SSscaled, randForest_RFI_pred)
    
    print('RFI Pretrained Random Forest:\n\tOOB Score: {:.3f}%\n\tR^2 score: {:.3f}%\n\tRuntime:   {:.3f} seconds'.format(randForest_RFI_oob*100, randForest_RFI_Rsq*100, time()-start))
    
    joblib.dump(randForest_RFI, 'randForest_RFI_approach_{}trees_{}resamp.save'.format(nTrees, n_resamp))
    
    if need_gc:
        del randForest_RFI, randForest_RFI_oob, randForest_RFI_pred, randForest_RFI_Rsq
        gc.collect();

do_rfi_pca=False
if do_rfi_pca:
    # **PCA Pretrained Random Forest Approach**
    print('Performing PCA on RFI', end=" ")
    start = time()
    
    pca = PCA(whiten=True)
    pca_rfi_cal_feature_set = pca.fit_transform(rfi_cal_feature_set)
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
                                                bootstrap=True,
                                                oob_score=True, 
                                                n_jobs=n_jobs, 
                                                random_state=42, 
                                                verbose=True, 
                                                warm_start=True)

    start=time()
    randForest_RFI_PCA.fit(pca_rfi_cal_feature_set, labels_SSscaled)
    
    randForest_RFI_PCA_oob = randForest_RFI_PCA.oob_score_
    randForest_RFI_PCA_pred= randForest_RFI_PCA.predict(pca_rfi_cal_feature_set)
    randForest_RFI_PCA_Rsq = r2_score(labels_SSscaled, randForest_RFI_PCA_pred)
    
    print('RFI Pretrained with PCA Random Forest:\n\tOOB Score: {:.3f}%\n\tR^2 score: {:.3f}%\n\tRuntime:   {:.3f} seconds'.format(
        randForest_RFI_PCA_oob*100, randForest_RFI_PCA_Rsq*100, time()-start))
    
    joblib.dump(randForest_RFI_PCA, 'randForest_RFI_PCA_approach_{}trees_{}resamp.save'.format(nTrees, n_resamp))
    
    if need_gc:
        del randForest_RFI_PCA, randForest_RFI_PCA_oob, randForest_RFI_PCA_pred, randForest_RFI_PCA_Rsq
        gc.collect();

do_rfi_ica = False
if do_rfi_ica:
    # **ICA Pretrained Random Forest Approach**
    print('Performing ICA on RFI', end=" ")
    ica = FastICA()
    start = time()
    
    ica_rfi_cal_feature_set = ica.fit_transform(rfi_cal_feature_set)
    print('took {} seconds'.format(time() - start))
    
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
                                                n_jobs=n_jobs, 
                                                random_state=42, 
                                                verbose=True, 
                                                warm_start=True)

    start=time()
    randForest_RFI_ICA.fit(ica_rfi_cal_feature_set, labels_SSscaled)
    
    randForest_RFI_ICA_oob = randForest_RFI_ICA.oob_score_
    randForest_RFI_ICA_pred= randForest_RFI_ICA.predict(ica_rfi_cal_feature_set)
    randForest_RFI_ICA_Rsq = r2_score(labels_SSscaled, randForest_RFI_ICA_pred)
    
    print('RFI Pretrained with ICA Random Forest:\n\tOOB Score: {:.3f}%\n\tR^2 score: {:.3f}%\n\tRuntime:   {:.3f} seconds'.format(
        randForest_RFI_ICA_oob*100, randForest_RFI_ICA_Rsq*100, time()-start))
    
    joblib.dump(randForest_RFI_ICA, 'randForest_RFI_ICA_approach_{}trees_{}resamp.save'.format(nTrees, n_resamp))
    
    if need_gc:
        del randForest_RFI_ICA, randForest_RFI_ICA_oob, randForest_RFI_ICA_pred, randForest_RFI_ICA_Rsq
        gc.collect();

print('\n\nFull Operation took {:.2f} minutes'.format((time() - start0)/60))
if pdb_stop: pdb.set_trace()
