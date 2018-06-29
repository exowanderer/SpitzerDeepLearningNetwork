try:
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-ns' , '--n_resamp'    , required=False, type=int , default=0    , help="Number of resamples to perform (GBR=1; No Resamp=0).")
    ap.add_argument('-nt' , '--n_trees'     , required=False, type=int , default=100  , help="Number of trees in the forest.")
    ap.add_argument('-c'  , '--core'        , required=False, type=int , default='A'  , help="Which Core to Use GBR only Uses 1 Core at a time.")
    ap.add_argument('-pp' , '--pre_process' , required=False, type=bool, default=True , help="Flag whether to use StandardScaler to pre-process the data.")
    ap.add_argument('-std', '--do_std'      , required=False, type=bool, default=False, help="Use Standard Random Forest Regression.")
    ap.add_argument('-pca', '--do_pca'      , required=False, type=bool, default=False, help="Use Standard Random Forest Regression with PCA preprocessing.")# nargs='?', const=True, 
    ap.add_argument('-ica', '--do_ica'      , required=False, type=bool, default=False, help="Use Standard Random Forest Regression with ICA preprocessing.")
    ap.add_argument('-rfi', '--do_rfi'      , required=False, type=bool, default=False, help="Use Standard Random Forest Regression with PCA preprocessing.")
    ap.add_argument('-gbr', '--do_gbr'      , required=False, type=bool, default=False, help="Use Gradient Boosting Regression with PCA preprocessing.")
    ap.add_argument('-rs' , '--random_state', required=False, type=int , default=42   , help="Seed for random state with which to reinitialize a specific instance.")
    ap.add_argument('-pdb', '--pdb_stop'    , required=False, type=bool, default=False, help="Stop the trace at the end with pdb.set_trace().")
    ap.add_argument('-nj' , '--n_jobs'      , required=False, type=int , default=-1   , help="Number of cores to use Default:-1.")
    ap.add_argument('-df' , '--data_file'   , required=False, type=str , default=''   , help="The csv file with the Spitzer Calibration Information.")
    args = vars(ap.parse_args())
    
    n_resamp= args['n_resamp']
    n_trees = args['n_trees']
    
    do_std  = args['do_std']
    do_pca  = args['do_pca']
    do_ica  = args['do_ica']
    do_rfi  = args['do_rfi']
    do_gbr  = args['do_gbr']
    do_pp   = args['pre_process']
    
    pdb_stop= args['pdb_stop']
    n_jobs  = args['n_jobs']
    sp_fname= args['data_file']
    
except Exception as e:
    # This section is for if/when I copy/paste the code into a ipython sesssion
    print('Error: {}'.format(e))
    
    n_resamp    = 0
    n_trees     = 100
    core        = 'A' # unknown
    do_std      = False
    do_pca      = False
    do_ica      = False
    do_rfi      = False
    do_gbr      = False
    do_pp       = True
    rand_state  = 42
    pdb_stop    = False
    n_jobs      = -1
    sp_fname    = ''

import pandas as pd
import numpy as np

import pdb

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection  import train_test_split
from sklearn.pipeline         import Pipeline
from sklearn.preprocessing    import StandardScaler, MinMaxScaler, minmax_scale
from sklearn.ensemble         import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.decomposition    import PCA, FastICA
from sklearn.externals        import joblib
from sklearn.metrics          import r2_score

from tqdm import tqdm

from glob                     import glob

from time import time
start0 = time()

def setup_features(dataRaw, label='flux', notFeatures=[], 
                    pipeline=None, verbose=True, returnAll=None):
    
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
        dataRaw = pd.read_csv(filename)
    elif isinstance(dataRaw, dict):
        dataRaw = pd.DataFrame(dataRaw)
    elif not isinstance(dataRaw, pd.DataFrame):
        raise TypeError('The input must be a `pandas.DataFrame` or a `dict` with Equal Size Entries (to convert to df here)')
    
    # WHY IS THIS ALLOWED TO NOT HAVE PARENTHESES?
    # assert isinstance(dataRaw, pd.DataFrame), 'The input must be a Pandas DataFrame or Dictionary with Equal Size Entries'
    
    inputData = dataRaw.copy()
    
    # PLDpixels = pd.DataFrame({key:dataRaw[key] for key in dataRaw.columns if 'pix' in key})
    pixCols = [colname for colname in inputData.columns if 'pix' in colname.lower() or 'pld' in colname.lower()]
    
    PLDnorm             = np.sum(np.array(inputData[pixCols]),axis=1)
    inputData[pixCols]  = (np.array(inputData[pixCols]).T / PLDnorm).T
    
    # # Overwrite the PLDpixels entries with the normalized version
    # for key in dataRaw.columns:
    #     if key in PLDpixels.columns:
    #         inputData[key] = PLDpixels[key]
    #
    # Assign the labels
    labels          = inputData[label].values
    
    # explicitly remove the label
    inputData.drop(label, axis=1, inplace=True)
    
    feature_columns = inputData.drop(notFeatures,axis=1).columns.values
    features        = inputData.drop(notFeatures,axis=1).values
    
    print('Shape of Features Array is', features.shape) if verbose else None
    
    if verbose: start = time()
    
    labels_scaled     = labels# label_scaler.fit_transform(labels[:,None]).ravel() if label_scaler   is not None else labels
    features_trnsfrmd = pipeline.fit_transform(features_scaled) if transformer is not None else features
    
    if verbose: print('took {} seconds'.format(time() - start))
    
    if returnAll == True:
        return features_trnsfrmd, dataRaw, pipeline
    
    if returnAll == 'features':
        return features_trnsfrmd
    
    if returnAll == 'with raw data':
        return features_trnsfrmd, dataRaw
    
    return features_trnsfrmd

def random_forest_wrapper(features, labels, n_trees, n_jobs, grad_boost=False, header='PCA', 
                            core_num=0, samp_num=0, loss='quantile', learning_rate=0.1, 
                            subsample=1.0, full_output=False):
    
    print('Performing {} Random Forest'.format(header))
    
    features_ = features.copy()
    labels_   = labels.copy()
    
    if grad_boost:
        rgr = GradientBoostingRegressor(n_estimators  = n_trees      , 
                                        loss          = loss         , 
                                        learning_rate = learning_rate, 
                                        subsample     = subsample    , 
                                        warm_start    = True         ,
                                        verbose       = True         )
        
        features, testX, labels, testY  = train_test_split(features_, labels_, test_size=0.25)
    else: 
        rgr = RandomForestRegressor(    n_estimators  = n_trees      ,
                                        n_jobs        = n_jobs       ,
                                        oob_score     = True         ,
                                        warm_start    = True         ,
                                        verbose       = True         )
    
    print('Feature Shape: {}\nLabel Shape: {}'.format(features.shape, labels.shape))
    
    start=time()
    rgr.fit(pca_cal_features_SSscaled, labels_SSscaled)
    
    rgr_oob = r2_score(testY, randForest.predict(testX)) if grad_boost else randForest.oob_score_ 
    rgr_Rsq = r2_score(labels_, randForest.predict(features_))
    
    test_label = {True:'Test R^2', False:'OOB'}
    
    print('{} Pretrained Random Forest:\n\t{} Score: \
                  {:.3f}%\n\tTrain R^2 score: {:.3f}%\
                  \n\tRuntime:   {:.3f} seconds'.format(header, test_label[grad_boost], 
                                                          randForest_oob*100, randForest_Rsq*100, time()-start))
    
    joblib.dump(randForest, 'randForest_{}_approach_{}trees_{}resamp_{}core.save'.format(header, n_trees, samp_num, core_num))
    
    if full_output: return randForest

if n_jobs == 1: print('WARNING: You are only using 1 core!')

# Check if requested to complete more than one operatiion
#   if so delete old instances

files_in_directory = glob('./*')

# ## Load CSVs data
flux_normalized       = ['fluxerr', 'bg_flux', 'sigma_bg_flux', 'flux']
spitzerCalNotFeatures = ['flux', 'fluxerr', 'dn_peak', 'xycov', 't_cernox', 'xerr', 'yerr', 'sigma_bg_flux']
spitzerCalFilename    = 'pmap_ch2_0p1s_x4_rmulti_s3_7.csv' if sp_fname == '' else sp_fname

spitzerCalRawData     = pd.read_csv(spitzerCalFilename)

for key in flux_normalized:
    spitzerCalRawData[key]        = spitzerCalRawData[key]      / np.median(spitzerCalRawData['flux'].values)

# spitzerCalRawData['fluxerr']        = spitzerCalRawData['fluxerr']      / np.median(spitzerCalRawData['flux'].values)
# spitzerCalRawData['bg_flux']        = spitzerCalRawData['bg_flux']      / np.median(spitzerCalRawData['flux'].values)
# spitzerCalRawData['sigma_bg_flux']  = spitzerCalRawData['sigma_bg_flux']/ np.median(spitzerCalRawData['flux'].values)
# spitzerCalRawData['flux']           = spitzerCalRawData['flux']         / np.median(spitzerCalRawData['flux'].values)

spitzerCalRawData['bmjd_err']       = np.median(0.5*np.diff(spitzerCalRawData['bmjd']))
spitzerCalRawData['np_err']         = np.sqrt(spitzerCalRawData['yerr'])

n_PLD   = len([key for key in spitzerCalRawData.keys() if 'pix' in key or 'pld' in key])

resampling_inputs = ['flux', 'xpos', 'ypos', 'xfwhm', 'yfwhm', 'bg_flux', 'bmjd', 'np'] + ['pix{}'.format(k) for k in range(1,10)]
resampling_errors = ['fluxerr', 'xerr', 'yerr', 'xerr', 'yerr', 'sigma_bg_flux', 'bmjd_err', 'np_err'] + ['fluxerr']*n_PLD

start = time()
print("Transforming Data ", end=" ")

operations = []
header = 'GBR' if do_gbr else 'RFI' if do_rfi else 'STD'

if do_pp: 
    print('Adding Standard Scaler Preprocessing to Pipeline')
    operations.append(('std_sclr', StandardScaler()))
    header     += '_SS'

if do_pca: 
    print('Adding PCA to Pipeline')
    operations.append(('pca', PCA(whiten=True)))
    header     += '_PCA'

if do_ica:
    print('Adding ICA to Pipeline')
    operations.append(('ica', FastICA(whiten=True)))
    header     += '_ICA'

pipe  = Pipeline(operations)

features, spitzerCalRawData, pipe_fitted  = setup_features( dataRaw       = spitzerCalResampled, 
                                                            transformer   = pipe, 
                                                            verbose       = verbose,
                                                            returnAll     = True)

if do_rfi: 
    importance_filename = 'randForest_STD_feature_importances.txt'
    
    if len(glob(importance_filename)) == 0: 
        raise Exception("MUST Run 'STD' operation before 'RFI', to generate file: {}".format(importance_filename))
    
    print('Computing Importances for RFI Random Forest')
    importances = np.loadtxt(importance_filename)
    indices     = np.argsort(importances)[::-1]
    
    imp_sum = np.cumsum(importances[indices])
    nImportantSamples = np.argmax(imp_sum >= 0.95) + 1

print('took {} seconds'.format(time() - start))

if 'core' in args.keys():
    core = args['core']
elif do_gbr:
    from glob import glob
    output_name    = 'randForest_{}_approach_{}trees_{}resamp_{}core.save'.format(header, n_trees, samp_num, core_num)
    existing_saves = glob('randForest_GBR_PCA_approach_{}trees_{}resamp_*core.save'.format(n_trees, n_resamp))
    
    core_nums = []
    for fname in existing_saves:
        core_nums.append(fname.split('randForest_GBR_PCA_approach_{}trees_{}resamp_'.format(n_trees, n_resamp))[-1].split('core.save')[0])
    
    core = max(core_nums) + 1
else:
    core = 'A'

pipeline_save_name      = 'spitzerCalFeature_pipeline_trnsfrmr_{}resamp_{}core.save'.format(n_resamp, core)

# Save the stack if the stack does not exist and the pipeline is not None
save_calibration_stacks = pipeline_save_name not in files_in_directory and pipe_fitted is not None

if save_calibration_stacks:
    # Need to Transform the Scaled Features based off of the calibration distribution
    joblib.dump(pipe_fitted, pca_trnsfrmr_save_name)

if n_resamp == 0:
    print('No Resampling')
    spitzerCalResampled = pd.DataFrame({colname:spitzerCalRawData[colname] for colname, colerr in tqdm(zip(resampling_inputs, resampling_errors), total=len(resampling_inputs))})
    
    features, spitzerCalRawData, pipe_fitted  = setup_features( dataRaw       = spitzerCalResampled, 
                                                                transformer   = pipe, 
                                                                verbose       = verbose,
                                                                returnAll     = True)
    
    features = features.T[indices][:nImportantSamples].T if do_rfi else features
    
    random_forest_wrapper(features, labels, n_trees, n_jobs, grad_boost=do_gbr, header=header, core_num=core, samp_num='A')

for k_samp in range(n_resamp):
    if k_samp == 0: print('Starting Resampling')
    
    spitzerCalResampled = {}
    for colname, colerr in tqdm(zip(resampling_inputs, resampling_errors), total=len(resampling_inputs)):
        if 'pix' in colname:
            spitzerCalResampled[colname]  = np.random.normal(spitzerCalRawData[colname], spitzerCalRawData[colname]*spitzerCalRawData['fluxerr'], size=(n_resamp,len(spitzerCalRawData))).flatten()
        else:
            spitzerCalResampled[colname]  = np.random.normal(spitzerCalRawData[colname], spitzerCalRawData[colerr], size=(n_resamp,len(spitzerCalRawData))).flatten()
    
    spitzerCalResampled = pd.DataFrame(spitzerCalResampled)
    
    features, spitzerCalRawData, pipe_fitted  = setup_features( dataRaw       = spitzerCalResampled, 
                                                                transformer   = pipe, 
                                                                verbose       = verbose,
                                                                returnAll     = True)
    
    features = features.T[indices][:nImportantSamples].T if do_rfi else features
    
    random_forest_wrapper(features, labels, n_trees, n_jobs, grad_boost=do_gbr, header=header, core_num=core, samp_num=k_samp)

print('\n\nFull Operation took {:.2f} minutes'.format((time() - start0)/60))
if pdb_stop: pdb.set_trace()

'''
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
'''