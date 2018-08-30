from argparse import ArgumentParser
from datetime import datetime
time_now = datetime.utcnow().strftime("%Y%m%d%H%M%S")

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

ap = ArgumentParser()
ap.add_argument('-ns' , '--n_resamp'    , required=False, type=int , default=0    , help="Number of resamples to perform (GBR=1; No Resamp=0).")
ap.add_argument('-nt' , '--n_trees'     , required=False, type=int , default=100  , help="Number of trees in the forest.")
ap.add_argument('-c'  , '--core'        , required=False, type=int , default=0    , help="Which Core to Use GBR only Uses 1 Core at a time.")
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
ap.add_argument('-v'  , '--verbose'     , required=False, type=str2bool, nargs='?', default=False, help="Whether to print out lots of things or just a few things")

try:
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
    
    verbose = args['verbose']
    
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
    do_pp       = False
    rand_state  = 42
    pdb_stop    = False
    n_jobs      = -1
    sp_fname    = ''
    verbose     = True

import pandas as pd
import numpy as np

import pdb

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection  import train_test_split
from sklearn.pipeline         import Pipeline
from sklearn.preprocessing    import StandardScaler, MinMaxScaler, minmax_scale
from sklearn.ensemble         import RandomForestRegressor, ExtraTreesRegressor#, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.decomposition    import PCA, FastICA
from sklearn.externals        import joblib
from sklearn.metrics          import r2_score

import xgboost as xgb

from tqdm import tqdm

from glob                     import glob

from time import time
start0 = time()

def setup_features_full(dataRaw, label='flux', notFeatures=[], pipeline=None, verbose=False, resample=False, returnAll=None):
    """Example function with types documented in the docstring.

        For production level usage: All scaling and transformations must be done 
            with respect to the calibration data distributions
        
        Args:
            features  (nD-array): Array of input raw features.
            labels    (1D-array): The second parameter.
            pipeline       (int): The first parameter.
            label_scaler   (str): The second parameter.
            feature_scaler (str): The second parameter.
        Returns:
            features_transformed, labels_scaled

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
    
    # Assign the labels
    n_PLD   = len([key for key in dataRaw.keys() if 'err' not in colname.lower() and ('pix' in key.lower() or 'pld' in key.lower())])
    
    input_labels = [colname for colname in dataRaw.columns if colname not in notFeatures and 'err' not in colname.lower()]
    errors_labels = [colname for colname in dataRaw.columns if colname not in notFeatures and 'err'     in colname.lower()]
    
    # resampling_inputs = ['flux', 'xpos', 'ypos', 'xfwhm', 'yfwhm', 'bg_flux', 'bmjd', 'np'] + ['pix{}'.format(k) for k in range(1,10)]
    # resampling_errors = ['fluxerr', 'xerr', 'yerr', 'xerr', 'yerr', 'sigma_bg_flux', 'bmjd_err', 'np_err'] + ['fluxerr']*n_PLD
    
    start = time()
    
    if resample:
        print("Resampling ", end=" ")
        inputData = pd.DataFrame({colname:np.random.normal(dataRaw[colname], dataRaw[colerr]) \
                                    for colname, colerr in tqdm(zip(input_labels, errors_labels), total=len(input_labels))
                                 })        
    
        print("took {} seconds".format(time() - start))
    else:
        inputData = pd.DataFrame({colname:dataRaw[colname] for colname in input_labels})
    
    if label in inputData.keys():
        labels = inputData[label]
        # explicitly remove the label
        inputData.drop(label, axis=1, inplace=True)
    else:
        labels = np.ones(len(inputData))
    
    feature_columns = inputData.drop(notFeatures,axis=1).columns
    features = inputData[feature_columns]# inputData.drop(notFeatures,axis=1)
    
    if verbose: print('Shape of Features Array is', features.shape)
    
    if verbose: start = time()
    
    # labels_scaled     = labels# label_scaler.fit_transform(labels[:,None]).ravel() if label_scaler   is not None else labels
    features_trnsfrmd = pipeline.fit_transform(features) if pipeline is not None else features
    
    if verbose: print('took {} seconds'.format(time() - start))
    
    collection = features_trnsfrmd, labels
    
    if returnAll == True:
        collection = features_trnsfrmd, labels, pipeline
    
    if returnAll == 'features':
        collection = features_trnsfrmd
    
    if returnAll == 'with raw data':
        collection.append(dataRaw)
    
    return collection

def setup_features_basic(dataRaw, label='flux', notFeatures=[], pipeline=None, verbose=False, resample=False, returnAll=None):
    inputData = dataRaw.copy()
    
    pixCols = [colname for colname in inputData.columns if 'pix' in colname.lower() or 'pld' in colname.lower()]
    
    input_labels = [colname for colname in dataRaw.columns if colname not in notFeatures and 'err' not in colname.lower()]
    input_labels = sorted(input_labels)
    inputData = pd.DataFrame({colname:dataRaw[colname] for colname in input_labels})
    
    PLDnorm             = np.sum(np.array(inputData[pixCols]),axis=1)
    inputData[pixCols]  = (np.array(inputData[pixCols]).T / PLDnorm).T
    
    if label in inputData.keys():
        labels = pd.DataFrame(inputData[label], columns=[label])
        
        # explicitly remove the label
        inputData.drop(label, axis=1, inplace=True)
    else:
        labels = np.ones(len(inputData))
    
    features = inputData.drop(notFeatures,axis=1)
    
    return features, labels

def random_forest_wrapper(features, labels, n_trees, n_jobs, grad_boost=False, header='PCA', 
                            core_num=0, samp_num=0, loss='quantile', learning_rate=0.1, 
                            max_depth=3, subsample=1.0, full_output=False, verbose=False):
    
    print('Performing {} Random Forest'.format(header))
    
    features_ = features.copy()
    labels_   = labels.copy()
    
    if grad_boost:
        rgr = xgb.XGBRegressor( max_depth = max_depth, 
                                learning_rate = learning_rate, 
                                n_estimators = n_trees, 
                                silent = not verbose, 
                                n_jobs = n_jobs)
        
        # objective='reg:linear', booster='gbtree', 
        # gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,
        # colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1,
        # scale_pos_weight=1, base_score=0.5, random_state=0, seed=None,
        # missing=None
        
        features, testX, labels, testY  = train_test_split(features_, labels_, test_size=0.25)
    else: 
        rgr = RandomForestRegressor(    n_estimators  = n_trees      ,
                                        n_jobs        = n_jobs       ,
                                        oob_score     = True         ,
                                        warm_start    = True         ,
                                        verbose       = verbose      )
    
    if verbose: print('Feature Shape: {}\nLabel Shape: {}'.format(features.shape, labels.shape))
    
    if verbose: start=time()
    
    rgr.fit(features, labels)
    
    rgr_oob = r2_score(testY, rgr.predict(testX)) if grad_boost else rgr.oob_score_ 
    rgr_Rsq = r2_score(labels_, rgr.predict(features_))
    
    test_label = {True:'Test R^2', False:'OOB'}
    
    if verbose: print('{} Pretrained Random Forest:\n\t{} Score: \
                       {:.3f}%\n\tTrain R^2 score: {:.3f}%\
                       \n\tRuntime:   {:.3f} seconds'.format(header, test_label[grad_boost], 
                                                              rgr_oob*100, rgr_Rsq*100, time()-start))
    
    output_savename = 'randForest_{}_approach_{}trees_{}resamp_{}core.save'.format(header, n_trees, samp_num, core_num)
    print('Storing New File to {}'.format(output_savename))
    joblib.dump(rgr, output_savename)
    
    if full_output: return rgr

if n_jobs == 1: print('WARNING: You are only using 1 core!')

# Check if requested to complete more than one operatiion
#   if so delete old instances

files_in_directory = glob('./*')

# ## Load CSVs data
flux_normalized = ['fluxerr', 'bg_flux', 'sigma_bg_flux', 'flux']
spitzerCalNotFeatures = ['flux', 'fluxerr', 'bmjd', 'dn_peak', 'xycov', 't_cernox', 'xerr', 'yerr', 'sigma_bg_flux']
spitzerCalFilename = 'pmap_ch2_0p1s_x4_rmulti_s3_7.csv' if sp_fname == '' else sp_fname

spitzerCalKeepFeatures = ['xpos', 'ypos', 'np', 'xfwhm', 'yfwhm', 'bg_flux', #'bmjd', 
                            'pix1', 'pix2', 'pix3', 'pix4', 'pix5', 'pix6', 'pix7', 'pix8', 'pix9']

spitzerCalRawData = pd.read_csv(spitzerCalFilename)

for key in flux_normalized:
    spitzerCalRawData[key] = spitzerCalRawData[key] / np.median(spitzerCalRawData['flux'].values)

spitzerCalRawData['fluxerr'] = spitzerCalRawData['fluxerr'] / np.median(spitzerCalRawData['flux'].values)
spitzerCalRawData['bg_flux'] = spitzerCalRawData['bg_flux'] / np.median(spitzerCalRawData['flux'].values)
spitzerCalRawData['sigma_bg_flux'] = spitzerCalRawData['sigma_bg_flux'] / np.median(spitzerCalRawData['flux'].values)
spitzerCalRawData['flux'] = spitzerCalRawData['flux'] / np.median(spitzerCalRawData['flux'].values)

spitzerCalRawData['bmjd_err'] = np.median(0.5*np.diff(spitzerCalRawData['bmjd']))
spitzerCalRawData['np_err'] = np.sqrt(spitzerCalRawData['yerr'])

for colname in spitzerCalRawData.columns:
    if 'err' not in colname.lower() and ('pix' in colname.lower() or 'pld' in colname.lower()):
        spitzerCalRawData[colname+'_err'] = spitzerCalRawData[colname] * spitzerCalRawData['fluxerr']

spitzer_cal_features, spitzer_cal_labels = setup_features_basic(spitzerCalRawData[['flux']+spitzerCalKeepFeatures])

start = time()
print("Transforming Data ", end=" ")

operations = []
header = 'GBR' if do_gbr else 'RFI' if do_rfi else 'STD'

if do_pp: 
    print('Adding Standard Scaler Preprocessing to Pipeline')
    operations.append(('std_sclr', StandardScaler()))
    header += '_SS'

if do_pca: 
    print('Adding PCA to Pipeline')
    operations.append(('pca', PCA(whiten=True)))
    header += '_PCA'

if do_ica:
    print('Adding ICA to Pipeline')
    operations.append(('ica', FastICA(whiten=True)))
    header += '_ICA'

pipe  = Pipeline(operations) if len(operations) else None

if do_rfi: 
    importance_filename = 'randForest_STD_feature_importances.txt'
    
    if len(glob(importance_filename)) == 0: 
        raise Exception("MUST Run 'STD' operation before 'RFI', to generate file: {}".format(importance_filename))
    
    print('Computing Importances for RFI Random Forest')
    importances = np.loadtxt(importance_filename)
    indices = np.argsort(importances)[::-1]
    
    imp_sum = np.cumsum(importances[indices])
    nImportantSamples = np.argmax(imp_sum >= 0.95) + 1

print('took {} seconds'.format(time() - start))

if 'core' in args.keys():
    core = args['core']
elif do_gbr:
    from glob import glob
    output_name = 'randForest_{}_approach_{}trees_{}resamp_{}core.save'.format(header, n_trees, samp_num, core_num)
    existing_saves = glob('randForest_{}_approach_{}trees_{}resamp_*core.save'.format(header, n_trees, n_resamp))
    
    core_nums = []
    for fname in existing_saves:
        core_nums.append(fname.split('randForest_{}_approach_{}trees_{}resamp_'.format(header, n_trees, n_resamp))[-1].split('core.save')[0])
    
    core = max(core_nums) + 1
else:
    core = 'A'

if n_resamp == 0:
    print('No Resampling')
    
    features, labels, pipe_fitted = setup_features( dataRaw       = spitzerCalRawData, 
                                                    pipeline      = pipe, 
                                                    verbose       = verbose,
                                                    notFeatures   = spitzerCalNotFeatures,
                                                    resample      = False,
                                                    returnAll     = True)
    
    features = features.T[indices][:nImportantSamples].T if do_rfi else features
    
    random_forest_wrapper(features, labels, n_trees, n_jobs, grad_boost=do_gbr, header=header, core_num=core, samp_num='no_', verbose=verbose)
    
    pipeline_save_name      = 'spitzerCalFeature_pipeline_trnsfrmr_no_resamp_{}core.save'.format(core)
    print('Saving NO RESAMP Pipeline as {}'.format(pipeline_save_name))
    
    # Save the stack if the stack does not exist and the pipeline is not None
    save_calibration_stacks = pipeline_save_name not in files_in_directory and pipe_fitted is not None
    
    # Need to Transform the Scaled Features based off of the calibration distribution
    if save_calibration_stacks: joblib.dump(pipe_fitted, pipeline_save_name)

for k_samp in tqdm(range(n_resamp),total=n_resamp):
    if k_samp == 0: print('Starting Resampling')
    
    spitzer_cal_features, spitzer_cal_labels, pipe_fitted = setup_features( dataRaw       = spitzerCalRawData, 
                                                    pipeline      = pipe   ,
                                                    verbose       = verbose,
                                                    resample      = True   ,
                                                    returnAll     = True   )
    
    features = features.T[indices][:nImportantSamples].T if do_rfi else features
    
    random_forest_wrapper(features, labels, n_trees, n_jobs, grad_boost=do_gbr, header=header, core_num=core, samp_num=k_samp, verbose=verbose)
    
    pipeline_save_name      = 'spitzerCalFeature_pipeline_trnsfrmr_{}resamp_{}core.save'.format(k_samp, core)
    
    print('Saving SAMP {} Pipeline as {} on Core {}'.format(k_samp, pipeline_save_name, core))
    # Save the stack if the stack does not exist and the pipeline is not None
    save_calibration_stacks = pipeline_save_name not in files_in_directory and pipe_fitted is not None
    
    # Need to Transform the Scaled Features based off of the calibration distribution
    if save_calibration_stacks: joblib.dump(pipe_fitted, pipeline_save_name)        

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
            features_transformed, labels_scaled

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
if pmap_xo3b:
    from glob import glob
    
    xo3b_files = glob('XO3_Data/XO3_r464*.csv')
    
    for fname in xo3b_files:
        if 'NALU' in fname:                     
            xo3b_files.remove(fname)     
    
    for fname in xo3b_files:
        if 'NALU' in fname:                     
            xo3b_files.remove(fname)     
    
    for fname in tqdm(xo3b_files, total=len(xo3b_files)):
        key = fname.split('_')[-1].split('.')[0]
        
        med_flux = np.median(xo3b_data[key]['raw']['flux'].values)
        
        xo3b_data[key] = {'raw':pd.read_csv(fname)}
        features, labels = setup_features_basic( dataRaw=val['raw'][['flux']+spitzerCalKeepFeatures])
        
        xo3b_data[key]['raw']['fluxerr'] = xo3b_data[key]['raw']['fluxerr'] / med_flux
        xo3b_data[key]['raw']['bg_flux'] = xo3b_data[key]['raw']['bg_flux'] / med_flux
        xo3b_data[key]['raw']['flux'] = xo3b_data[key]['raw']['flux'] / med_flux
        xo3b_data[key]['features'] = features
        xo3b_data[key]['labels'] = labels
        xo3b_data[key]['pmap'] = xgb_rgr.predict(features)
