print('BEGIN BIG COPY PASTE ')
# This section is for if/when I copy/paste the code into a ipython sesssion
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

def setup_features(dataRaw, label='flux', notFeatures=[], pipeline=None, verbose=False, resample=False, returnAll=None):
    
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
    
    # # Overwrite the PLDpixels entries with the normalized version
    # for key in dataRaw.columns:
    #     if key in PLDpixels.columns:
    #         inputData[key] = PLDpixels[key]
    #
    # Assign the labels
    n_PLD   = len([key for key in dataRaw.keys() if 'err' not in colname.lower() and ('pix' in key.lower() or 'pld' in key.lower())])
    
    input_labels  = [colname for colname in dataRaw.columns if colname not in notFeatures and 'err' not in colname.lower()]
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
    
    labels  = inputData[label].values
    
    # explicitly remove the label
    inputData.drop(label, axis=1, inplace=True)
    
    feature_columns = inputData.drop(notFeatures,axis=1).columns.values
    features        = inputData.drop(notFeatures,axis=1).values
    
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
flux_normalized       = ['fluxerr', 'bg_flux', 'sigma_bg_flux', 'flux']
spitzerCalNotFeatures = ['flux', 'fluxerr', 'dn_peak', 'xycov', 't_cernox', 'xerr', 'yerr', 'sigma_bg_flux']
spitzerCalFilename    = 'pmap_ch2_0p1s_x4_rmulti_s3_7.csv' if sp_fname == '' else sp_fname

spitzerCalRawData     = pd.read_csv(spitzerCalFilename)

for key in flux_normalized:
    spitzerCalRawData[key]  = spitzerCalRawData[key] / np.median(spitzerCalRawData['flux'].values)

spitzerCalRawData['bmjd_err']       = np.median(0.5*np.diff(spitzerCalRawData['bmjd']))
spitzerCalRawData['np_err']         = np.sqrt(spitzerCalRawData['yerr'])

for colname in spitzerCalRawData.columns:
    if 'err' not in colname.lower() and ('pix' in colname.lower() or 'pld' in colname.lower()):
        spitzerCalRawData[colname+'_err'] = spitzerCalRawData[colname] * spitzerCalRawData['fluxerr']

start = time()
print("Transforming Data ", end=" ")

operations = []
header = 'GBR' if do_gbr else 'RFI' if do_rfi else 'STD'

pipe  = Pipeline(operations) if len(operations) else None

features, labels, pipe_fitted = setup_features( dataRaw       = spitzerCalRawData, 
                                                pipeline      = pipe, 
                                                verbose       = verbose,
                                                resample      = False,
                                                returnAll     = True)

print('END OF BIG COPY PASTE')
print('BEGIN NEW RandomizedSearchCV Section')

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score, mean_squared_error

def rmse(yt,yp): return np.sqrt(mean_squared_error(yt,yp))

grad_boost=False
header='XGB'
core = 'A'
pipe = None
features_ = features.copy()
labels_   = labels.copy()


n_iters = 100
n_jobs = -1
cv = 10
verbose = True # for RSCV
silent = True # for XGB
random_state = 42

max_depths = np.arange(1,20)
n_estimators = np.array([base*10**power for base in [1,2,5] for power in np.arange(1,5)])
learning_rates = np.array([base*10.**power for base in np.float32([1,2,5]) for power in np.arange(-4,0)])

param_dict = dict(max_depth=max_depths, learning_rate=learning_rates, n_estimators=n_estimators)

scoring = {'rmse': make_scorer(rmse, greater_is_better=False), 'r2':make_scorer(r2_score)} 

max_depth = 3
learning_rate=0.1
n_estimators = 100

xgb_rgr = xgb.XGBRegressor( max_depth = max_depth, 
                            learning_rate = learning_rate, 
                            n_estimators = n_estimators, 
                            silent = silent, 
                            n_jobs = n_jobs)

do_rscv = False
if do_rscv:
    rscv = RandomizedSearchCV(xgb_rgr, param_dict, 
                                n_iter = n_iters, 
                                scoring = scoring, 
                                n_jobs = n_jobs, 
                                cv = cv, 
                                verbose = verbose,
                                random_state = random_state, 
                                return_train_score = 'warn', 
                                refit = False,
                                pre_dispatch=1)

    rscv.fit(features_, labels_)
    
    joblib.dump(rscv, 'spitzer_cal_RSCV_fit_XGB_results.joblib.save')

do_tpot = True
if do_tpot:
    from tpot import TPOTRegressor
    n_generations = 100
    population_size = 100
    
    # param_dict['subsample'] = np.arange(0.05, 1.01, 0.05),
    param_dict['min_child_weight'] = range(1, 21)
    
    regressor_config_dict = {'xgboost.XGBRegressor': param_dict}
    scoring_ = 'r2'
    tpot_rgr = tpot.TPOTRegressor(  generations=n_generations, population_size=population_size, 
                                    scoring=scoring_, cv=cv, n_jobs=n_jobs,
                                    random_state=random_state, config_dict=regressor_config_dict,
                                    warm_start=True,
                                    periodic_checkpoint_folder='./tpot_checkpoints/',
                                    #, early_stop=None,
                                    verbosity=3)
    
    tpot_rgr.fit(features_, labels_)
    
    joblib.dump(tpot_rgr, 'spitzer_cal_TPOT_fit_XGB_results.joblib.save')