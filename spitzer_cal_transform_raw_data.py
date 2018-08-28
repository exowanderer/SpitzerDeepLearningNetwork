from multiprocessing import set_start_method, cpu_count
#set_start_method('forkserver')

import os
os.environ["OMP_NUM_THREADS"] = str(cpu_count())  # or to whatever you want

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

ap.add_argument('-ds', '--data_set', type=str, required=False, default='', help='The csv file containing the data with which to train.')
ap.add_argument('-rs', '--random_state', type=int, required=False, default=42, help='Integer value to initialize train/test splitting randomization')
ap.add_argument('-pp', '--pre_process', type=str2bool, nargs='?', required=False, default=True, help='Toggle whether to MinMax-preprocess the features.')
ap.add_argument('-pca', '--pca_transform', type=str2bool, nargs='?', required=False, default=True, help='Toggle whether to PCA-pretransform the features.')
ap.add_argument('-v', '--verbose', type=str2bool, nargs='?', required=False, default=False, help='Whether to set verbosity = True or False (default).')

try:
    args = vars(ap.parse_args())
except:
    args = {}
    args['random_state'] = ap.get_default('random_state')
    args['pre_process'] =  ap.get_default('pre_process')
    args['pca_transform'] =  ap.get_default('pca_transform')
    args['verbose'] = ap.get_default('verbose')
    args['data_set'] = ap.get_default('data_set')

do_pp = args['pre_process']
do_pca = args['pca_transform']

verbose = args['verbose']
data_set_fname = args['data_set']

print('BEGIN BIG COPY PASTE ')

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
    # if label in notFeatures: notFeatures.remove(label)
    
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
    
    labels  = dataRaw[label].values
    
    # explicitly remove the label
    if label in inputData.columns: inputData.drop(label, axis=1, inplace=True)
    
    feature_columns = [colname for colname in inputData.columns if colname not in notFeatures]
    print('\n\n','flux' in notFeatures, 'flux' in feature_columns, '\n\n')
    features = inputData[feature_columns].values
    
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

# ## Load CSVs data
tobe_flux_normalized = ['fluxerr', 'bg_flux', 'sigma_bg_flux', 'flux']
spitzerCalNotFeatures = ['flux', 'fluxerr', 'dn_peak', 'xycov', 't_cernox', 'xerr', 'yerr', 'sigma_bg_flux']
spitzerCalFilename = 'pmap_ch2_0p1s_x4_rmulti_s3_7.csv' if data_set_fname == '' else data_set_fname

spitzerCalRawData = pd.read_csv(spitzerCalFilename)

for key in tobe_flux_normalized:
    spitzerCalRawData[key] = spitzerCalRawData[key] / np.median(spitzerCalRawData['flux'].values)

spitzerCalRawData['bmjd_err'] = np.median(0.5*np.diff(spitzerCalRawData['bmjd']))
spitzerCalRawData['np_err'] = np.sqrt(spitzerCalRawData['yerr'])

for colname in spitzerCalRawData.columns:
    if 'err' not in colname.lower() and ('pix' in colname.lower() or 'pld' in colname.lower()):
        spitzerCalRawData[colname+'_err'] = spitzerCalRawData[colname] * spitzerCalRawData['fluxerr']

start = time()
print("Transforming Data ", end=" ")

operations = []
# header = 'GBR' if do_gbr else 'RFI' if do_rfi else 'STD'

pipe = Pipeline(operations) if len(operations) else None

not_features_now = []
for feat_name in spitzerCalNotFeatures:
    if feat_name in spitzerCalRawData.columns:
        not_features_now.append(feat_name)

features, labels, pipe_fitted = setup_features( dataRaw = spitzerCalRawData, 
                                                pipeline = pipe, 
                                                verbose = verbose,
                                                notFeatures = not_features_now,
                                                resample = False,
                                                returnAll = True)

n_samples = len(features)

std_scaler_from_raw = StandardScaler()
pca_transformer_from_std_scaled = PCA()
minmax_scaler_transformer_raw = MinMaxScaler()
minmax_scaler_transformer_pca = MinMaxScaler()

operations = []
operations.append(('std_scaler', StandardScaler()))
operations.append(('pca_transform', PCA()))
operations.append(('minmax_scaler', MinMaxScaler()))

full_pipe = Pipeline(operations)
full_pipe_transformed_features = full_pipe.fit_transform(features)

standard_scaled_features = std_scaler.fit_transform(features)
pca_standard_scaled_features = pca_transformer.fit_transform(standard_scaled_features)
minmax_scaled_features_raw = minmax_scaler_transformer_raw.fit_transform(features)
minmax_scaled_features_pca = minmax_scaler_transformer_pca.fit_transform(pca_standard_scaled_features)

joblib.dump(full_pipe, 'pmap_full_pipe_transformer_16features.joblib.save')
joblib.dump(std_scaler_from_raw, 'pmap_standard_scaler_transformer_16features.joblib.save')
joblib.dump(pca_transformer_from_std_scaled, 'pmap_pca_transformer_from_stdscaler_16features.joblib.save')
joblib.dump(minmax_scaler_transformer_raw, 'pmap_minmax_scaler_transformer_from_raw_16features.joblib.save')
joblib.dump(minmax_scaler_transformer_pca, 'pmap_minmax_scaler_transformer_from_pca_16features.joblib.save')

pd.Dataframe(labels, index=range(n_samples)).to_csv('pmap_raw_labels.csv')
pd.DataFrame(features, index=range(n_samples)).to_csv('pmap_raw_16features.csv')
pd.DataFrame(full_pipe_transformed_features, index=range(n_samples)).to_csv('pmap_full_pipe_transformed_16features.csv')
pd.DataFrame(standard_scaled_features, index=range(n_samples)).to_csv('pmap_standard_scaled_16features.csv')
pd.DataFrame(pca_standard_scaled_features, index=range(n_samples)).to_csv('pmap_pca_transformed_from_stdscaler_16features.csv')
pd.DataFrame(minmax_scaled_features_raw, index=range(n_samples)).to_csv('pmap_minmax_transformed_from_raw_16features.csv')
pd.DataFrame(minmax_scaled_features_pca, index=range(n_samples)).to_csv('pmap_minmax_transformed_from_pca_16features.csv')
