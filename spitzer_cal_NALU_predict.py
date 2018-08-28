from multiprocessing import set_start_method, cpu_count
# set_start_method('forkserver')

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

ap.add_argument('-d', '--directory', type=str, required=True, help='The tensorflow ckpt save file')
ap.add_argument('-ds', '--data_set', type=str, required=False, default='', help='The csv file containing the data to predict with')
ap.add_argument('-sn', '--save_name', type=str, required=False, default='', help='The csv file containing the data to predict with')
ap.add_argument('-v', '--verbose', type=str2bool, nargs='?', required=False, default=False, help='Whether to set verbosity = True or False (default)')

try:
    args = vars(ap.parse_args())
except:
    args = {}
    args['data_set'] = ap.get_default('data_set') # test case 'XO3_Data/XO3_r46468096.csv'
    args['save_name'] = ap.get_default('save_name')
    args['directory'] = ap.get_default('directory')
    args['verbose'] = ap.get_default('verbose')

data_set_fname = args['data_set']
save_filename = args['save_name']

default_data = data_set_fname is ap.get_default('data_set')
save_data = save_filename is not ap.get_default('save_name')

verbose = args['verbose']

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

print('END OF BIG COPY PASTE')

print('BEGIN NEW HyperParameter Optimization.')

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

''' NALU: Nearual Arithmentic Logical Unit
        
        NALU uses memory and logic gates to train a unique TF layer to modify the gradients of the weights.
        This seems to be very smilar to a LSTM layer, but for a non-RNN.
        This code has been specifically implemented with tensorflow.
        
        Code source: https://github.com/grananqvist/NALU-tf
        Original paper: https://arxiv.org/abs/1808.00508 (Trask et al.)
'''
import numpy as np
import tensorflow as tf

def nalu(input_layer, num_outputs):
    """ Neural Arithmetic Logic Unit tesnorflow layer
    
    Arguments:
    input_layer - A Tensor representing previous layer
    num_outputs - number of ouput units 
    
    Returns:
    A tensor representing the output of NALU
    """
    
    shape = (int(input_layer.shape[-1]), num_outputs)
    
    # define variables
    W_hat = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
    M_hat = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
    G = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
    
    # operations according to paper
    W = tf.tanh(W_hat) * tf.sigmoid(M_hat)
    m = tf.exp(tf.matmul(tf.log(tf.abs(input_layer) + 1e-7), W))
    g = tf.sigmoid(tf.matmul(input_layer, G))
    a = tf.matmul(input_layer, W)
    out = g * a + (1 - g) * m
    
    return out

def median_sub_nan(thingy):
    thingy[np.isnan(thingy)] = np.median(thingy[~np.isnan(thingy)])
    return thingy

if __name__ == "__main__":
    N_FEATURES = features.shape[-1]
    IMPORT_DIR = args['directory']
    # EXAMPLE_IMPORT_DIR = 'saves_1535336013.53251_WS13413134.134134_nnl2_nnn1_nc1_bs32_lr0.001_ne2000_ts0.75_rs42'
    
    dir_parts = IMPORT_DIR.split('/')[1].split('_')
    
    if 'WS' not in IMPORT_DIR:
        add_one = True
    else:
        add_one = False
    
    # short_names = ['nnl', 'nnn', 'nc', 'ts', 'rs', 'ne', 'lr' , 'bs']
    args_names = ['n_nalu_layers', 'n_nalu_neurons', 'n_epochs', 'n_classes', 'batch_size', 'learning_rate', 'test_size', 'random_state']
    
    short_names = {}
    for arg_name in args_names:
        arg_name_parts = arg_name.split('_')
        short_name = ''
        for arg_name_part in arg_name_parts:
            short_name = short_name + arg_name_part[0]
        short_names[arg_name] = short_name
    
    short_names['pre_process'] = 'PP'
    short_names['pca_transform'] = 'PCA'
    
    args = { arg_name: IMPORT_DIR.split(short_names[arg_name])[-1].split('_')[0].split('/')[0] for k, arg_name in enumerate(args_names) }
    
    N_NALU_LAYERS = int(args['n_nalu_layers'])
    N_NALU_NEURONS = int(args['n_nalu_neurons']) if int(args['n_nalu_neurons']) > 0 else N_FEATURES
    N_CLASSES = int(args['n_classes']) # = 1 for regression
    TEST_SIZE = float(args['test_size'])
    RANDOM_STATE = int(args['random_state'])
    
    N_EPOCHS = int(args['n_epochs'])
    LEARNING_RATE = float(args['learning_rate'])
    BATCH_SIZE = int(args['batch_size'])
    
    do_pp = bool(args['pre_process'])
    do_pca = bool(args['pca_transform'])
    
    ''' PCA Requires Standard Scaling
            Here we Standard Scale, and then immediate PCA transform, simultaneously
    '''
    features_in = PCA().fit_transform(StandardScaler().fit_transform(features)) if do_pca else features.copy()
    
    ''' Because PCA requires standard scaling, let's save some CPUs by ignoring this pre-processing
    '''
    if do_pp: features_in = MinMaxScaler().fit_transform(features_in)
    
    if default_data:
        ''' Testing the network success'''
        idx_train, idx_test = train_test_split(np.arange(labels.size), test_size=0.75, random_state=42)
        X_data, Y_data = features_in[idx_train], labels[idx_train][:,None]
    else:
        ''' Predicting a future solution'''
        X_data, Y_data = features_in, labels
    
    ''' Construct Load Path '''
    print("Loaded model stored in path: {}".format(IMPORT_DIR))
    
    with tf.device("/cpu:0"):
        # tf.reset_default_graph()
        
        # define placeholders and network
        X = tf.placeholder(tf.float32, shape=[None, N_FEATURES])
        Y_true = tf.placeholder(tf.float32, shape=[None, 1])
        
        # Setup NALU Layers
        nalu_layers = {'nalu0':nalu(X,N_NALU_NEURONS)}
        for kn in range(1, N_NALU_LAYERS):
            nalu_layers['nalu{}'.format(kn)] = nalu(nalu_layers['nalu{}'.format(kn-1)], N_NALU_NEURONS)
        
        Y_pred = nalu(nalu_layers['nalu{}'.format(N_NALU_LAYERS-1)], N_CLASSES) # N_CLASSES = 1 for regression
        
        # loss and train operations
        loss = tf.nn.l2_loss(Y_pred - Y_true) # NALU uses mse
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        train_op = optimizer.minimize(loss)
        
        # Add an op to initialize the variables.
        init_op = tf.global_variables_initializer()
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
    
    sess_config = tf.ConfigProto(device_count={"CPU": cpu_count()},
                            inter_op_parallelism_threads=2,
                            intra_op_parallelism_threads=1)
    
    with tf.Session(config=sess_config) as sess:
        # Restore variables from disk.
        saver.restore(sess, IMPORT_DIR)
        print("Model restored.")
        # Check the values of the variables
        ys_pred = Y_pred.eval(feed_dict={X: X_data})
        
        if np.isnan(ys_pred).any(): ys_pred = median_sub_nan(ys_pred)
        
        print("\n\nR2_Test: {}\n\n".format(r2_score(Y_data,ys_pred)))
        
        if save_data: 
            print('Saving to {}'.format(save_filename))
            DataFrame(ys_pred, index=range(ys_pred.size)).to_csv(save_filename)

