from multiprocessing import set_start_method, cpu_count
set_start_method('forkserver')

import os
os.environ["OMP_NUM_THREADS"] = str(cpu_count())  # or to whatever you want

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
        # scale_pos_weight=1, base_score=0.5, RANDOM_STATE=0, seed=None,
        # missing=None
        
        features, testX, labels, testY  = train_test_split(features_, labels_, TEST_SIZE=0.25)
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
print('BEGIN NEW HyperParameter Optimization.')

from sklearn.metrics import r2_score


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

def generate_dataset(size=10000, op='sum', n_features=2):
    """ Generate dataset for NALU toy problem
    
    Arguments:
    size - number of samples to generate
    op - the operation that the generated data should represent. sum | prod 
    Returns:
    X - the dataset
    Y - the dataset labels
    """
    
    X = np.random.randint(9, size=(size, n_features))
    
    if op == 'prod':
        Y = np.prod(X, axis=1, keepdims=True)
    else:
        Y = np.sum(X, axis=1, keepdims=True)
    
    return X, Y

if __name__ == "__main__":
    from tqdm import tqdm
    from argparse import ArgumentParser
    
    ap = ArgumentParser()
    
    ap.add_argument('-id', '--import_dir', type=str, required=True, help='The tensorflow ckpt save file')
    ap.add_argument('-ed', '--export_dir', type=str, required=False, default='nalu_tf_save_dir/saves_warm_start_{}'.format(time()), help='The tensorflow ckpt save file')
    ap.add_argument('-nnl', '--n_nalu_layers', type=int, required=False, default=1, help='Whether to use 1 (default), 2, or ... N NALU layers.')
    ap.add_argument('-nnn', '--n_nalu_neurons', type=int, required=False, default=1, help='How many features on the second NALU layer')
    ap.add_argument('-ne', '--n_epochs', type=int, required=False, default=200, help='Number of N_EPOCHS to train the network with.')
    ap.add_argument('-nc', '--n_classes', type=int, required=False, default=1, help='n_classes == 1 for Regression (default); > 1 for Classification.')
    ap.add_argument('-bs', '--batch_size', type=int, required=False, default=32, help='Batch size: number of samples per batch.')
    ap.add_argument('-lr', '--learning_rate', type=float, required=False, default=1e-3, help='Learning rate: how fast the optimizer moves up/down the gradient.')
    ap.add_argument('-ts', '--test_size', type=float, required=False, default=0.75, help='How much to split the train / test ratio')
    ap.add_argument('-rs', '--random_state', type=int, required=False, default=42, help='Integer value to initialize train/test splitting randomization')
    
    args = vars(ap.parse_args())
    
    IMPORT_DIR = args['import_dir']
    EXPORT_DIR = args['export_dir']
    N_NALU_LAYERS = args['n_nalu_layers']
    N_NALU_NEURONS = args['n_nalu_neurons']
    N_CLASSES = args['n_classes'] # = 1 for regression
    TEST_SIZE = args['test_size']
    RANDOM_STATE = args['random_state']
    
    N_EPOCHS = args['n_epochs']
    LEARNING_RATE = args['learning_rate']
    BATCH_SIZE = args['batch_size']
    
    WS_TIME = IMPORT_DIR.split('/')[1].split('_')[1]
    EXPORT_DIR = EXPORT_DIR + '_WS{}_nnl{}_nnn{}_nc{}_bs{}_lr{}_ne{}_ts{}_rs{}/'.format(WS_TIME, N_NALU_LAYERS, N_NALU_NEURONS, N_CLASSES, 
                                                                                        BATCH_SIZE, LEARNING_RATE, N_EPOCHS, TEST_SIZE, RANDOM_STATE)
    
    print("Saving models to path: {}".format(EXPORT_DIR))
    
    idx_train, idx_test = train_test_split(np.arange(labels.size), test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_data, Y_data = features[idx_train], labels[idx_train][:,None]
    
    N_FEATURES = X_data.shape[-1]
    LAST_BIT = X_data.shape[0]-BATCH_SIZE*(X_data.shape[0]//BATCH_SIZE)
    
    # Force integer number of batches total by dropping last "<BATCH_SIEZ" number of samples
    X_data_use = X_data[:-LAST_BIT].copy()
    Y_data_use = Y_data[:-LAST_BIT].copy()
    
    N_FEATURES = X_data.shape[-1]
    
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
    
    sess_config = tf.ConfigProto(
                    device_count={"CPU": cpu_count()},
                    inter_op_parallelism_threads=cpu_count(),
                    intra_op_parallelism_threads=cpu_count())
    
    with tf.Session(config=sess_config) as sess:
        saver.restore(sess, IMPORT_DIR)
        
        for ep in tqdm(range(N_EPOCHS)):
            i = 0
            gts = 0
            
            while i < len(X_data_use):
                xs, ys = X_data_use[i:i+BATCH_SIZE], Y_data_use[i:i+BATCH_SIZE]
                
                _, ys_pred, l = sess.run([train_op, Y_pred, loss], 
                        feed_dict={X: xs, Y_true: ys})
                
                # calculate number of correct predictions from batch
                gts += np.sum(np.isclose(ys, ys_pred, atol=1e-4, rtol=1e-4)) 
                
                i += BATCH_SIZE
            
            ytest_pred = Y_pred.eval(feed_dict={X: features[idx_test]})
            test_r2 = r2_score(labels[idx_test][:,None], ytest_pred)
            # print("Test R2 Score: {}".format(test_r2_score))
            
            acc = gts/len(Y_data_use)
            train_r2 = r2_score(ys, ys_pred)
            print('epoch {}, loss: {:.5}, accuracy: {:.5}, Batch R2: {:.5}, Test R2: {:.5}'.format(ep, l, acc, train_r2, test_r2))
            
            file_name = "model_epoch{}_l{:.5}_a{:.5}_BatchR2-{:.5}_TestR2-{:.5}.ckpt".format(ep, l, acc, train_r2, test_r2)
            save_path = saver.save(sess, EXPORT_DIR + file_name)
            # print("Model saved in path: %s" % save_path)
        
        ep = '_FINAL'
        
        file_name = "model_epoch{}_l{:.5}_a{:.5}_BatchR2-{:.5}_TestR2-{:.5}.ckpt".format(ep, l, acc, train_r2, test_r2)
        save_path = saver.save(sess, EXPORT_DIR + file_name)
        print("Model saved in path: {}".format(save_path))
