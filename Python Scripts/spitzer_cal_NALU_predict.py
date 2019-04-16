from multiprocessing import set_start_method, cpu_count
set_start_method('forkserver')

import os
os.environ["OMP_NUM_THREADS"] = str(cpu_count())  # or to whatever you want

import numpy as np
import pandas as pd
import tensorflow as tf

from argparse import ArgumentParser
from datetime import datetime
time_now = datetime.utcnow().strftime("%Y%m%d%H%M%S")

from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def establish_inputs_from_filename(args_):
    IMPORT_DIR = args_['trained_DNN_ckpt']
    # EXAMPLE_IMPORT_DIR = 'saves_1535336013.53251_WS13413134.134134_nnl2_nnn1_nc1_bs32_lr0.001_ne2000_ts0.75_rs42'
    
    dir_parts = IMPORT_DIR.split('/')[1].split('_')
    
    if 'WS' not in IMPORT_DIR:
        add_one = True
    else:
        add_one = False
    
    # short_names = ['nnl', 'nnn', 'nc', 'ts', 'rs', 'ne', 'lr' , 'bs']
    args_names = ['n_nalu_layers', 'n_nalu_neurons', 'n_epochs', 'n_classes', 
                    'batch_size', 'learning_rate', 'test_size', 'random_state', 
                    'pre_process', 'pca_transform']
    
    short_names = {}
    for arg_name in args_names:
        arg_name_parts = arg_name.split('_')
        short_name = ''
        for arg_name_part in arg_name_parts:
            short_name = short_name + arg_name_part[0]
        short_names[arg_name] = short_name
    
    short_names['pre_process'] = 'PP'
    short_names['pca_transform'] = 'PCA'
    
    args_ = { arg_name: IMPORT_DIR.split(short_names[arg_name])[-1].split('_')[0].split('/')[0] \
                                for k, arg_name in enumerate(args_names) }
    
    N_NALU_LAYERS = int(args_['n_nalu_layers'])
    N_NALU_NEURONS = int(args_['n_nalu_neurons']) if int(args_['n_nalu_neurons']) > 0 else N_FEATURES
    N_CLASSES = int(args_['n_classes']) # = 1 for regression
    TEST_SIZE = float(args_['test_size'])
    RANDOM_STATE = int(args_['random_state'])
    
    N_EPOCHS = int(args_['n_epochs'])
    LEARNING_RATE = float(args_['learning_rate'])
    BATCH_SIZE = int(args_['batch_size'])
    
    DO_PP = bool(int(args_['pre_process']))
    DO_PCA = bool(int(args_['pca_transform']))
    
    return IMPORT_DIR, N_NALU_LAYERS, N_NALU_NEURONS, N_CLASSES, TEST_SIZE, \
            RANDOM_STATE, N_EPOCHS, LEARNING_RATE, BATCH_SIZE, DO_PP, DO_PCA

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return True
        # raise argparse.ArgumentTypeError('Boolean value expected.')

ap = ArgumentParser()

ap.add_argument('-tdc', '--trained_DNN_ckpt', type=str, required=False, default='', help='The tensorflow ckpt save file with the trained network.')
ap.add_argument('-ds', '--data_set', type=str, required=False, default='', help='The csv file containing the data with which to predict.')
ap.add_argument('-sn', '--save_name', type=str, required=False, default='', help='The csv file containing the data with which to predict.')
ap.add_argument('-ct', '--check_test_r2', type=str2bool, nargs='?', required=False, default=False, help='Flag if input data is training data.')
ap.add_argument('-v', '--verbose', type=str2bool, nargs='?', required=False, default=False, help='Whether to set verbosity = True or False (default).')

try:
    args = vars(ap.parse_args())
except Exception as e:
    args = {}
    args['data_set'] = ap.get_default('data_set') # test case 'XO3_Data/XO3_r46468096.csv'
    args['save_name'] = ap.get_default('save_name')
    args['trained_DNN_ckpt'] = ap.get_default('trained_DNN_ckpt')
    args['verbose'] = ap.get_default('verbose')
    args['check_test_r2'] = ap.get_default('check_test_r2')

''' Default ("best") Trained TF-DNN ckpt (so far)'''
if args['trained_DNN_ckpt'] is ap.get_default('trained_DNN_ckpt'):
    base_dir = 'nalu_tf_save_dir/'
    sub_dir = 'saves_20180829053116_nnl1_nnn16_nc1_bs32_lr0.001_ne2000_ts0.75_rs42_PP1_PCA0/'
    file_name = 'model_epoch_FINAL_l1.1785e-05_a0.20335_BatchR2-0.97701_TestR2-0.97597.ckpt'
    
    args['trained_DNN_ckpt'] = base_dir + sub_dir + file_name

check_test_r2 = args['check_test_r2']
data_set_fname = args['data_set'] if not check_test_r2 else ap.get_default('data_set')
save_filename = args['save_name']

save_data = save_filename is not ap.get_default('save_name')

verbose = args['verbose']

IMPORT_DIR, N_NALU_LAYERS, N_NALU_NEURONS, N_CLASSES, TEST_SIZE, \
    RANDOM_STATE, N_EPOCHS, LEARNING_RATE, BATCH_SIZE, DO_PP, DO_PCA = establish_inputs_from_filename(args)

if check_test_r2:
    # Check R2-score with testing data from original data set
    label_n_error_filename = 'pmap_raw_labels_and_errors.csv'
    
    print("[INFO] Loading in raw labels and errors from {}".format(label_n_error_filename))
    labels_df = pd.read_csv(label_n_error_filename)
    
    labels = labels_df['Flux'].values[:,None]
    labels_err = labels_df['Flux_err'].values
    
    # Feature File Switch
    if DO_PP and DO_PCA:
        features_input_filename = 'pmap_full_pipe_transformed_16features.csv'
    elif DO_PP:
        features_input_filename = 'pmap_minmax_transformed_from_raw_16features.csv'
    elif DO_PCA:
        features_input_filename = 'pmap_pca_transformed_from_stdscaler_16features.csv'
    else:
        features_input_filename = 'pmap_raw_16features.csv'
    
    print("[INFO] Loading in pre-processed features from {}".format(features_input_filename))
    features_input = pd.read_csv(features_input_filename).drop(['idx'], axis=1).values
else:
    # Preprocess new data
    print("[INFO] Loading pipelines on disk via joblib.")
    
    # Feature File Switch
    if DO_PP and DO_PCA:
        transformer_input_filename = 'pmap_full_pipe_transformer_16features.joblib.save'
    elif DO_PP:
        transformer_input_filename = 'pmap_minmax_scaler_transformer_from_raw_16features.joblib.save'
    elif DO_PCA:
        transformer_input_filename = 'pmap_pca_transformer_from_stdscaler_16features.joblib.save'
    else:
        transformer_input_filename = False
    
    print("[INFO] Loading in pre-processed features from {}".format(transformer_input_filename))
    pipeline = joblib.load(transformer_input_filename) if transformer_input_filename else None
    
    features_from_file = pd.read_csv(data_set_fname)
    
    features_from_file.drop(['xerr', 'yerr', 'flux', 'fluxerr', 'sigma_bg_flux'], axis=1, inplace=True)
    features_from_file = features_from_file.values
    # colnames = ['xpos', 'xerr', 'ypos', 'yerr', 'flux', 'fluxerr', 'np', 'xfwhm',\
    #        'yfwhm', 'bmjd', 'bg_flux', 'sigma_bg_flux', 'pix1', 'pix2', 'pix3',\
    #        'pix4', 'pix5', 'pix6', 'pix7', 'pix8', 'pix9']
    
    features_input = pipeline.transform(features_from_file) if transformer_input_filename else features_from_file

N_FEATURES = features_input.shape[-1]

print()
print('[INFO] Completed loading and processing of data, with: ')
print('\t{:15}: {}'.format('IMPORT_DIR', IMPORT_DIR))
print('\t{:15}: {}'.format('N_NALU_LAYERS', N_NALU_LAYERS))
print('\t{:15}: {}'.format('N_NALU_NEURONS', N_NALU_NEURONS))
print('\t{:15}: {}'.format('N_CLASSES', N_CLASSES))
print('\t{:15}: {}'.format('TEST_SIZE', TEST_SIZE))
print('\t{:15}: {}'.format('RANDOM_STATE', RANDOM_STATE))
print('\t{:15}: {}'.format('N_EPOCHS', N_EPOCHS))
print('\t{:15}: {}'.format('LEARNING_RATE', LEARNING_RATE))
print('\t{:15}: {}'.format('BATCH_SIZE', BATCH_SIZE))
print('\t{:15}: {}'.format('DO_PP', DO_PP))
print('\t{:15}: {}'.format('DO_PCA', DO_PCA))
print()

if check_test_r2:
    print('\t{:15}: {}'.format('Test Data', features_input_filename)
else:
    print('\t{:15}: {}'.format('Transformer', transformer_input_filename))
    print('\t{:15}: {}'.format('Data Filename', data_set_fname))

''' NALU: Nearual Arithmentic Logical Unit
        
        NALU uses memory and logic gates to train a unique TF layer to modify the gradients of the weights.
        This seems to be very smilar to a LSTM layer, but for a non-RNN.
        This code has been specifically implemented with tensorflow.
        
        Code source: https://github.com/grananqvist/NALU-tf
        Original paper: https://arxiv.org/abs/1808.00508 (Trask et al.)
'''
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

if __name__ == "__main__":
    if check_test_r2:
        ''' Testing the network success'''
        idx_train, idx_test = train_test_split(np.arange(labels.size), test_size=0.75, random_state=42)
        X_data, Y_data = features_input[idx_test], labels[idx_test]#[:,None]
    else:
        ''' Predicting a future solution'''
        X_data = features_input
    
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
        
        if check_test_r2: print("\n\nR2_Test: {}\n\n".format(r2_score(Y_data,ys_pred)))
        
        if save_data: 
            print('Saving to {}'.format(save_filename))
            pd.DataFrame(ys_pred, index=range(ys_pred.size)).to_csv(save_filename)

