from multiprocessing import set_start_method, cpu_count
set_start_method('forkserver')

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

ap.add_argument('-id', '--import_dir', type=str, required=True, help='The tensorflow ckpt save file')
ap.add_argument('-ed', '--export_dir', type=str, required=False, default='nalu_tf_save_dir/saves_warm_start_{}'.format(time_now), help='The tensorflow ckpt save file')
ap.add_argument('-nnl', '--n_nalu_layers', type=int, required=False, default=1, help='Whether to use 1 (default), 2, or ... N NALU layers.')
ap.add_argument('-nnn', '--n_nalu_neurons', type=int, required=False, default=0, help='How many features on the second NALU layer')
ap.add_argument('-ne', '--n_epochs', type=int, required=False, default=200, help='Number of N_EPOCHS to train the network with.')
ap.add_argument('-nc', '--n_classes', type=int, required=False, default=1, help='n_classes == 1 for Regression (default); > 1 for Classification.')
ap.add_argument('-bs', '--batch_size', type=int, required=False, default=32, help='Batch size: number of samples per batch.')
ap.add_argument('-lr', '--learning_rate', type=float, required=False, default=1e-3, help='Learning rate: how fast the optimizer moves up/down the gradient.')
ap.add_argument('-ts', '--test_size', type=float, required=False, default=0.75, help='How much to split the train / test ratio')
ap.add_argument('-rs', '--random_state', type=int, required=False, default=42, help='Integer value to initialize train/test splitting randomization')
ap.add_argument('-pp', '--pre_process', type=str2bool, nargs='?', required=False, default=True, help='Toggle whether to MinMax-preprocess the features.')
ap.add_argument('-pca', '--pca_transform', type=str2bool, nargs='?', required=False, default=True, help='Toggle whether to PCA-pretransform the features.')
ap.add_argument('-v', '--verbose', type=str2bool, nargs='?', required=False, default=False, help='Whether to set verbosity = True or False (default).')
ap.add_argument('-ds', '--data_set', type=str, required=False, default='', help='The csv file containing the data with which to train.')

try:
    args = vars(ap.parse_args())
except:
    args = {}
    args['import_dir'] = ''; print('\n\n**FINDME: THIS WILL BREAK LATER**\n\n')
    args['export_dir'] = ap.get_default('export_dir')
    args['n_nalu_layers'] = ap.get_default('n_nalu_layers')
    args['n_nalu_neurons'] = ap.get_default('n_nalu_neurons')
    args['n_epochs'] = ap.get_default('n_epochs')
    args['n_classes'] = ap.get_default('n_classes')
    args['batch_size'] = ap.get_default('batch_size')
    args['learning_rate'] = ap.get_default('learning_rate')
    args['test_size'] = ap.get_default('test_size')
    args['random_state'] = ap.get_default('random_state')
    args['pre_process'] =  ap.get_default('pre_process')
    args['pca_transform'] =  ap.get_default('pca_transform')
    args['verbose'] = ap.get_default('verbose')
    args['data_set'] = ap.get_default('data_set')

do_pp = args['pre_process']
do_pca = args['pca_transform']

verbose = args['verbose']
data_set_fname = args['data_set']

'''
print("loading pipelines on disk vis joblib.")
full_pipe = joblib.load('pmap_full_pipe_transformer_16features.joblib.save')
std_scaler_from_raw = joblib.load('pmap_standard_scaler_transformer_16features.joblib.save')
pca_transformer_from_std_scaled = joblib.load('pmap_pca_transformer_from_stdscaler_16features.joblib.save')
minmax_scaler_transformer_raw = joblib.load('pmap_minmax_scaler_transformer_from_raw_16features.joblib.save')
minmax_scaler_transformer_pca = joblib.load('pmap_minmax_scaler_transformer_from_pca_16features.joblib.save')
'''

label_n_error_filename = 'pmap_raw_labels_and_errors.csv'
print("Loading in raw labels and errors from {}".format(label_n_error_filename))
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

print("Loading in pre-processed features from {}".format(features_input_filename))
features_input = pd.read_csv(feature_input_filename).drop(['idx'], axis=1).values

print('BEGIN NEW HyperParameter Optimization.')

from sklearn.metrics import r2_score
from sklearn.utils import shuffle

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

if __name__ == "__main__":
    N_FEATURES = features_input.shape[-1]
    IMPORT_DIR = args['import_dir']
    EXPORT_DIR = args['export_dir']
    N_NALU_LAYERS = args['n_nalu_layers']
    N_NALU_NEURONS = args['n_nalu_neurons'] if args['n_nalu_neurons'] > 0 else N_FEATURES
    N_CLASSES = args['n_classes'] # = 1 for regression
    TEST_SIZE = args['test_size']
    RANDOM_STATE = args['random_state']
    
    N_EPOCHS = args['n_epochs']
    LEARNING_RATE = args['learning_rate']
    BATCH_SIZE = args['batch_size']
    
    WS_TIME = IMPORT_DIR.split('/')[1].split('_')[1]
    
    EXPORT_DIR = EXPORT_DIR + '_WS{}_nnl{}_nnn{}_nc{}_bs{}_lr{}_ne{}_ts{}_rs{}_PP{}_PCA{}/'.format(WS_TIME, N_NALU_LAYERS, N_NALU_NEURONS, N_CLASSES, 
                                                                                                    BATCH_SIZE, LEARNING_RATE, N_EPOCHS, TEST_SIZE, RANDOM_STATE, 
                                                                                                    {True:1, False:0}[DO_PP], {True:1, False:0}[DO_PCA])
    
    print("Saving models to path: {}".format(EXPORT_DIR))
    
    idx_train, idx_test = train_test_split(np.arange(labels.size), test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    X_data, Y_data = features_input[idx_train], labels[idx_train][:,None]
    
    LAST_BIT = X_data.shape[0]-BATCH_SIZE*(X_data.shape[0]//BATCH_SIZE)
    
    # Force integer number of batches total by dropping last "<BATCH_SIEZ" number of samples
    X_data_use = X_data[:-LAST_BIT].copy()
    Y_data_use = Y_data[:-LAST_BIT].copy()
    
    output_dict = {}
    output_dict['loss'] = np.zeros(N_EPOCHS)
    output_dict['accuracy'] = np.zeros(N_EPOCHS)
    output_dict['R2_train'] = np.zeros(N_EPOCHS)
    output_dict['R2_test'] = np.zeros(N_EPOCHS)
    output_dict['chisq_train'] = np.zeros(N_EPOCHS)
    output_dict['chisq_test'] = np.zeros(N_EPOCHS)
    
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
            
            # reshuffle the indices every epoch to avoid 
            X_data_use, Y_data_use = shuffle(X_data_use, Y_data_use)
            
            # for k in range(N_EPOCHS):
            #     batch_now = range(k*N_EPOCHS, (k+1)*N_EPOCHS)
            while i < len(X_data_use):
                xs, ys = X_data_use[i:i+BATCH_SIZE], Y_data_use[i:i+BATCH_SIZE]
                
                _, ys_pred, l = sess.run([train_op, Y_pred, loss], 
                        feed_dict={X: xs, Y_true: ys})
                
                # calculate number of correct predictions from batch
                gts += np.sum(np.isclose(ys, ys_pred, atol=1e-4, rtol=1e-4)) 
                
                i += BATCH_SIZE
            
            ytest_pred = Y_pred.eval(feed_dict={X: features_input[idx_test]})
            
            test_r2 = r2_score(labels[idx_test][:,None], ytest_pred)
            # print("Test R2 Score: {}".format(test_r2_score))
            
            acc = gts/len(Y_data_use)
            train_r2 = r2_score(ys, ys_pred)
            
            print('epoch {}, loss: {:.5}, accuracy: {:.5}, Batch R2: {:.5}, Test R2: {:.5}'.format( ep, l, acc, train_r2, test_r2))
            """
            output_dict['loss'][ep] = l
            output_dict['accuracy'][ep] = acc
            output_dict['R2_train'][ep] = train_r2
            output_dict['R2_test'][ep] = test_r2
            output_dict['chisq_train'][ep] = chisq(ys.flatten(), ys_pred.flatten(), spitzerCalRawData['fluxerr'][i:i+BATCH_SIZE])
            output_dict['chisq_test'][ep] = chisq(labels[idx_test], ytest_pred.flatten(), spitzerCalRawData['fluxerr'][idx_test])
            """
            file_name = 'model_epoch{}_l{:.5}_a{:.5}_BatchR2-{:.5}_TestR2-{:.5}.ckpt'.format(ep, l, acc, train_r2, test_r2))
            
            save_path = saver.save(sess, EXPORT_DIR + file_name)
        
        ep = '_FINAL'
        file_name = 'model_epoch{}_l{:.5}_a{:.5}_BatchR2-{:.5}_TestR2-{:.5}.ckpt'.format(ep, l, acc, train_r2, test_r2))
        
        save_path = saver.save(sess, EXPORT_DIR + file_name)
        
        print("Model saved in path: {}".format(save_path))
        """
        try:
            pd.DataFrame(output_dict, index=range(N_EPOCHS)).to_csv(EXPORT_DIR+ "model_loss_acc_BatchR2_TestR2_DataFrame.csv")
        except Exception as e:
            print('DataFrame to CSV broke because', str(e))
        """

