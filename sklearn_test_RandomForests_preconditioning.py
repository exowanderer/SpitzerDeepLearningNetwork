import pandas as pd
import numpy as np
# import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)

import pdb

import warnings
warnings.filterwarnings("ignore")

import gc

from matplotlib import pyplot as plt
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import StandardScaler, MinMaxScaler, minmax_scale
from sklearn.ensemble         import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.decomposition    import PCA, FastICA
from sklearn.externals        import joblib
from sklearn.metrics          import r2_score

plt.rcParams['figure.dpi'] = 300

from corner import corner

from sklearn.metrics import r2_score

from time import time
start0 = time()

from sklearn import feature_selection, feature_extraction, decomposition

# ## Load CSVs data
spitzerDataRaw  = pd.read_csv('pmap_ch2_0p1s_x4_rmulti_s3_7.csv')

PLDpixels = pd.DataFrame({key:spitzerDataRaw[key] for key in spitzerDataRaw.columns.values if 'pix' in key})

PLDnorm = np.sum(np.array(PLDpixels),axis=1)

PLDpixels = (PLDpixels.T / PLDnorm).T

spitzerData = spitzerDataRaw.copy()
for key in spitzerDataRaw.columns: 
    if key in PLDpixels.columns:
        spitzerData[key] = PLDpixels[key]

testPLD = np.array(pd.DataFrame({key:spitzerData[key] for key in spitzerData.columns.values if 'pix' in key}))
assert(not sum(abs(testPLD - np.array(PLDpixels))).all())
print('Confirmed that PLD Pixels have been Normalized to Spec')

pixKey = [key for key in spitzerData.columns if 'pix' in key]

notFeatures     = ['flux', 'fluxerr', 'dn_peak']#, 'yerr', 'xerr', 'xycov']
feature_columns = spitzerData.drop(notFeatures,axis=1).columns.values
features        = spitzerData.drop(notFeatures,axis=1).values
labels          = spitzerData['flux'].values

stdScaler     = StandardScaler()
# minMaxScaler  = MinMaxScaler()

# features_MMscaled   = minMaxScaler.fit_transform(features)
# labels_MMscaled     = minMaxScaler.fit_transform(labels[:,None]).ravel()
features_SSscaled   = stdScaler.fit_transform(features)
labels_SSscaled     = stdScaler.fit_transform(labels[:,None]).ravel()

# **PCA Pretrained Random Forest Approach**
print('Performing PCA', end=" ")
pca = PCA()
start = time()
pca_feature_set = pca.fit_transform(features_SSscaled)
print('took {} seconds'.format(time() - start))

# **ICA Pretrained Random Forest Approach**
print('Performing ICA', end=" ")
ica = FastICA()
start = time()
ica_feature_set = ica.fit_transform(features_SSscaled)
print('took {} seconds'.format(time() - start))

nTrees = 1000

# **Standard Random Forest Approach**
# for nComps in range(1,spitzerData.shape[1]):
randForest_STD = RandomForestRegressor(n_estimators=nTrees, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',\
                                     max_leaf_nodes=None, bootstrap=True, oob_score=True, n_jobs=-1, random_state=42, verbose=0, warm_start=True)

start=time()
randForest_STD.fit(features_SSscaled, labels_SSscaled)

randForest_STD = joblib.load('randForest_standard_approach.save', mmap_mode='r+')

randForest_STD_oob = randForest_STD.oob_score_
randForest_STD_pred= randForest_STD.predict(features_SSscaled)
randForest_STD_Rsq = r2_score(labels_SSscaled, randForest_STD_pred)

print('Standard Random Forest:\n\tOOB Score: {:.3f}%\n\tR^2 score: {:.3f}%\n\tRuntime:   {:.3f} seconds'.format(randForest_STD_oob*100, randForest_STD_Rsq*100, time()-start))

joblib.dump(randForest_STD, 'randForest_standard_approach.save')
# del randForest_STD, randForest_STD_pred
# _ = gc.collect()

# for nComps in range(1,spitzerData.shape[1]):
start=time()
randForest_PCA = RandomForestRegressor(n_estimators=nTrees, criterion='mse', max_depth=None, min_samples_split=2,min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', \
                                        max_leaf_nodes=None, bootstrap=True, oob_score=True, n_jobs=-1, random_state=42, verbose=0, warm_start=True)

randForest_PCA.fit(pca_feature_set, labels_SSscaled)

randForest_PCA_oob = randForest_PCA.oob_score_
randForest_PCA_pred= randForest_PCA.predict(features_SSscaled)
randForest_PCA_Rsq = r2_score(labels_SSscaled, randForest_PCA_pred)

print('PCA Pretrained Random Forest:\n\tOOB Score: {:.3f}%\n\tR^2 score: {:.3f}%\n\tRuntime:   {:.3f} seconds'.format(randForest_PCA_oob*100, randForest_PCA_Rsq*100, time()-start))

joblib.dump(randForest_PCA, 'randForest_PCA_approach.save')

# del randForest_PCA, randForest_PCA_pred
# _ = gc.collect()

# for nComps in range(1,spitzerData.shape[1]):
randForest_ICA = RandomForestRegressor(n_estimators=nTrees, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', \
                                        max_leaf_nodes=None, bootstrap=True, oob_score=True, n_jobs=-1, random_state=42, verbose=0, warm_start=True)

start=time()
randForest_ICA.fit(ica_feature_set, labels_SSscaled)

randForest_ICA_oob = randForest_ICA.oob_score_
randForest_ICA_pred= randForest_ICA.predict(features_SSscaled)
randForest_ICA_Rsq = r2_score(labels_SSscaled, randForest_ICA_pred)

print('ICA Pretrained Random Forest:\n\tOOB Score: {:.3f}%\n\tR^2 score: {:.3f}%\n\tRuntime:   {:.3f} seconds'.format(randForest_ICA_oob*100, randForest_ICA_Rsq*100, time()-start))

joblib.dump(randForest_ICA, 'randForest_ICA_approach.save')
# del randForest_ICA
# _ = gc.collect()

# **Importance Sampling**

# randForest_STD = joblib.load('/Volumes/My Passport for Mac/TychoBkup/CloudResearch/SpitzerCal_RandForestSaves/randForest_standard_approach.save', mmap_mode='r+')

importances = randForest_STD.feature_importances_
indices     = np.argsort(importances)[::-1]
std         = np.std([tree.feature_importances_ for tree in randForest_STD.estimators_],axis=0)

cumsum = np.cumsum(importances[indices])
nImportantSamples = np.argmax(cumsum >= 0.95) + 1
nImportantSamples

# del randForest_STD
# _ = gc.collect()

# **Random Forest Pretrained Random Forest Approach**
rfi_feature_set = features_SSscaled.T[indices][:nImportantSamples].T

# for nComps in range(1,spitzerData.shape[1]):
randForest_RFI = RandomForestRegressor(n_estimators=nTrees, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', \
                                        max_leaf_nodes=None, bootstrap=True, oob_score=True, n_jobs=-1, random_state=42, verbose=0, warm_start=True)

start=time()
randForest_RFI.fit(rfi_feature_set, labels_SSscaled)

randForest_RFI_oob = randForest_RFI.oob_score_
randForest_RFI_pred= randForest_RFI.predict(features_SSscaled)
randForest_RFI_Rsq = r2_score(labels_SSscaled, randForest_RFI_pred)

print('RFI Pretrained Random Forest:\n\tOOB Score: {:.3f}%\n\tR^2 score: {:.3f}%\n\tRuntime:   {:.3f} seconds'.format(randForest_RFI_oob*100, randForest_RFI_Rsq*100, time()-start))

joblib.dump(randForest_RFI, 'randForest_RFI_approach.save')
# del randForest_RFI
# _ = gc.collect()
pdb.set_trace()
