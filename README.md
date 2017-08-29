# Spitzer TSO Deep Learning Network

Tensorflow deep learning network for Spitzer time series observations trained on calibration data

Spitzer Time Series Observations (TSO) have been riddled with complex mutlidimensional correlations over time.  These correlations strongly interfere with exoplanet analysis and have inhibited the significant reproducibility for exoplanet observations (Ingalls et al 2017). 

Several important exoplanet research papers have been published that developed non-linear anlaysis methods to mitigate these correlated signals (Ballard et al 2010; Stevenson et al 2010; Fraine et al 2013; Lewis et al. 2013; Deming et al. 2015; Evans et al 2015; Morello et al. 2016; and Krick et al 2016).  For a full discussion of comparing these methods side-by-side, see Ingalls et al. 2016.

The end result of these comparative analysis is that machine learning is a robust, but complicated method for interpreting and mitigating correlated noise signals.  Ingalls+2016 particularly focused on reproducibility (real world data) and accuracy (synthetic data). This approach revealed that wavelet+ICA (independent component analysis) provided a dramatic improvement over other methods. Notebly, Gaussian Processes did not perfrom siginficanlty better than other methods, and failed at most reproducibility metrics.

Inspired by this work, the Spitzer Analysis Temporal Community (SATCom) has combined expertise and begun to investigate rigorous machine learning models to best incorporate substantive expertise and machine learning algorithms.

Our approach is to combine all calibration data from years of Spitzer calibraiton observations.  The data has been curated by Jim Ingalls and Jessica Krick of IPAC at Caltech. For access to the raw data files, please contact them.

**Here we present the analysis of the Spitzer calibration data with Tensorflow Deep Learning Networks (DLNs)**

We first start with an example from Kaggle (https://www.kaggle.com/usersumit/tensorflow-dnnregressor/notebook) that developed an example for using the canned estimator DNNRegrssor from tf.contrib.learn.

We adapted the notebook from Sumit Kuthari to include our Spitzer calibration observations. 

**The raw data includes:**

1) X & Y gaussian center positions values
2) X & Y gaussian center positions uncertainty
3) X & Y gaussian center positions covariance
4) Flux over time
5) Flux uncertainty over time
6) Noise Pixels (i.e. Beta-Pixels; see Lewis et al. 2013)
7) X & Y full-width half-max values
8) Data Number Peaks
9) Barycentric Modified Juliean Date (time array)
10) t_cernox (?)
11) Background Flux (annular?)
12) Background Flux Uncertainty (annular?)
13) 9 pixel values over time (for PLD analysis; central/max pixel + 8 bordering pixels)

**Our DLN analysis takes in 18 of these 24 features:**

1) 9 pixel values
2) X & Y gaussian center positions
3) X & Y FWHM values
4) Noise pixels
5) Barycentric Modified Julian Date 
6) Background flux over time
7) Data Number Peaks
8) t_cernox

**Our DLN analysis has one output feature set:**
1) Flux ove time: assuming that the calibrator star does not vary on timescales relevant to this analysis, a resulting flat residuals would be the most optimal solution for processing existing and future Spitzer TSO observations.

**Our Preprocessing steps**
1) Jessica Krick ran a 4-sigma outlier rejection algorithm that removed all samples that deviate from a (purposefully flawed) 4-sigma threshold.  The remaining data is Normal to within reason.
2) We normalize the 9 pixel vectors: The 9 pixel values are stored in raw electrons per sec; but need to be normalized to remove any astrophysical features that can be modeled after the systematics are mitigated.
3) Scikit-learn MinMaxScaler: DLNs nominally expect that all feature values exist between 0 and 1; so we used the MinMaxScalar to set all values from their intrinsic levels to span 0-1.  Because J.Krick performed a robust outlier rejection -- and the values obey reasonable Normality -- these vectors should not suffer from poor scaling issues from outliers.
4) The flux vector ('output') was also outlier rejectiong and MinMax scaled, which is nominally expected from most DLNs algorithms.

**Our next steps are to choose DLN parameters, such as:**
1) Dropout ratio (currently 50%)
2) Optimizer (currently Adagrad)
3) number of hidden layers (currently 2)
4) number of units per layer (current 10 & 10)
5) preprocessing / scaling steps; notably, we could try StandardScaler instead of MinMaxScaler

**Following this, we will start to turn off features and compare results**

Our results are compared using testing accuracy as well R-squared values.


**So far So good**

Using only the Kaggle (Sumit Kuthari) notebook as is, we were able to achieve R-square = 99.7% with the test (unused) values. Moreover, it seems that the testing residuals from our DLN using ReLU are symmetric, while those from Tanh and Sigmoid tend to under estimate or over estimate -- only a small percentage -- respectively.

We are trying all of the above methods to minimize overfitting and optimize feature selection. 

Notably after introducing dropout=0.5 (50% dropped neurons) -- used to minimize overfitting -- the test R-squared changed dramatically: 

**Dropout = 50%; AdaGrad; 50k iterations**

| Activation  | Validation  | R-Squared |
| ----------- | ----------- | --------- |
| **Function**    | **Accuracy**    |           |
| ReLU        |    99.7%    |   -42%    |
| Sigmoid     |    99.7%    |  -271%    |
| Tanh        |    99.5%    |   +85%    |

At the same time, the validation/training loss improved significantly (a sign of limited overfitting).


**Compring Dropout with No Dropouts; AdaGrad; 50k iterations**

| Activation  | Val_acc / Train_acc   | Val_acc / Train_acc    |
| ----------- | --------------------- | ---------------------- |
| **Function**    | **Ratio without dropout** | **Ratio with 50% dropout** |
| ReLU        |           4.8         |         0.743          |
| Sigmoid     |          42.2         |         0.941          |
| Tanh        |           3.9         |         0.159          |

**Validation accuracy with 50% dropouts**

Moreover, the validation accuracy is more believable with Dropout=0.5: 

| Activation  | Validation Accuracy   | Validation Accuracy    |
| ----------- | --------------------- | ---------------------- |
| **Function**    | **Ratio without dropout** | **Ratio with 50% dropout** |
| ReLU        |        99.997%        |         99.44%         |
| Sigmoid     |        99.927%        |         98.58%         |
| Tanh        |        99.993%        |         99.96%         |

**Using AdaM with 100k and 50% Dropouts**

Because it's possible that the training sessions had not converged, we tested new chains using an AdaM optimizer for 100k iterations. The results did impove on all accounts, except a marginal decrease in the R-squared test for Tanh.

| Activation      | Validation   | R-Squared | Val_acc / Train_acc        |
| -----------     | -----------  | --------- | -------------------------- |
| **Function**    | **Accuracy** |           | **Ratio with 50% dropout** |
| ReLU            |    99.58%    |   30.82%  |           0.761            |
| Sigmoid         |    99.87%    |   34.38%  |           0.357            |
| Tanh            |    99.96%    |   80.20%  |           0.182            |

- The top two R-Squared values are significantly improved, which implies that the 50k chains with AdaGrad indeed had not converged.  
- The third R-squared value is slightly lower than previously, but this could also be a lack of convergence with 50k iterations of AdaGrad (previous chain). 
- Lastly, the Sigmoid function -- not usually considered a contender -- was significantly imporve with more iterations and a faster optimzier (AdaM)

---

We tested the major three DLN activation functions (relu, sigmoid, tanh) with a 2 and 3 layers (butterfly architecture: 10-10 & 10-5-10). We used dropout=0.5 for regularization and with excessive iterations (100k) to ensure convergence. We then compared four metrics for 'success'

1) Validation accuracy: by training on 40% of the data and evaluating the MSE with a separate 40% of the data, we consisistently found that the validation accuracy was >99%.
2) A metric that we made up is the ratio of the validation loss to the training loss; a number close to one would imply a lack of over-fitting
3) The R-squared test is another way to measure over-fitting, but also provides a metric for the balance of the training set.
4) The symmetric of the residuals: by plotting the test and predicted on the same figure and the residuals on another, we can "see" if the the predictions match the samples. So far, it seems that ReLU produces the most symmetric residuals; but not always the best validation accuracy or R-sqaured values.


---
**Feature Optimization Dimensionality Reduction**

The next step is feature optimization.  This could include dimensionality reduction (i.e. PCA) or feature selection (i.e. feature importance).  Note that the features have already been outlier rejected (cleaned) and MinMax scaled (0 to 1).

Now we've expanded the previous DLN approach to include both Random Forests and PCA to pre-select or transform, respectively, the data into a more linearly separable hyperspace.

1) Using PCA, and assuming that we want to capture at least 95% of the variance, we have shown that only 9 CPA compoenents are necessary to inject in the DLN.  That reduces the feature space from 21 to 9.

2) Using Random Forests, we have also found that 9 features are necessary to capture at least 95% of the feature importance.

These metrics are not connected; the matching number of features should not be interpreted as though the PCA is identifying the same features as the Random Forest. But both methods do agree that only 40% of the features are necessary to capture the vast majority of the information in the feature set.

---

**Ensemble Deep Learning Idea**

Build 100+ DLNs with 10k iterations (that's short) over bootstrap with replacement samplings. Each one build a random combination of 2 - 5 layers, with 2 - nFeatures units per layer. Symmetry will be enforced; but this is an assumption.

First: Randomize samples with replacement to generate bootstrapped dataset with out of bag error handling.

Second: Randomly choose an activation function: sigmoid, tanh, relu, conv1d + max pooling - defines the genus of the butterflies

Third: Randomly design the network architecture:

1) Randomly choose nLayers {2 to 5}
2) Randomly choose nUnits0 in innermost layer {2 to (nFeatures - nLayers)}
   - steps 1 and 2 define the butterfly's species
3) (a) If nLayers is even: set nUnits_layer1 = nUnits_layer0
   (b) If nLayers is odd:  Randomly choose nUnits_layer1 from {nUnits_layer0 to nFeatures}
4) Set nUnits_layer_n1 = nUnits_layer1 (= enforces butterfly structure)
5) Repeat steps 4 & 5 until nLayers is reached

Fourth: weakly train the networks. Run for 10k iterations w/ Adam optimizer and no dropouts (regularization done by ensemble).

Fifth: Ensemble prediction = weighted average of individual predictions.

Justification: By combining multiple, weak, small networks into a network of networks, the algorithm will act like a random forest of deep learning networks.  Given the enforced symmetry into a butterfly structure, I'll call this a "Flock of Butterflies" method of FoB -- or Farfalle.

**How Many DLNs will that take**

- 10 for each activation function
- 10 for each set of layers (2,3,4,5)

10 * (10 + 10 + 10 + 10) = 400 DLNs

With ~20 minutes to train 10k iterations per DLN, will take ~144 hours or ~6 days. 

**Genetic adaption**

1) Train two of every species of butterfly (8 butterflies), make an ensemble prediction, keep the butterfly (per species) whose individual predictions are closest to the ensemble prediction (use maximum entropy, or MSE).
2) Randomly generate a new butterfly per species with small modifications (+/- 1 unit per layer; +/- one layer)
   -- set the new layers (low probability) and units (medium probability) with random samples from nearby units mean & std
3) Retrain these 8 butterflies
4) Repeat steps 2 & 3 until "convergence".
   - either choose to iterate for nGenerations
   - or choose to iterate until "enough" butterflies have been re-trained for nMaxIterations (~100k; or 10 generations)

Using a genetic style algorithm, we will train a medium sized set of butterflies (small, weakly trained, symmetric DLNs), 'keep' the butterfly closest to the ensemble. It is assumed that the butterfly closest to the ensemble weight average is a more fit individual per species at reproducing the goal -- a stronger ensemble.

The result is expected to be a set of 4 (or 8) strongly trained, small symmetric DLNs that float around the evolved ensemble prediction. The 4 evolved butterflies have been equivalently trained for nMaxIterations.
