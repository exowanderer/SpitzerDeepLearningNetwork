# Spitzer TSO Deep Learning Network

Tensorflow deep learning network for Spitzer time series observations trained on calibration data

Spitzer Time Series Observations (TSO) have been riddled with complex mutlidimensional correlations over time.  These correlations strongly interfere with exoplanet analysis and have inhibited the significant reproducibility for exoplanet observations (Ingalls et al 2017). 

Several important exoplanet research papers have been published that developed non-linear anlaysis methods to mitigate these correlated signals (Ballard et al 2010; Stevenson et al 2010; Fraine et al 2013; Lewis et al. 2013; Deming et al. 2015; Evens et al 2015; Morello et al. 2016; and Krick et al 2016).  For a full discussion of comparing these methods side-by-side, see Ingalls et al. 2016.

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

**Our next steps are to choose DLN parameters, such as:**
1) Dropout ratio (currently 50%)
2) Optimizer (currently Adagrad)
3) number of hidden layers (currently 2)
4) number of units per layer (current 10 & 10)

**Following this, we will start to turn off features and compare results**

Our results are compared using testing accuracy as well R-squared values.

Using only the Kaggle (Sumit Kuthari) notebook as is, we were able to achieve R-square = 99.7% with the test (unused) values.
