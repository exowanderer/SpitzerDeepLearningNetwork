from batman    import TransitModel, TransitParams
from exoparams import PlanetParams
from functools import partial
from lmfit     import Model, Parameters
from pandas    import read_csv
from pylab     import *;ion()

def bin_array(arr, uncs = None,  binsize=100, KeepTheChange = False):
    '''
        Given nSize = size(time), nCols = binsize, nRows = nSize / nCols

            this function reshapes the 1D array into a new 2D array of dimension
                nCols x nRows or binsize x (nSize / nCols)

            after which, the function collapses the array into a new 1D array by taking the median

        Because most input arrays cannot be subdivided into an even number of subarrays of size `binsize`
            we actually first slice the array into a 1D array of size `nRows*binsize`.

            The mean of the remaining elements from the input array is then taken as the final element
                in the output array

    '''
    nSize   = arr.size
    nCols   = int(nSize / binsize)
    nRows   = binsize

    EqSize  = nRows*nCols
    
    useArr  = arr[:EqSize].copy()   # this array can be subdivided evenly
    
    if uncs is not None:
        # weighted mean profile
        useUncs = uncs[:EqSize].copy()   # this array can be subdivided evenly
        binArr  = median((useArr / useUncs).reshape(nCols, nRows).mean(axis=1)/ useUncs.reshape(nCols, nRows))
        stdArr  = median((useArr / useUncs).reshape(nCols, nRows).std(axis=1) / useUncs.reshape(nCols, nRows))
        
        if KeepTheChange:
            SpareArr    = arr[EqSize:].copy()
            SpareUncs   = uncs[EqSize:].copy()

            binTC       = median((SpareArr / SpareUncs)) / median(SpareUncs.reshape(nCols, nRows))
            stdTC       = median((SpareArr / SpareUncs)) / median(SpareUncs.reshape(nCols, nRows))

            binArr  = concatenate((binArr, [binTC]))
            stdArr  = concatenate((stdArr, [stdTC]))
    else:
        # standard mean profile
        binArr  = mean(useArr.reshape(nCols, nRows),axis=1)
        stdArr  = std(useArr.reshape(nCols, nRows),axis=1) / sqrt(nSize)
        
        if KeepTheChange:
            SpareArr    = arr[EqSize:].copy()
            binTC       = median(SpareArr)
            stdTC       = std(SpareArr)

            binArr  = concatenate((binArr, [binTC]))
            stdArr  = concatenate((stdArr, [stdTC]))

    return binArr, stdArr

def batman_lmfit_model(period, tCenter, bImpact, rsap, edepth, tdepth, ecc, omega, times, 
                        u1=None, u2=None, ldtype = 'uniform', transittype="primary"):#, bm_params=None):
    
    if tdepth is not 0.0 or edepth is not 0.0:
        # if bm_params is None:
        #     bm_params = batman.TransitParams() # object to store transit parameters
        
        bm_params = batman.TransitParams() # object to store transit parameters
        
        aprs  = 1 / rsap
        inc   = b2inc(bImpact, aprs, ecc, omega)*180/pi
        
        bm_params.per       = period  # orbital period
        bm_params.t0        = tCenter # time of inferior conjunction
        # bm_params.bImpact   = bImpact # b, impact parameter
        bm_params.r_a       = rsap  # RsAp
        bm_params.a         = aprs    # semi-major axis (in units of stellar radii)
        bm_params.fp        = edepth  # f
        # bm_params.tdepth    = tdepth  # from Fraine et al. 2014s
        bm_params.rp        = sqrt(tdepth) # planet radius (in units of stellar radii)
        bm_params.ecc       = ecc     # eccentricity
        bm_params.w         = omega   # longitude of periastron (in degrees)
        bm_params.inc       = inc # orbital inclination (in degrees)
        bm_params.limb_dark = ldtype  # limb darkening model
        bm_params.u         = []      # limb darkening coefficients
        
        if u1 is not None and ldtype is not 'uniform':
            bm_params.u.append(u1)
        elif u1 is not None and ldtype is 'uniform':
            raise ValueError('If you set `u1`, you must also set `ldtype` to either linear or quadratic')
        if u2 is not None and ldtype is 'quadratic':
            bm_params.u.append(u2)
        elif u2 is not None and ldtype is not 'quadratic':
            raise ValueError('If you set `u2`, you must also set `ldtype` quadratic')
        
        bm_params.delta_phase = deltaphase_eclipse(bm_params.ecc, bm_params.w)
        bm_params.t_secondary = bm_params.t0 + bm_params.per*bm_params.delta_phase
        
        m_eclipse = batman.TransitModel(bm_params, times, transittype=transittype).light_curve(bm_params)
    else:
        return ones(times.size)
    
    return m_eclipse

xo3b_rf       = read_csv('XO3_r46468096_STD_save.txt') if len(argv) < 2 else read_csv(argv[1])
planet_params = PlanetParams('XO-3 b')

iPeriod   = planet_params.per.value
iTCenter  = planet_params.tt.value-2400000.5
iBImpact  = planet_params.b.value
iRsAp     = 1.0/planet_params.ar.value
iEdepth   = 400/1e6 # blind guess
iTdepth   = planet_params.depth.value
iEcc      = planet_params.ecc.value*0.0 # this is wrong, but we can work with it
iOmega    = planet_params.om.value*pi/180

initialParams_eclipse = Parameters()

initialParams_eclipse.add('tCenter' , iTCenter, True)# , 0.0 , 1.0)
initialParams_eclipse.add('edepth'  , iEdepth , True)# , 0.0 , 1.0)

partial_eclipse    = partial(batman_lmfit_model, period      = iPeriod, 
                                                 bImpact     = iBImpact, 
                                                 rsap        = iRsAp, 
                                                 tdepth      = iTdepth, 
                                                 ecc         = iEcc, 
                                                 omega       = iOmega, 
                                                 times       = xo3b_rf['bmjd'].values, 
                                                 transittype = "secondary")

lcEclipse = Model(partial_eclipse)

stime1 = time()
fitResults_Eclipse = lcEclipse.fit(data, params=initialParams_eclipse, method='powell')

print('Operation took {} seconds'.format(time() - stime1))

flux_bin, flux_err = bin_array(fitResults_Eclipse.data)
plot(xo3b_rf['bmjd'].values, fitResults_Eclipse.data)