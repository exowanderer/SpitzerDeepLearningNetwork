from argprase  import ArgumentParser
from batman    import TransitModel, TransitParams
from exoparams import PlanetParams
from functools import partial
from lmfit     import Model, Parameters
from pandas    import read_csv
from pylab     import *;ion()
from time      import time

from scipy.signal import medfilt
from statsmodels.robust import scale

def clipOutliers(array, nBins=101, nSig=10):
        if not (nBins % 2):
            print('nBins must be odd; `clipOutliers` is adding 1')
            nBins += 1
        
        array           = array.copy()
        medfilt_array   = medfilt(array, nBins)
        mad_array       = scale.mad(array)
        outliers        = abs(array - medfilt_array) > nSig * mad_array
        array[outliers] = medfilt_array[outliers]
        
        return array

def deltaphase_eclipse(ecc, omega):
    return 0.5*( 1 + (4. / pi) * ecc * cos(omega))

def bin_array(arr, uncs = None,  nbins=100, KeepTheChange = False):
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
    nCols   = nbins#int(nSize / binsize)
    nRows   = int(nSize / nbins)
    
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

def batman_lmfit_model(times, period, tCenter, inc, aprs, edepth, tdepth, ecc, omega, crvtur=0.0, slope=0.0, intcpt=0.0,
                        u1=None, u2=None, ldtype = 'uniform', transittype="primary"):#, bm_params=None):
    
    # print(period, tCenter, inc, aprs, edepth, tdepth, ecc, omega)
    if tdepth is not 0.0 or edepth is not 0.0:
        # if bm_params is None:
        #     bm_params = TransitParams() # object to store transit parameters
        
        bm_params = TransitParams() # object to store transit parameters
        
        # aprs  = 1 / rsap
        # inc   = b2inc(bImpact, aprs, ecc, omega)*180/pi
        
        bm_params.per       = period  # orbital period
        bm_params.t0        = tCenter # time of inferior conjunction
        # bm_params.bImpact   = bImpact # b, impact parameter
        # bm_params.r_a       = rsap  # RsAp
        bm_params.a         = aprs    # semi-major axis (in units of stellar radii)
        bm_params.fp        = edepth  # f
        # bm_params.tdepth    = tdepth  # from Fraine et al. 2014s
        bm_params.rp        = sqrt(abs(tdepth)) # planet radius (in units of stellar radii)
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
        
        m_eclipse = TransitModel(bm_params, times, transittype=transittype).light_curve(bm_params)
        
        if transittype == 'secondary':
            m_eclipse = m_eclipse - edepth
        
        line_model = intcpt
        if slope != 0.0:
            line_model = line_model + slope*(times-times.mean())
        if crvtur != 0.0:
            line_model = line_model + crvtur*(times-times.mean())**2.
        
        m_eclipse = m_eclipse * line_model
    else:
        return slope*(times-median(times)) + intcpt
    
    return m_eclipse

ap  = ArgumentParser()
ap.add_argument('-f', '--filename', require=True, default='', help='Filename with Spitzer transit observation data.')
# xo3b_filename = 'XO3_r46468096_STD_save.txt'
xo3b_filename = 'XO3_r46468352_STD_save.txt' if len(argv) < 2 else argv[1]
xo3b_rf       = read_csv(xo3b_filename)
planet_params = PlanetParams('XO-3 b')

iPeriod   = planet_params.per.value
iTCenter  = planet_params.tt.value-2400000.0 # -2400000.5
iInc      = planet_params.i.value
iApRs     = planet_params.ar.value
iEdepth   = 1520/1e6 # From Ingalls et al 2016
iTdepth   = planet_params.depth.value
iEcc      = planet_params.ecc.value#*0.0 # this is wrong, but we can work with it
iOmega    = planet_params.om.value*pi/180

rf_data_norm     = xo3b_rf['RF_Predict'].values / median(xo3b_rf['RF_Predict'].values)
flux_data_norm   = xo3b_rf['flux'].values       / median(xo3b_rf['flux'].values)
phots_clean_data0= flux_data_norm / rf_data_norm

phots_clean_data = clipOutliers(phots_clean_data0)

xo3b_times = xo3b_rf['bmjd'].values

flux_bin, flux_err_bin = bin_array(phots_clean_data, nbins=200)
time_bin, _            = bin_array(xo3b_times, nbins=200)

gTCenter = iTCenter + 0.033110936259618029

iCrvtur, iSlope, iIntcpt = 0.0, 0.0, 1.0
init_model  = batman_lmfit_model(xo3b_times, iPeriod, gTCenter, iInc, iApRs, iEdepth, iEdepth, iEcc, iOmega, iCrvtur, iSlope, iIntcpt, transittype='secondary')

initialParams_eclipse = Parameters()

initialParams_eclipse.add_many( ('period'  , iPeriod , False, 0, inf),
                                ('tCenter' , gTCenter, True , gTCenter - 0.005, gTCenter + 0.005),
                                ('inc'     , iInc    , True , 0, 90.), 
                                ('aprs'    , iApRs   , True , 0, inf),
                                ('edepth'  , iEdepth , True , 0, 1),#, iEdepth-500/1e6,iEdepth+500/1e6),
                                ('tdepth'  , iTdepth , False, 0, 1),#, iEdepth-500/1e6,iEdepth+500/1e6),
                                ('ecc'     , iEcc    , False, 0, 1),
                                ('omega'   , iOmega  , False, 0, 360),
                                ('crvtur'  , 0.0     , False),
                                ('slope'   , 0.0     , True),
                                ('intcpt'  , 1.0     , True))

lcEclipse = Model(batman_lmfit_model, independent_vars=['times', 'transittype'])

stime1 = time()
fitResults_Eclipse = lcEclipse.fit( phots_clean_data, 
                                    params      = initialParams_eclipse, 
                                    method      = 'powell',
                                    times       = xo3b_times, 
                                    transittype = "secondary")

print('Operation took {} seconds'.format(time() - stime1))

efit_period, efit_tCenter, efit_inc, efit_aprs, efit_edepth, efit_tdepth, efit_ecc, efit_omega, efit_crvtur, efit_slope, efit_intcpt = fitResults_Eclipse.best_values.values()

gTCenter = iTCenter + 0.5*iPeriod + 0.033110936259618029

initialParams_transit = Parameters()

initialParams_transit.add_many( ('period'  , iPeriod , False, 0., inf),
                                ('tCenter' , gTCenter, True , gTCenter - 0.005, gTCenter + 0.005),
                                ('inc'     , iInc    , True , 0., 90.),
                                ('aprs'    , iApRs   , True , 0.,inf),
                                ('edepth'  , iEdepth , False, 0., 1.),
                                ('tdepth'  , iEdepth , True , 0., 1.),#, iEdepth-500/1e6,iEdepth+500/1e6),
                                ('ecc'     , iEcc    , False, 0., 1.),
                                ('omega'   , iOmega  , False, 0., 360.),
                                ('crvtur'  , 0.0     , False),
                                ('slope'   , 0.0     , True),
                                ('intcpt'  , 1.0     , True))

lcTransit = Model(batman_lmfit_model, independent_vars=['times', 'transittype'])

stime1 = time()
fitResults_Transit = lcTransit.fit( phots_clean_data, 
                                    params      = initialParams_transit, 
                                    method      = 'powell',
                                    times       = xo3b_times, 
                                    transittype = "primary")

print('Operation took {} seconds'.format(time() - stime1))
tfit_period, tfit_tCenter, tfit_inc, tfit_aprs, tfit_edepth, tfit_tdepth, tfit_ecc, tfit_omega, tfit_crvtur, tfit_slope, tfit_intcpt = fitResults_Transit.best_values.values()

efit_model = batman_lmfit_model(xo3b_times, efit_period, efit_tCenter, efit_inc, efit_aprs, efit_edepth, efit_tdepth, efit_ecc, efit_omega, efit_crvtur, efit_slope, efit_intcpt, transittype='secondary')
tfit_model = batman_lmfit_model(xo3b_times, tfit_period, tfit_tCenter, tfit_inc, tfit_aprs, tfit_edepth, tfit_tdepth, tfit_ecc, tfit_omega, tfit_crvtur, tfit_slope, tfit_intcpt, transittype='primary')

errorbar(time_bin, flux_bin, flux_err_bin, fmt='.', label='data')
# plot(xo3b_times, init_model, label='Initial Model', lw=3)
plot(xo3b_times, efit_model, label='Fit Eclipse Model: {} ppm'.format(int(efit_edepth*1e6)), lw=3)
# plot(xo3b_times, tfit_model, label='Fit Transit Model: {} ppm'.format(int(tfit_tdepth*1e6)), lw=3)

aorName = xo3b_filename.split('/')[1].split('_')[1]
title('XO-3 b - ' + aorName)

legend(loc=0)

fig_savename = 'XO3_Data/XO3_{}_-_BinnedData_and_Bestfit_Model_{}ppm.png'.format(aorName, int(efit_edepth*1e6))
savefig(fig_savename)

print('Saving figure as {}'.format(fig_savename))