"""
    In the context of track before detect lets try to see if we can reproduce Al plots in the tkbd scenario
    This python implementation is the translation of Al's code
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# first load the radarcrosssection object
RCS=pd.read_csv('RadarCrossSection.csv')
print(RCS.head(), RCS.keys())

RCS = RCS.drop(columns=['Source']) # removed sources since it is not interesting

# rename the columns for a better explanability
RCS = RCS.rename(columns={'Lower RCS (m^2)':'lower',
                          'Upper RCS (m^2)':'upper',
                          'Mid (m^2)':'mid'})

def not_log_db(snr_db):
    return 10.**(snr_db/10.)

# loaded the visibility objects
r_det_miss = 1000 # detection for a missile
sigma_miss = 0.5 # radar cross section for a missile
SNR_det_db = 13 # standard SNR detection in DB
#SNR_det = 10**(SNR_det_db/10.)  # not in log scale
SNR_det = not_log_db(SNR_det_db)

### RADAR equation ###
# r_det_miss**4 = k*sigma_miss/SNR_det
# k is the constant radara power, gain, noise profile - we can assume this is constant

k = SNR_det*r_det_miss**4/sigma_miss  # in whatever units

SNR_tkbd_db = 7 # assuming a 6 db detectability increase
SNR_tkdb = not_log_db(SNR_tkbd_db)


# define this mid
RCS['MIDDLE'] = np.exp(0.5*np.log10(RCS['lower']) +0.5*np.log10(RCS['upper']))

RCS['rDET'] = (k*RCS['MIDDLE']/SNR_det)**(0.25)  ## radar function
RCS['tkdb'] = (k*RCS['MIDDLE']/SNR_tkdb)**(0.25)

plt.plot(RCS['MIDDLE'], RCS['rDET'], color='red')
plt.plot(RCS['MIDDLE'], RCS['tkdb'], color='blue')
plt.xscale('log')
plt.yscale('log')
plt.show()
