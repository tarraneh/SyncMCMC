#!/usr/bin/env python                                                          
#                        
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from model import *
sns.set_style("white")



# Values from Zauderer et al and Berger et al

F_v = np.array([10**1.47,10**0.96,10**0.98,10**0.97,10**1.11,10**1.21,10**1.39,10**1.52,10**1.58,10**1.51,10**1.56,10**1.63,10**1.99,10**1.92,10**1.72,10**1.64,10**1.60])
va = np.array([10**11.01,10**10.11,10**10.02,10**9.96,10**9.95,10**9.96,10**9.99,10**9.95,10**9.97,10**9.82,10**9.83,10**9.90,10**10.04,10**9.96,10**9.71,10**9.62,10**9.58])
vm = np.array([10**11.74,10**11.41,10**11.21,10**10.99,10**10.78,10**10.62,10**10.53,10**10.39,10**10.26,10**10.13,10**10.04,10**9.99,10**9.67,10**9.54,10**9.38,10**9.28,10**9.13])


load_file = np.loadtxt('results/spectrum1_results_granotsari')
data_array = load_file[load_file[:,0].argsort()]
load_file_spec2 = np.loadtxt('results/spectrum2_results_granotsari')
data_array2 = load_file_spec2[load_file_spec2[:,0].argsort()]
load_file_spec3 = np.loadtxt('results/spectrum3_results_granotsari')
data_array3 = load_file_spec3[load_file_spec3[:,0].argsort()]
load_file_spec4 = np.loadtxt('results/spectrum4_results_granotsari')
data_array4 = load_file_spec4[load_file_spec4[:,0].argsort()]



days = data_array[:,0]
spec1_flux = data_array[:,1]
spec1_va = data_array[:,2]
spec1_vm = data_array[:,3]

spec2_flux = data_array2[:,1]
spec2_va = data_array2[:,2]
spec2_vm = data_array2[:,3]

spec3_flux = data_array3[:,1]
spec3_va = data_array3[:,2]
spec3_vm = data_array3[:,3]

spec4_flux = data_array4[:,1]
spec4_va = data_array4[:,2]
spec4_vm = data_array4[:,3]

# Compare to all models

#plt.plot(days,spec1_flux,label='Spectrum 1',color='#1b9e77',lw='0.7',marker='o')
#plt.plot(days,spec2_flux,label='Spectrum 2',color='gray',lw='0.7',marker='h',ms=7)
#plt.plot(days,F_v,label='Paper Values',color='k',lw='0.7',marker='D')
#plt.plot(days,spec3_flux,marker='*',label='Weighted Spectrum (F1=F2)',color='#d95f02',lw='0.7',ms=10)
#plt.plot(days,spec4_flux,marker='s',label='Weighted Spectrum',color='#7570b3',lw='0.7')
#plt.axvline(x=225,color='k',ls='--',lw='0.7')
#plt.legend(loc='lower left')
#plt.xlabel('Time [days]')
#plt.ylabel('Flux [mJy]')
#plt.xscale('log')
#plt.yscale('log')
#plt.show()
#
#plt.plot(days,spec1_va,label='Spectrum 1',color='#1b9e77',marker='o',lw='0.7')
#plt.plot(days,spec2_va,label='Spectrum 2',color='gray',lw='0.7',marker='h',ms=7)
#plt.plot(days,va,label='Paper Values',color='k',lw='0.7',marker='D')
#plt.plot(days,spec3_va,marker='*',label='Weighted Spectrum (F1=F2)',color='#d95f02',ms=10,lw='0.7')
#plt.plot(days,spec4_va,marker='s',label='Weighted Spectrum',color='#7570b3',lw='0.7')
#plt.axvline(x=225,color='k',ls='--',lw='0.7')
#plt.legend(loc= 'lower left')
#plt.xlabel('Time [days]')
#plt.ylabel('va [Hz]')
#plt.xscale('log')
#plt.yscale('log')
#plt.show()
#
#plt.plot(days,spec1_vm,label='Spectrum 1',color='#1b9e77',marker='o',lw='0.7')
#plt.plot(days,spec2_vm,label='Spectrum 2',color='gray',lw='0.7',marker='h',ms=7)
#plt.plot(days,vm,label='Paper Values',color='k',lw='0.7',marker='D')
#plt.plot(days,spec3_vm,marker='*',label='Weighted Spectrum (F1=F2)',color='#d95f02',ms=10,lw='0.7')
#plt.plot(days,spec4_vm,marker='s',label='Weighted Spectrum',color='#7570b3',lw='0.7')
#plt.axvline(x=225,color='k',ls='--',lw='0.7')
#plt.legend(loc = 'lower left')
#plt.xlabel('Time [days]')
#plt.ylabel('vm [Hz]')
#plt.xscale('log')
#plt.yscale('log')
#plt.show()
#


# Compare to weighted spectra only

plt.plot(days,spec3_flux,marker='*',label='Weighted Spectrum (F1=F2)',color='#d95f02',lw='0.7',ms=10)
plt.plot(days,F_v,label='Paper Values',color='#1b9e77',lw='0.7',marker='o')
plt.plot(days,spec4_flux,marker='s',label='Weighted Spectrum',color='#7570b3',lw='0.7')
plt.axvline(x=225,color='k',ls='--',lw='0.7')
plt.xlabel('Time [days]')
plt.ylabel('Flux [mJy]')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='lower left')
plt.show()


plt.plot(days,spec3_va,marker='*',label='Weighted Spectrum (F1=F2)',color='#d95f02',lw='0.7',ms=10)
plt.plot(days,va,label='Paper Values',color='#1b9e77',lw='0.7',marker='o')
plt.plot(days,spec4_va,marker='s',label='Weighted Spectrum',color='#7570b3',lw='0.7')
plt.axvline(x=225,color='k',ls='--',lw='0.7')
plt.xlabel('Time [days]')
plt.ylabel('va [Hz]')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='lower left')
plt.show()


plt.plot(days,spec3_vm,marker='*',label='Weighted Spectrum (F1=F2)',color='#d95f02',lw='0.7',ms=10)
plt.plot(days,vm,label='Paper Values',color='#1b9e77',lw='0.7',marker='o')
plt.plot(days,spec4_vm,marker='s',label='Weighted Spectrum',color='#7570b3',lw='0.7')
plt.axvline(x=225,color='k',ls='--',lw='0.7')
plt.xlabel('Time [days]')
plt.ylabel('vm [Hz]')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='lower left')
plt.show()

v_range = np.linspace(1E9,1E14,1E3)

eta = spec1_va[11]/spec1_vm[11]
gamma = 2.2
t = 216.
d = 5.544E-1
z = 0.3534
frac_a = 1.
frac_v = 1.


# Find peak frequency and flux

mod = spectrum(v_range,spec1_flux[11],spec1_va[11],spec1_vm[11])
F_p = np.max(mod)
v_p = v_range[mod.argmax()]

print F_p,v_p

# Calculate equipartition radius in cm

Req = (7.5E17) * F_p**(2./3.) * d**(4./3.) * (v_p/1.E10)**(-17./12.) * eta**(35./36.) * (1.+z)**(-5./3.) * t**(-5./12.) * frac_a**(-7./12.) * frac_v**(-1./12.)

# Calculate minimum total energy in ergs

Eeq = (5.7E47) * F_p**(2./3.) * d**(4./3.) * (v_p/1.E10)**(1./12.) * eta**(5./36.) * (1.+z)**(-5./3.) * t**(13./12.) * frac_a**(-1./12.) * frac_v**(5./12.)

# Calculate bulk lorentz factor

lorentz_factor = 12. * F_p**(1./3.) * d**(2./3.) * (v_p/1.E10)**(-17./24.) * eta**(35./72.) * (1.+z)**(-1./3.) * t**(-17./24.) * frac_a**(-7./24.) * frac_v**(-1./24.)

print Req
print Eeq
print lorentz_factor

Req_arr = np.array([6.61084596776e+13,3.7058522393e+14,1.37424686326e+15,2.37096238136e+15,1.12570713224e+15,1.04404946928e+15,1.1266476643e+15,1.92669930886e+15,6.496990707e+15,2.70165965742e+15,5.9348631389e+15,1.91212738689e+15])
Eeq_arr = np.array([6.85119705191e+48,1.03585742287e+49,1.64582390191e+49,2.18723006694e+49,3.82763217272e+49,7.07889187921e+49,1.23073877327e+50,1.47487391918e+50,1.87267549186e+50,1.90621442287e+50,2.37588574181e+50,2.45144195485e+50
])
lorentz_arr = np.array([0.0586147905412,0.0981313446748,0.154294500941,0.167345751703,0.0901415785016,0.072935531287,0.0656150258732,0.0718430570554,0.115753611716,0.066033768259,0.0884781312927,0.0479617737177])


plt.plot(days[0:12],Req_arr,marker='o')
plt.ylabel('r [cm]')
plt.xlabel('Time [days]')
plt.xscale('log')
plt.yscale('log')
plt.show()
#
plt.plot(days[0:12],Eeq_arr,marker='o')
plt.ylabel('E [ergs]')
plt.xlabel('Time [days]')
plt.xscale('log')
plt.yscale('log')
plt.show()
#
#plt.plot(days[0:12],lorentz_arr,marker='o')
#plt.ylabel('LF')
#plt.xlabel('Time [days]')
#plt.xscale('log')
#plt.yscale('log')
#plt.show()#