#!/usr/bin/env python                                                          
#                        
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from model import *
sns.set_style("white")



# Load data files

load_file = np.loadtxt('results/spectrum1_results_granotsari')
data_array = load_file[load_file[:,0].argsort()]
load_file_spec2 = np.loadtxt('results/spectrum2_results_granotsari')
data_array2 = load_file_spec2[load_file_spec2[:,0].argsort()]
load_file_spec3 = np.loadtxt('results/spectrum3_results_granotsari')
data_array3 = load_file_spec3[load_file_spec3[:,0].argsort()]


# Parse data files

days = data_array[:,0]
spec1_flux = data_array[:,1]
spec1_va = data_array[:,4]
spec1_vm = data_array[:,7]
spec1_flux_plus = data_array[:,2]
spec1_va_plus = data_array[:,5]
spec1_vm_plus = data_array[:,8]
spec1_va_minus = data_array[:,6]
spec1_vm_minus = data_array[:,9]
spec1_flux_minus = data_array[:,3]


spec2_flux = data_array2[:,1]
spec2_va = data_array2[:,4]
spec2_vm = data_array2[:,7]


spec3_flux = data_array3[:,1]
spec3_va = data_array3[:,4]
spec3_vm = data_array3[:,7]


# Plot va and vm vs time for a given model

plt.subplot(2,1,1)
plt.plot(days,spec1_va+spec1_va_plus,marker='h',label='Upper Bound',color='k',lw='0.7')
plt.plot(days,spec1_va-spec1_va_minus,label='Lower Bound',marker='o',color='k',lw='0.7')
plt.plot(days,spec1_va,label='Estimate',color='#1b9e77',lw='0.7',marker='*')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Frequency [Hz]')
plt.title(r'$\nu_a$')
plt.legend()


plt.subplot(2,1,2)
plt.plot(days,spec1_vm+spec1_vm_plus,marker='h',label='Upper Bound',color='k',lw='0.7')
plt.plot(days,spec1_vm-spec1_vm_minus,label='Lower Bound',marker='o',color='k',lw='0.7')
plt.plot(days,spec1_vm,label='Estimate',color='#1b9e77',lw='0.7',marker='*')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Time [Days]')
plt.ylabel('Frequency [Hz]')
plt.title(r'$\nu_m$')
plt.legend()
plt.show()


# Plot flux vs time for a given model

plt.plot(days,spec1_flux_plus+spec1_flux,label='Upper Bound',marker='p',color='k',lw='0.7')
plt.plot(days,spec1_flux-spec1_flux_minus,label='Lower Bound',marker='d',color='k',lw='0.7')
plt.plot(days,spec1_flux,label='Estimate',color='#1b9e77',lw='0.7',marker='*')
plt.xlabel('Time [Days]')
plt.ylabel('Flux [mJy]')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
