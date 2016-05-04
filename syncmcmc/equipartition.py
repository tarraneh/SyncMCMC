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


spec2_flux = data_array2[:,1]
spec2_va = data_array2[:,4]
spec2_vm = data_array2[:,7]


spec3_flux = data_array3[:,1]
spec3_va = data_array3[:,4]
spec3_vm = data_array3[:,7]





# Define known parameters

v_range = np.linspace(1E9,1E14,1E3)
gamma = 2.2
d = 5.544E-1
z = 0.3534
frac_a = 1.
frac_v = 1.



Req_spec1 = []
Eeq_spec1 = []
lorentz_spec1 = []

Req_spec2 = []
Eeq_spec2 = []
lorentz_spec2 = []

Req_spec3 = []
Eeq_spec3 = []
lorentz_spec3 = []




for (flux,va,vm,t) in zip(spec1_flux,spec1_va,spec1_vm,days):
   # Find the peak of model 1 spectrum
   mod = spectrum(v_range,flux,va,vm)
   F_p = np.max(mod)
   v_p = v_range[mod.argmax()]
   eta = va/vm


   # Calculate equipartition radius in cm
   
   Req = (7.5E17) * F_p**(2./3.) * d**(4./3.) * (v_p/1.E10)**(-17./12.) * eta**(35./36.) * (1.+z)**(-5./3.) * t**(-5./12.) * frac_a**(-7./12.) * frac_v**(-1./12.)
   
   # Calculate minimum total energy in ergs
   
   Eeq = (5.7E47) * F_p**(2./3.) * d**(4./3.) * (v_p/1.E10)**(1./12.) * eta**(5./36.) * (1.+z)**(-5./3.) * t**(13./12.) * frac_a**(-1./12.) * frac_v**(5./12.)
   
   # Calculate bulk lorentz factor
   
   lorentz_factor = 12. * F_p**(1./3.) * d**(2./3.) * (v_p/1.E10)**(-17./24.) * eta**(35./72.) * (1.+z)**(-1./3.) * t**(-17./24.) * frac_a**(-7./24.) * frac_v**(-1./24.)
   
   Req_spec1.append(Req)
   Eeq_spec1.append(Eeq)
   lorentz_spec1.append(lorentz_factor)


for (flux,va,vm,t) in zip(spec2_flux,spec2_va,spec2_vm,days):
   mod = spectrum_2(v_range,flux,va,vm)
   F_p = np.max(mod)
   v_p = v_range[mod.argmax()]
   eta = va/vm



   # Calculate equipartition radius in cm
   
   Req = (7.5E17) * F_p**(2./3.) * d**(4./3.) * (v_p/1.E10)**(-17./12.) * eta**(35./36.) * (1.+z)**(-5./3.) * t**(-5./12.) * frac_a**(-7./12.) * frac_v**(-1./12.)
   
   # Calculate minimum total energy in ergs
   
   Eeq = (5.7E47) * F_p**(2./3.) * d**(4./3.) * (v_p/1.E10)**(1./12.) * eta**(5./36.) * (1.+z)**(-5./3.) * t**(13./12.) * frac_a**(-1./12.) * frac_v**(5./12.)
   
   # Calculate bulk lorentz factor
   
   lorentz_factor = 12. * F_p**(1./3.) * d**(2./3.) * (v_p/1.E10)**(-17./24.) * eta**(35./72.) * (1.+z)**(-1./3.) * t**(-17./24.) * frac_a**(-7./24.) * frac_v**(-1./24.)
   
   Req_spec2.append(Req)
   Eeq_spec2.append(Eeq)
   lorentz_spec2.append(lorentz_factor)




for (flux,va,vm,t) in zip(spec3_flux,spec3_va,spec3_vm,days):
   mod = weighted_spectrum(v_range,flux,va,vm)
   F_p = np.max(mod)
   v_p = v_range[mod.argmax()]
   eta = va/vm



   # Calculate equipartition radius in cm
   
   Req = (7.5E17) * F_p**(2./3.) * d**(4./3.) * (v_p/1.E10)**(-17./12.) * eta**(35./36.) * (1.+z)**(-5./3.) * t**(-5./12.) * frac_a**(-7./12.) * frac_v**(-1./12.)
   
   # Calculate minimum total energy in ergs
   
   Eeq = (5.7E47) * F_p**(2./3.) * d**(4./3.) * (v_p/1.E10)**(1./12.) * eta**(5./36.) * (1.+z)**(-5./3.) * t**(13./12.) * frac_a**(-1./12.) * frac_v**(5./12.)
   
   # Calculate bulk lorentz factor
   
   lorentz_factor = 12. * F_p**(1./3.) * d**(2./3.) * (v_p/1.E10)**(-17./24.) * eta**(35./72.) * (1.+z)**(-1./3.) * t**(-17./24.) * frac_a**(-7./24.) * frac_v**(-1./24.)
   
   Req_spec3.append(Req)
   Eeq_spec3.append(Eeq)
   lorentz_spec3.append(lorentz_factor)




plt.plot(days,Req_spec3,marker='d',color='#7570b3',lw='0.7')
plt.ylabel('r [cm]')
plt.xlabel('Time [days]')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
#
plt.plot(days,Eeq_spec1,marker='o',color='#1b9e77',lw='0.7')
plt.ylabel('E [ergs]')
plt.xlabel('Time [days]')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
#
plt.plot(days,lorentz_spec3,marker='d',color='#7570b3',lw='0.7')
plt.ylabel('LF')
plt.xlabel('Time [days]')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()#