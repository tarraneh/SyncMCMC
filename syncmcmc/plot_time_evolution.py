#!/usr/bin/env python                                                          
#                        
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("white")


load_file = open('mcmc_results','r')
lines = load_file.readlines()
days = []
for i in lines:
    days.append(i.split(' ')[0])
load_file.close()


days = []
spec1_flux = []
spec3_flux = []
spec4_flux = []
spec1_va = []
spec3_va = []
spec4_va = []
spec1_vm = []
spec3_vm = []
spec4_vm = []
edo_flux = []
edo_va = []
edo_vm = []
for line in open('mcmc_results'):
   lines = line.strip()
   if not line.startswith("#"):
      columns = line.split()
      days.append(columns[0])
      spec1_flux.append(columns[1])
      spec3_flux.append(columns[3])
      spec4_flux.append(columns[4])
      spec1_va.append(columns[5])
      spec3_va.append(columns[7])
      spec4_va.append(columns[8])
      spec1_vm.append(columns[9])
      spec3_vm.append(columns[11])
      spec4_vm.append(columns[12])
      #edo_flux.append(columns[13])
      #edo_va.append(columns[14])
      #edo_vm.append(columns[15])




plt.plot(days,spec1_flux,label='Spectrum 1',color='#1b9e77',lw='0.7',marker='o')
plt.plot(days,spec3_flux,marker='*',label='Weighted Spectrum (F1=F2)',color='#d95f02',lw='0.7',ms=10)
plt.plot(days,spec4_flux,marker='s',label='Weighted Spectrum',color='#7570b3',lw='0.7')
plt.legend(loc=2)
plt.xlabel('Days')
plt.ylabel('Flux [mJy]')
plt.xscale('log')
plt.yscale('log')
plt.show()

plt.plot(days,spec1_va,label='Spectrum 1',color='#1b9e77',marker='o',lw='0.7')
plt.plot(days,spec3_va,marker='*',label='Weighted Spectrum (F1=F2)',color='#d95f02',ms=10,lw='0.7')
plt.plot(days,spec4_va,marker='s',label='Weighted Spectrum',color='#7570b3',lw='0.7')
plt.legend(loc=2)
plt.xlabel('Days')
plt.ylabel('va [Hz]')
plt.xscale('log')
plt.yscale('log')
plt.show()

plt.plot(days,spec1_vm,label='Spectrum 1',color='#1b9e77',marker='o',lw='0.7')
plt.plot(days,spec3_vm,marker='*',label='Weighted Spectrum (F1=F2)',color='#d95f02',ms=10,lw='0.7')
plt.plot(days,spec4_vm,marker='s',label='Weighted Spectrum',color='#7570b3',lw='0.7')
plt.legend()
plt.xlabel('Days')
plt.ylabel('vm [Hz]')
plt.xscale('log')
plt.yscale('log')
plt.show()