import numpy as np
import re
F_mcmc = 2
va_mcmc = 3424
vm_mcmc = 987654


data_file = 'Sw1644+57_51days'
days = re.findall('\d+',data_file.split('_')[1])
print days[0]

F_values = [float(days[0]),F_mcmc,va_mcmc,vm_mcmc]
print F_values

with open("spectrum1_results","ab") as input_file:
    np.savetxt(input_file,F_values, fmt='%1.5f',newline=' ')
    input_file.write('\n')


