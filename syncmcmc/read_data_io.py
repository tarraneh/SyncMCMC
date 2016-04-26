import numpy as np


# Load data


def load_data(data_file):
    flux = []
    freqs = []
    error = []
    for line in open(data_file):
       lines = line.strip()
       if not line.startswith("#"):
          columns = line.split(',')
          freqs.append(columns[0])
          flux.append(columns[1])
          error.append(columns[2].rstrip('\n'))
    
    flux = np.array(flux).astype(float)
    freqs = np.array(freqs).astype(float)
    error = np.array(error).astype(float)
    return flux, freqs, error
    