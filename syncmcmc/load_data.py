class Fluxes:

	"""
	frequency: Frequencies of observations [Hz]
	flux: Flux densities [mJy]
	error: Error on flux values [mJy]
	"""

	def __init__(self,(frequency, flux, error)):
		
		self.frequency = frequency
		self.flux = flux
		self.error = error




	def readClassFile(filename,Fluxes, separator="\t", comment='#'):
		try:
			file = open(filename, "r")
		except:
			raise IOError, "Can't open file %s for reading" % filename

		for line in file:
			if comment is not None and line.startswith(comment):
				continue

		tuple = line.rstrip().split(separator)
		try:
			Fluxes(tuple)
			return Fluxes(tuple)
		except ValueError:
			raise IOError, "Line of file %s does not have the required format: \n%s" % (filename, line)


