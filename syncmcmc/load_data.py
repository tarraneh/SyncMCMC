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
