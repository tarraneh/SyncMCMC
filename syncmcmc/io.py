from data_class import Fluxes

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



