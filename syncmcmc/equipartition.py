import numpy as np








# Calculate equipartition radius in cm

Req = (7.5E17) * F_p**(2./3.) * d**(4./3.) * (v_p/1.E10)**(-17./12.) * eta**(35./36.) * (1.+z)**(-5./3.) * t**(-5./12.) * frac_a**(-7./12.) * frac_v**(-1./12.)

# Calculate minimum total energy in ergs

Eeq = (5.7E47) * F_p**(2./3.) * d**(4./3.) * (v_p/1.E10)**(1./12.) * eta**(5./36.) * (1.+z)**(-5./3.) * t**(13./12.) * frac_a**(-1./12.) * frac_v**(5./12.)

# Calculate bulk lorentz factor

lorentz_factor = 12. * F_p**(1./3.) * d**(2./3.) * (v_p/1.E10)**(-17./24.) * eta**(35./72.) * (1.+z)**(-1./3.) * t**(-17./24.) * frac_a**(-7./24.) * frac_v**(-1./24.)
