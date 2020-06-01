import numpy as np
from scipy.spatial import ConvexHull

def rubberband(wavenumbers, spectrum):
    """
    A rubberband baseline is essentially the lower half of tyhe convex hull 
    of the data. See this discussion for more details and code.

    https://dsp.stackexchange.com/questions/2725/how-to-perform-a-rubberband-correction-on-spectroscopic-data
    """

    # First find the convex hull using scipy.spatial tools
    tmp = np.vstack([ wavenumbers.flatten(),spectrum.flatten()]).transpose()
    v = ConvexHull(tmp).vertices
    
    # rool it untill the first point is the first point measured
    v = np.roll(v, -v.argmin())
    # We don't care aboput the top half of the convex hull
    v = v[:v.argmax()]

    # Use interpolation to get the baseline across the whole 
    # spectrum
    return np.interp(wavenumbers, wavenumbers[v], spectrum[v])
    





