Compute Image Moments
=====================

Image moments describe the shape of a region. They're useful for computing
centroids, orientation, and shape descriptors.

All moments
-----------

Get raw, central, and normalized central moments in one call::

    m = mask.moments()

    # Raw moments (area, center of mass, spread)
    area = m['m00']
    cx = m['m10'] / m['m00']  # centroid x
    cy = m['m01'] / m['m00']  # centroid y

    # Central moments (translation-invariant)
    mu20 = m['mu20']  # variance in x
    mu02 = m['mu02']  # variance in y
    mu11 = m['mu11']  # covariance

    # Normalized central moments (scale-invariant)
    nu20 = m['nu20']
    nu11 = m['nu11']

The dictionary has the same keys as ``cv2.moments()``.

Hu moment invariants
--------------------

For shape matching across translation, scale, and rotation :footcite:`hu1962visual`::

    hu = mask.hu_moments()  # Returns array of 7 values

Hu moments are derived from normalized central moments and are invariant
under translation, scale, and rotation. The first 6 are also invariant
under reflection; the 7th changes sign under reflection.

Compare two shapes::

    hu1 = mask1.hu_moments()
    hu2 = mask2.hu_moments()

    # Log-transform for numerical stability (like cv2.matchShapes)
    hu1_log = np.sign(hu1) * np.log10(np.abs(hu1) + 1e-10)
    hu2_log = np.sign(hu2) * np.log10(np.abs(hu2) + 1e-10)

    distance = np.sum(np.abs(hu1_log - hu2_log))

Orientation from moments
------------------------

Compute the orientation (angle of the major axis) of a region::

    m = mask.moments()

    # Angle of principal axis in radians
    theta = 0.5 * np.arctan2(2 * m['mu11'], m['mu20'] - m['mu02'])

Eccentricity
------------

Measure how elongated a shape is::

    m = mask.moments()

    # Eigenvalues of the covariance matrix
    a = m['mu20'] / m['m00']
    b = m['mu11'] / m['m00']
    c = m['mu02'] / m['m00']

    lambda1 = 0.5 * (a + c + np.sqrt((a - c)**2 + 4*b**2))
    lambda2 = 0.5 * (a + c - np.sqrt((a - c)**2 + 4*b**2))

    eccentricity = np.sqrt(1 - lambda2 / lambda1)

.. footbibliography::
