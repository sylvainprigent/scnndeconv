"""Misc functions to calculate shapes"""

import numpy as np


def my_disk(center, radius):
    """Calculate the coordinates of points inside a disk

    Parameters
    ----------
    center: tuple
        (x, y) coordinates of the center
    radius: int
        Radius of the disk (in pixels)

    Return
    ------
    a tuple of the (xx, yy) points coordinates

    """
    sr = radius * radius
    cc = []
    rr = []
    for x in range(center[0] - radius, center[0] + radius + 1):
        for y in range(center[1] - radius, center[1] + radius + 1):
            if (x - center[0]) * (x - center[0]) + (y - center[1]) * (
                    y - center[1]) <= sr:
                rr.append(x)
                cc.append(y)
    return np.array(cc), np.array(rr)


def my_disk_ring(center, radius, alpha):
    """Calculate the coordinates of points inside a disk ring

    Parameters
    ----------
    center: tuple
        (x, y) coordinates of the center
    radius: int
        Radius of the disk (in pixels)
    alpha: int
        Width of the ring in pixels

    Return
    ------
    a tuple of the (xx, yy) points coordinates

    """

    sr = radius * radius
    sa = (radius + alpha) * (radius + alpha)
    ro = radius + alpha + 1
    cc = []
    rr = []
    for x in range(center[0] - ro, center[0] + ro + 1):
        for y in range(center[1] - ro, center[1] + ro + 1):
            value = (x - center[0]) * (x - center[0]) + (y - center[1]) * (
                        y - center[1])
            if sr < value <= sa:
                rr.append(x)
                cc.append(y)
    return np.array(cc), np.array(rr)


def disk_patch(radius):
    """Create a patch with a disk shape where intensities sum to one

    Parameters
    ----------
    radius: int
        Radius of the disk (in pixels)

    """
    inner_patch = np.zeros((2 * radius + 1, 2 * radius + 1))
    irr, icc = my_disk((radius, radius), radius)
    inner_patch[irr, icc] = 1
    inner_patch /= np.sum(inner_patch)
    return inner_patch


def ring_patch(radius, alpha):
    """Create a patch with a ring shape where intensities sum to one

    Parameters
    ----------
    radius: int
        Radius of the disk (in pixels)
    alpha: int
        Width of the ring in pixels

    """
    outer_patch = np.zeros((2 * (radius + alpha) + 1, 2 * (radius + alpha) + 1))
    orr, occ = my_disk_ring((radius + alpha, radius + alpha), radius, alpha)
    outer_patch[orr, occ] = 1
    outer_patch /= np.sum(outer_patch)
    return outer_patch
