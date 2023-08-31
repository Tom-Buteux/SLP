import pandas as pd
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def makePolygons(quads,data):
    polygons = []
    coords_list = []
    for quad in quads:
        # if the data source contains x/y columns, use those
        if 'size' in data.columns:
            coords = data.loc[quad, ['x', 'y']].values
        elif 'RA' in data.columns:
            coords = data.loc[quad, ['RA', 'DE']].values
        else:
            raise Exception('Data source does not contain x/y or RA/DE columns')

        coords = order_points(coords)
        coords_list.append(coords)
        poly = Polygon(coords, closed=True)
        polygons.append(poly)
    p = PatchCollection(polygons, edgecolor='k', linewidth=1, alpha=0.2, linestyle='--')

        # Generate colors
    colors = 100*np.random.rand(len(polygons))

    # Set facecolors
    p.set_array(np.array(colors))
    return p, coords_list

def order_points(pts):
    """
    This function orders a set of 2D points in a counterclockwise direction with respect to their centroid.
    
    Parameters:
    pts (List[Tuple[float, float]]): A list of tuples where each tuple contains the x and y coordinates of a point.
    
    Returns:
    numpy.ndarray: A 2D numpy array of shape (N, 2), where N is the number of input points. The points are sorted
    counterclockwise with respect to their centroid.
    
    Note:
    This function uses the arctan2 function from numpy to compute the angle of each point with respect to the
    centroid. The points are then sorted by these angles using numpy's argsort function.
    """
    # convert the points to a numpy array
    pts = np.array(pts)
    # compute the centroid of the points
    centroid = np.mean(pts, axis=0)
    # sort the points by their angle with respect to the centroid
    angles = np.arctan2(pts[:,1] - centroid[1], pts[:,0] - centroid[0])
    pts = pts[np.argsort(angles)]
    return pts


def rotate_point(x, y, angle_rad, cx, cy):
    """Rotate point (x, y) around center (cx, cy) by angle_rad radians"""
    x_new = cx + np.cos(angle_rad) * (x - cx) - np.sin(angle_rad) * (y - cy)
    y_new = cy + np.sin(angle_rad) * (x - cx) + np.cos(angle_rad) * (y - cy)
    return x_new, y_new


