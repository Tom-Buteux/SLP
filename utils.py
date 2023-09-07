import numpy as np
from astropy.wcs import WCS
import pandas as pd

def findABCD(coordinates):
    points = np.array(coordinates)
    dist = np.linalg.norm(points[:,None] - points[None,:], axis=2)


    # A and B are the points that are furthest apart
    i, j = np.unravel_index(np.argmax(dist, axis=None), dist.shape)

    A = points[i]
    B = points[j]

    # assign the other 2 points to C and D
    other_indices = [idx for idx in range(len(points)) if idx not in [i, j]]
    C = points[other_indices[0]]
    D = points[other_indices[1]]

    # if B is closer to the mean of C and D than A is, swap A and B
    if np.linalg.norm(B - np.mean([C, D], axis=0)) < np.linalg.norm(A - np.mean([C, D], axis=0)):
        A = points[j]
        B = points[i]

    # C is closer to A than D is
    if np.linalg.norm(C - A) > np.linalg.norm(D - A):
        C = points[other_indices[1]]
        D = points[other_indices[0]]



    
    return A, B, C, D, np.max(dist)




def sortABCD(coordinates):
    points = np.array(coordinates)
    num_points = len(points)
    distances = np.zeros((num_points, num_points))

    for i in range(num_points):
        for j in range(i+1, num_points):
            dist = np.linalg.norm(points[i] - points[j])
            distances[i, j] = dist


    i, j = np.unravel_index(np.argmax(distances, axis=None), distances.shape)
    A = points[i]
    B = points[j]

    other_indices = [idx for idx in range(num_points) if idx not in [i, j]]
    C = points[other_indices[0]]
    D = points[other_indices[1]]

    original_points = [A,B,C,D]


    # Translate all points so A moves to (0, 0)
    translated_points = [A,B,C,D] - A

    # if a is at position translated_points[1], swap a and b
    if np.linalg.norm(translated_points[1]) < np.linalg.norm(translated_points[0]):
        translated_points[[0, 1]] = translated_points[[1, 0]]

    # Calculate theta_B
    theta_B = np.arctan2(translated_points[1][1], translated_points[1][0])

    # Calculate the angle to rotate B onto the line x = y
    theta = np.pi/4 - theta_B

    # Create the rotation matrix
    R = np.array([[np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]])


    # Rotate all points
    rotated_points = np.dot(R, translated_points.T).T

    # Scale all points so B is at (1, 1)
    scaled_points = rotated_points / rotated_points[1]

    # if Cx + Dx > 1, subtract all x-coordinates from 1
    if scaled_points[2][0] + scaled_points[3][0] > 1:
        scaled_points = np.subtract(1,scaled_points)

    if np.linalg.norm(scaled_points[1]) < np.linalg.norm(scaled_points[0]):
        scaled_points[[0, 1]] = scaled_points[[1, 0]]

    # ensure that Cx < Dx
    if scaled_points[2][0] > scaled_points[3][0]:
        scaled_points[[2, 3]] = scaled_points[[3, 2]]

    
    return original_points, scaled_points, np.max(distances)


def findScale(points):
    num_points = len(points)
    max_distance = 0.0  # Initialize max_distance to zero
    
    for i in range(num_points):
        for j in range(i+1, num_points):
            distance = np.linalg.norm(points[i] - points[j])
            if distance > max_distance:
                max_distance = distance
                
    return max_distance




def hashcode(quad):
    """
    This function takes a set of four points, called a 'quad', as input, and generates a unique identifier, or 'hash code', 
    for this quad. This is done by transforming the coordinates of the quad in several ways (translation, rotation, scaling) 
    and then creating a flat list of coordinates of two points. 

    In detail, the transformation process includes the following steps:
    - Finding the pair of points in the quad that are furthest apart and naming these A and B.
    - Naming the remaining points C and D based on their distance from A.
    - Translating the points so that A is at the origin.
    - Rotating the points so that B lies on the line y=x.
    - Scaling the points so that B is at (1,1).
    - If the sum of the x-coordinates of C and D is larger than 1, subtracting each point's coordinates from 1.
    - If the x-coordinate of C is larger than that of D, swapping C and D.
    - Creating a hash code by concatenating the coordinates of C and D.

    Args:
        quad (list): A list of four 2D points, where each point is represented as a list of two numbers.

    Returns:
        tuple: A tuple of four numbers, which represent the transformed coordinates of points C and D in the order (Cx, Cy, Dx, Dy).
    
    Note:
        The function is optimized to minimize run time.
    """
    # convert the quad to a NumPy array
    quad = np.array(quad)

    _,points,_ = sortABCD(quad)
        

    
    hashcode = [points[2][0],points[2][1],points[3][0],points[3][1]]

    return hashcode[0],hashcode[1],hashcode[2],hashcode[3]


def RotateCoordinates(x, y, angle):
    """
    This function rotates a given 2D point (x, y) around the origin (0, 0) by a given angle.

    Parameters:
    x (float): The x-coordinate of the point.
    y (float): The y-coordinate of the point.
    angle (float): The angle in radians by which the point should be rotated counter-clockwise.

    Returns:
    List[float, float]: A list containing the x and y coordinates of the rotated point.
    """
    xRot = x * np.cos(angle) - y * np.sin(angle)
    yRot = x * np.sin(angle) + y * np.cos(angle)
    return [xRot, yRot]

def centroid(vertices):
    """
    Finds the centroid of a polygon given its vertices.
    Works with both 2D and 3D polygons.

    :param vertices: A list of vertices, where each vertex is a list or tuple with 2 or 3 elements
    :return: Centroid coordinates as a tuple
    """
    # Ensure that the vertices are in a numpy array
    vertices = np.array(vertices)

    # Check whether the vertices are in 2D or 3D
    if vertices.shape[1] not in [2, 3]:
        raise ValueError("Vertices must be in 2D or 3D")

    # If vertices are in 2D, make them 3D by adding a z-coordinate of 0
    if vertices.shape[1] == 2:
        vertices = np.hstack([vertices, np.zeros((vertices.shape[0], 1))])

    # Compute the centroid
    centroid_coords = vertices.mean(axis=0)
    return tuple(centroid_coords)

def find_orthogonal_set(vec):
    """Find two vectors that are orthogonal to the given vector `vec`."""
    if np.linalg.norm(vec) == 0:
        raise ValueError("The input vector must not be the zero vector.")
    
    # Normalize the input vector
    vec = vec / np.linalg.norm(vec)
    
    # Initialize candidates for orthogonal vectors
    candidates = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    
    # Make sure the candidates are not parallel to the input vector
    candidates = [c for c in candidates if np.abs(np.dot(vec, c)) < 1.0]
    
    # Choose one candidate and find a vector orthogonal to it and the input vector
    chosen = candidates[0]
    ortho1 = np.cross(vec, chosen)
    ortho1 = ortho1 / np.linalg.norm(ortho1)  # normalize
    
    # Find another vector that is orthogonal to both `vec` and `ortho1`
    ortho2 = np.cross(vec, ortho1)
    ortho2 = ortho2 / np.linalg.norm(ortho2)  # normalize
    
    return ortho1, ortho2

def quad2wcs(img_quad,img_data,cat_quad,cat_data,image):
    img_shape = np.shape(image)
    print('shape: ', img_shape)

    # takes in a list of quads for the image and the cat.
    # for each pair of quads, create a wcs object
    # for each wcs, find the ra and dec of the image corners
    
    img_coords = img_data.loc[list(img_quad),['x','y']].values
    cat_coords = cat_data.loc[list(cat_quad),['RA','DE']].values
    img_A, img_B, img_C, img_D, _ = findABCD(img_coords)
    cat_A, cat_B, cat_C, cat_D, _ = findABCD(cat_coords)
    w = WCS(naxis=2)
    w.wcs.crval = (cat_A[0],cat_A[1])
    w.wcs.crpix = (img_A[0],img_A[1])
    w.wcs.cdelt = (img_shape[0]/3600,img_shape[1]/3600)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # calculate the world coordinates of the image centre
    pxlcen = np.array([img_shape[0]/2,img_shape[1]/2])
    wldcen = w.all_pix2world(pxlcen[0],pxlcen[1],1)

    return w, wldcen, pxlcen

def calculate_quadrilateral_area(coords):
    # Coords should be an array of shape (4, 2)
    a, b, c, d = coords
    # Calculate vectors
    ab = b - a
    ad = d - a
    bc = c - b
    # Compute the areas of the two triangles
    area1 = np.abs(np.cross(ab, ad)) / 2
    area2 = np.abs(np.cross(bc, ab)) / 2
    # Sum the areas of the triangles to get the area of the quadrilateral
    return area1 + area2

coords = [[11.75632222, 11.97369972],
 [11.51820064, 12.10740374],
 [11.7608145,  12.07167874],
 [11.62139425, 12.10887813]]


