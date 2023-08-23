import numpy as np



def findAB(coordinates):
    coordinates = np.array(coordinates)
    # Compute the pairwise distances using NumPy's broadcasting
    diffs = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)

    # Mask the diagonal (distance of point from itself) to avoid selecting it
    np.fill_diagonal(distances, 0)

    # Find the indices of the maximum distance
    i, j = np.unravel_index(distances.argmax(), distances.shape)

    point_A = coordinates[i]
    point_B = coordinates[j]

    # A should always be the point that the other points are closest to [MAY NEED TO CHANGE FOR JUST X-AXIS]
    if distances[i].sum() > distances[j].sum():
        point_A, point_B = point_B, point_A

    return point_A, point_B, distances.max()



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

    # compute the pairwise distances using broadcasting
    diff = quad[:, np.newaxis, :] - quad[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))

    # find the indices of the two points that are furthest apart
    max_dist = np.max(dist)
    idx = np.argwhere(dist == max_dist)
    i, j = idx[0]
    A = quad[i]
    B = quad[j]

    # find C and D based on distance from A
    CD = quad[np.logical_not(np.logical_or(quad == A, quad == B))].reshape(2,-1)
    dist_from_A = np.linalg.norm(CD - A[np.newaxis, :], axis=1)
    C, D = CD[np.argsort(dist_from_A)]
    array = [A,B,C,D]

    # setting array[0] to (0,0) and scaling the rest of the coords to match    
    i = 0
    zero_point = array[0]
    for point in array:
        array[i] = np.subtract(array[i],zero_point)
        i +=1
        
    # rotating quad so that the line  quad[0] --> quad[2] is x=y line 
    # Defining the 'up' coordinate frame
    y_axis = (0, 1)
    # Finding the angle (alpha) between quad[0] and (1,1)
    # A.B = |A||B| cos(alpha)
    alpha = np.absolute(np.arccos(np.dot(y_axis, array[1]) / (np.linalg.norm(y_axis) * np.linalg.norm(array[1]))))  # I think arc cos returns the positive value but abs() to make sure
   
    # Ensuring that the quad is rotated so that AB is inline with y=x on the image axis
    if array[1][0] >= 0:
      angle = alpha - np.deg2rad(45)
    else:
         angle = -(alpha + np.deg2rad(45))

         
    for i in range(0,4):
           array[i] = list(RotateCoordinates(x=array[i][0], y=array[i][1], angle=angle))
 
   
    #  scaling quad so array[1] = [1,1]
    x_scale = array[1][0]
    y_scale = array[1][1]

    #  scaling quad so array[1] = [1,1]
    for i in range(4):
        array[i][0] = array[i][0]/x_scale
        array[i][1] = array[i][1]/y_scale
        
 
    #  creating hashcode 
    #  if Xc + Xd > 1: 1 - each point
    #  if Xc > Xd: swap C and D
    if array[2][0] + array[3][0] > 1:
        array = np.subtract(1,array)
    
    if array[2][0] > array[3][0]:
        hashcode = [array[3],array[2]]
    else:
        hashcode = [array[2],array[3]]
        
    
    # flatten hashcode
    def flatten(l):
        return [item for sublist in l for item in sublist]
    
    
    hashcode = flatten(hashcode)

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
