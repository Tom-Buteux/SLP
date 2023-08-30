import numpy as np

def findABCD(coordinates):
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

    return A, B, C, D, np.max(distances)




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

