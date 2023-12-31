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

    # A is closer to the mean of C and D than B is
    if np.linalg.norm(A - np.mean([C, D], axis=0)) > np.linalg.norm(B - np.mean([C, D], axis=0)):
        A, B = B, A

    # C is closer to A than D is
    if np.linalg.norm(A - C) > np.linalg.norm(A - D):
        C, D = D, C

    return [A, B, C, D], np.max(distances)




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

    # A is closer to the mean of C and D than B is
    if np.linalg.norm(A - np.mean([C, D], axis=0)) > np.linalg.norm(B - np.mean([C, D], axis=0)):
        A, B = B, A

    # C is closer to A than D is
    if np.linalg.norm(A - C) > np.linalg.norm(A - D):
        C, D = D, C

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

def project_point_to_plane(point, n): # n is the normal of the plane
    # find the distance from the point to the plane
    # vec is the vector from the point on plane to the point
    vec = np.subtract(point, np.array([0,0,0]))
    # d is the distance from the point to the plane
    d = np.dot(vec, n)
    # project the point onto the plane
    projected_point = np.subtract(point, np.multiply(d,n))
    return projected_point

def project_point_to_sphere(point,n): # n is the normal of the plane
    # find the orthogonal vectors to the normal vector
    u, v = find_orthogonal_set(n)
    # find the x,y,z coordinates of the point
    point = np.multiply(point[0],u) + np.multiply(point[1],v)
    # add the normal vector to the point
    point = point + n
    # normalize the point
    point = point / np.linalg.norm(point)
    return point

def convert_cartesian_to_RA_DEC(point):
    # convert the point to spherical coordinates
    dec = np.arcsin(point[2])
    ra = np.arctan2(point[1],point[0])

    # convert the ra and dec to degrees
    dec = np.degrees(dec)
    if ra < 0:
        ra = 2*np.pi+ra
    ra = np.degrees(ra)
    return ra, dec

def rotate_2d_points_around_point(points, angle, pivot):
    """
    Rotate 2D points around a pivot point in an anti-clockwise direction.

    Parameters:
    points (list): List of points as numpy arrays ([x, y])
    angle (float): Angle of rotation in radians.
    pivot (list): The point around which to rotate ([x, y])

    Returns:
    list: List of rotated points as numpy arrays.
    """
    
    # Define the 2D rotation matrix for anti-clockwise rotation
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    pivot = np.array(pivot)  # Convert pivot to numpy array
    rotated_points = []  # List to store rotated points

    for point in points:
        point = np.array(point)  # Convert point to numpy array
        translated_point = point - pivot  # Translate to origin (relative to pivot)
        
        # Perform anti-clockwise rotation using the rotation matrix
        rotated_translated_point = np.dot(rotation_matrix, translated_point)
        
        # Translate back to original position (relative to pivot)
        rotated_point = rotated_translated_point + pivot
        
        rotated_points.append(rotated_point)  # Add to list of rotated points

    return rotated_points
    


