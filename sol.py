"""
Solver Steps:
1. load in img_codes, cat_codes, img_quads, cat_quads, img_data, cat_data
2. find the closest N matching hashcodes in the catalogue for each hashcode in the image
3. convert the matching hashcodes into quads and then into a list of stars and then into a WCS object
4. test the WCS object against the image
5. if the test is successful, return the WCS object
OUTPUT: w (WCS object)
"""
# imports 
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import cv2
import pandas as pd
from scipy.spatial import cKDTree
from itertools import combinations
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib as mpl
import time
import img
import utils
import plots

import argparse

# Initialize argument parser
parser = argparse.ArgumentParser(description="Plate Solving Software")

# Define arguments
parser.add_argument("-n", "--image_number", type=int, help="The image number for the file you want to access", required=True)
parser.add_argument("-f", "--fov", type=float, help="Field of view of the image (degrees)", required=True)
parser.add_argument("-p", "--pixel_threshold", type=float, help="Pixel threshold", required=True)
parser.add_argument("-m", "--match_tolerance", type=float, help="Match tolerance", required=True)

# Parse arguments
args = parser.parse_args()

# Use arguments in your code
image_file = f"test_sets/60arcmin{args.image_number}.fits"
image_FOV = args.fov
pixel_threshold = args.pixel_threshold
match_tolerance = args.match_tolerance


def load_catalogue():
    # loading in cat_data, cat_quads, cat_codes
    cat_data = pd.read_pickle('cat_data.pkl')
    cat_quads = np.load('quads.npy', allow_pickle=True)
    cat_codes = np.load('hashcodes.npy', allow_pickle=True)
    cat_tree = cKDTree(cat_codes)
    cat_tree_cartesian = cKDTree(cat_data[['x','y','z']].values)
    print('catalogue data loaded successfully')
    print(len(cat_quads), 'catalogue quads')
    return cat_data, cat_quads, cat_codes, cat_tree, cat_tree_cartesian


def load_image(file_name):
    # loading in img_data, img_quads, img_codes
    img_data, image_size, img_tree, image, target, initial_image  = img.imgSetUp(file_name)
    img_quads = []
    img_codes = []
    return img_data, image_size, img_tree, image, target, initial_image, img_quads, img_codes

def check_img_codes_for_matches(img_codes, cat_tree, acceptable_distance):
    """
    Compares image hashcodes to catalog hashcodes and finds matches within a given distance.

    Parameters:
    ----------
    img_codes : array-like
        An array of hashcodes generated from the image's star patterns.

    cat_tree : cKDTree object
        A KD-Tree generated from catalog hashcodes, used for spatial searches.

    acceptable_distance : float
        The maximum distance within which two hashcodes are considered to be a match.

    Returns:
    -------
    matching_img_indices : list of int
        List of indices in img_codes that have a matching hashcode within acceptable_distance in cat_tree.

    matching_cat_indices : list of int
        List of indices in cat_tree that match with img_codes within acceptable_distance.

    """
    # compare the img codes to the cat_tree of codes
    distances, indices = cat_tree.query(img_codes, k=1)
    # return the indices of any hashcodes that have a distance < acceptable_distance
    rows = np.array(np.where(distances < acceptable_distance)).ravel()
    # find the indices of the matching hashcodes
    matching_cat_indices = indices[rows]
    matching_img_indices = rows.tolist()
    return matching_img_indices, matching_cat_indices


def index_to_cart_coords(ind, quads, data):
    """
    Converts a list of indices to a list of coordinates.

    Parameters:
    ----------
    ind : int
        The index of the quad in the quads list.

    data : pandas.DataFrame
        The data source from which the indices were generated.

    quads : list of tuples
        A list of quads, where each quad is a tuple of indices.

    Returns:
    -------
    coords : array-like
        A list of coordinates corresponding to the indices.

    quad : tuple
        A tuple of indices corresponding to the stars in data

    """
    quad = quads[ind]
    # try x and y coords or RA and DE coords
    try:
        coords = data.loc[list(quad),['x','y','z']].values
        # find the centroid of the quad
        plane_normal = utils.centroid(coords) # this is a normal vector to the hypothesis image plane

        # use the plane normal to project points onto the plane
        projected_quad = np.array([utils.project_point_to_plane(point, plane_normal) for point in coords])
        # finding the orthogonal set of the plane normal
        u,v = utils.find_orthogonal_set(plane_normal)
        # find the angle that u makes with the x-axis
        
        # find the coordinates of the projected quad in the new coordinate system
        coords = [[np.dot(point, u), np.dot(point, v)] for point in projected_quad] 


    except:
        coords = data.loc[list(quad),['x','y']].values
    # raise an error if the coordinates are not found
    if len(coords) == 0:
        raise Exception('No coordinates found')

    return coords, quad

def coords_to_WCS(img_coords, cat_coords):
    """
    Converts 2 lists of coordinates (pixel and world) to a WCS object.

    Parameters:
    ----------
    img_coords : array-like
        A list of pixel coordinates.

    cat_coords : array-like
        A list of world coordinates.

    Returns:
    -------
    w : astropy.wcs.WCS
        A WCS object.
    """

    # reordering the coords so that A and B match in both i.e. img_A is the same star as cat_A
    img_coords, _, _ = utils.sortABCD(img_coords)
    cat_coords, _, _ = utils.sortABCD(cat_coords)
    img_A = img_coords[0]
    img_B = img_coords[1]
    cat_A = cat_coords[0]
    cat_B = cat_coords[1]

    # create a WCS object
    w = WCS(naxis=2)
    w.wcs.crpix = [img_A[0], img_A[1]]
    w.wcs.crval = [cat_A[0], cat_A[1]]
    w.wcs.ctype = ["X", "Y"]
    # THIS SECTION REGARDS THE CREATION OF THE CD MATRIX
    # calculate the scale
    img_scale = np.linalg.norm(np.subtract(img_B, img_A))
    cat_scale = np.linalg.norm(np.subtract(cat_B, cat_A))
    scale = cat_scale / img_scale # degrees per pixel

    # calculate the rotation matrix
    img_AB = np.subtract(img_B, img_A)
    cat_AB = np.subtract(cat_B, cat_A)
    theta = np.arccos(np.dot(img_AB, cat_AB) / (np.linalg.norm(img_AB) * np.linalg.norm(cat_AB))) + np.radians(90)



    # calculate the CD matrix
    cd11 = scale * np.cos(theta)
    cd12 = -scale * np.sin(theta)
    cd21 = scale * np.sin(theta)
    cd22 = scale * np.cos(theta)
    w.wcs.cd = np.array([[cd11, cd12], [cd21, cd22]])
    
    return w

def test_WCS(cat_quad, cat_data, cat_tree_cartesian, image_FOV, w, threshold, img_quad_xy):
    """
    Tests a WCS object against the image to see if the WCS object is a good fit.

    Parameters:
    ----------
    cat_quad : tuple
        A tuple of indices corresponding to the stars in cat_data.

    cat_data : pandas.DataFrame
        The data source from which the indices were generated.

    cat_tree_cartesian : cKDTree object
        A KD-Tree generated from catalog cartesian coordinates, used for spatial searches.

    image_FOV : float
        The field of view of the image in degrees.

    w : astropy.wcs.WCS
        A WCS object to be tested

    threshold : float
        The maximum distance (in pixel space) within which two stars are considered to be a match.

    Returns:
    -------
    verified_matches : int
        The number of stars in the cat_quad that are within threshold range of a star in the image.

    """

    # find the 'x','y','z' centroid of the cat quad
    cat_centroid = utils.centroid(cat_data.loc[list(cat_quad),['x','y','z']].values)
    normal_vector = cat_centroid
    # search the area around the quad centroid for stars within the FOV of the image
    ind = cat_tree_cartesian.query_ball_point(cat_centroid, r = 2 * np.sin(np.radians(image_FOV)/2))

    plane_normal = cat_centroid # this is a normal vector to the hypothesis image plane
    # finding the orthogonal set of the plane normal
    u,v = utils.find_orthogonal_set(plane_normal)
    """ rotate the cat_pix to match the image """
    cat_test_xyz = cat_data.loc[cat_data.iloc[ind].index,['x','y','z']].values
    cat_quad_xyz = cat_data.loc[list(cat_quad),['x','y','z']].values
    cat_quad_proj = [utils.project_point_to_plane(point, plane_normal) for point in cat_quad_xyz]
    cat_test_proj = [utils.project_point_to_plane(point, plane_normal) for point in cat_test_xyz]
    cat_quad_uv =  [[np.dot(point, u), np.dot(point, v)] for point in cat_quad_proj]
    cat_test_uv =  [[np.dot(point, u), np.dot(point, v)] for point in cat_test_proj]
    img_quad_xy = img_quad_xy
    # find A for both the img and cat quads
    cat_quad_uv,_ = utils.findABCD(cat_quad_uv)
    img_quad_xy,_ = utils.findABCD(img_quad_xy)
    # convert to arrays
    cat_quad_uv = np.array(cat_quad_uv)
    img_quad_xy = np.array(img_quad_xy)
    cat_quad_xy = w.all_world2pix(cat_quad_uv,1)
    cat_test_xy = w.all_world2pix(cat_test_uv,1)


    
    
    cat_AB = np.subtract(cat_quad_xy[1], cat_quad_xy[0])
    img_AB = np.subtract(img_quad_xy[1], img_quad_xy[0])
    # find the angle from cat_A to img_A
    theta = np.arctan2(cat_AB[0]*img_AB[1] - cat_AB[1]*img_AB[0], cat_AB[0]*img_AB[0] + cat_AB[1]*img_AB[1])

    cat_quad_xy = np.array(utils.rotate_2d_points_around_point(cat_quad_xy, theta, w.wcs.crpix))
    cat_test_xy = np.array(utils.rotate_2d_points_around_point(cat_test_xy, theta, w.wcs.crpix))
    # check to see how many of the stars are in the image (threshold variable)
    ind2 = img_tree.query_ball_point(cat_test_xy, r = threshold) # ind2 is the iloc indicies of the stars in img_data that are within threshold of the cat_quad
    # find the number of non-empty lists
    verified_matches = np.count_nonzero(ind2)
    
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].imshow(image, cmap='gray')
    ax[0].scatter(img_quad_xy[:,0], img_quad_xy[:,1], s=100, c=['r','g','b','y'])
    ax[0].set_title('Image')
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[1].imshow(image, cmap='gray')
    ax[1].scatter(cat_quad_xy[:,0], cat_quad_xy[:,1], s=100, c=['r','g','b','y'])
    ax[1].set_title('Catalogue')

    ax[1].set_xlabel('u')
    ax[1].set_ylabel('v')
    plt.show()
    """
    

    return verified_matches, normal_vector, cat_test_xy


def calculate_centre_and_roll(w,image,n):
     # find the RA and DE of centre of the image
    width = np.shape(image)[0]
    height = np.shape(image)[1]
    centre = w.all_pix2world(width/2, height/2, 1)

    centre = utils.project_point_to_sphere(centre,n)

    centre = utils.convert_cartesian_to_RA_DEC(centre)
    print('centre in RA DEC: ', centre)
    cd_matrix = w.wcs.cd

    # Using arctan2 to get the angle
    roll_angle_rad = np.arctan2(-cd_matrix[0, 1], cd_matrix[1, 1])

    # Convert the angle to degrees
    roll_angle_deg = np.rad2deg(roll_angle_rad)

    # Adjust the angle to be in the range [0, 360]
    roll_angle_deg = (roll_angle_deg + 180) % 360

    print('roll (degrees E of N): ', roll_angle_deg)
    return centre, roll_angle_deg




#


print('------------------')

# setting up stores for plotting
all_img_quads = []
all_cat_quads = []

import pandas as pd

# Create a list to hold each row as a DataFrame
rows_list = [
    {'Name': 'Load Catalogue', 'Value': 0, 'Times Run': 0},
    {'Name': 'Load Image', 'Value': 0, 'Times Run': 0},
    {'Name': 'Generate Image Quads', 'Value': 0, 'Times Run': 0},
    {'Name': 'Match Codes', 'Value': 0, 'Times Run': 0},
    {'Name': 'Convert Index to Coords', 'Value': 0, 'Times Run': 0},
    {'Name': 'Convert to WCS', 'Value': 0, 'Times Run': 0},
    {'Name': 'Test WCS', 'Value': 0, 'Times Run': 0},
    {'Name': 'Calculate Results', 'Value': 0, 'Times Run': 0},
    {'Name': 'Total Time', 'Value': 0, 'Times Run': np.nan},
    {'Name': 'N', 'Value': 0, 'Times Run': np.nan},
    {'Name': 'Found', 'Value': 0, 'Times Run': np.nan},
    {'Name': 'Image Name', 'Value': 0, 'Times Run': np.nan},
    {'Name': 'Image_FOV', 'Value': 0, 'Times Run': np.nan},
    {'Name': 'Pixel Threshold', 'Value': 0, 'Times Run': np.nan},
    {'Name': 'Match Tolerance', 'Value': 0, 'Times Run': np.nan},
    {'Name': 'Target RA', 'Value': 0, 'Times Run': np.nan},
    {'Name': 'Target DE', 'Value': 0, 'Times Run': np.nan},
    {'Name': 'Error RA', 'Value': 0, 'Times Run': np.nan},
    {'Name': 'Error DE', 'Value': 0, 'Times Run': np.nan},
    {'Name': 'Catalogue RA bounds', 'Value': 0, 'Times Run': np.nan},
    {'Name': 'Catalogue DE bounds', 'Value': 0, 'Times Run': np.nan},
    {'Name': 'Device Name', 'Value': 0, 'Times Run': np.nan},

]

# Convert each dictionary in rows_list to a DataFrame and concatenate them
diagnostics = pd.concat([pd.DataFrame([i]) for i in rows_list], ignore_index=True)


# Set 'Name' as the index
diagnostics.set_index('Name', inplace=True)





# to store WCS objects
w_store = []

t_load_cat = time.time()
cat_data, cat_quads, cat_codes, cat_tree, cat_tree_cartesian = load_catalogue()
time_to_load_cat = time.time() - t_load_cat
# adding these times to the diagnostics
diagnostics.loc[diagnostics.index == 'Load Catalogue', 'Value'] += time_to_load_cat
diagnostics.loc[diagnostics.index == 'Load Catalogue', 'Times Run'] += 1


# initialising time of solve
t1 = time.time()
t_load_img = time.time()
# creating img_data, img_quads, img_codes

img_data, image_size, img_tree, image, target, initial_image, img_quads, img_codes = load_image(image_file)
time_to_load_img = time.time() - t_load_img
# adding these times to the diagnostics
diagnostics.loc[diagnostics.index == 'Load Image', 'Value'] += time_to_load_img
diagnostics.loc[diagnostics.index == 'Load Image', 'Times Run'] += 1

# limiting N to eaither the number of stars detected by blob detection or an input value
N_max = np.min([70,len(img_data)])
# creating a None-Object for WCS incase of failure
w = None
# setting intial found condition to false
found = 0

# looping through N, adding one star at a time
for N in range(4,N_max):
    # if a verified match has been found, break the loop
    if found == 3:
        break
    t_loop = time.time()
    # generate the quads and hashcodes for the image (only for the latest star added)
    img_quads, img_codes = img.generateQuads(N, img_data, image_size, img_quads, img_codes, image)
    time_to_generate_img_quads = time.time() - t_loop
    # adding these times to the diagnostics
    diagnostics.loc[diagnostics.index == 'Generate Image Quads', 'Value'] += time_to_generate_img_quads
    diagnostics.loc[diagnostics.index == 'Generate Image Quads', 'Times Run'] += 1
    # printing the nyumber of NEW quads generated
    print('image quads found: ', len(img_quads))
    # check if their are any quads
    if len(img_quads) == 0:
        # if there are no new quads, continue
        continue
    
    # find matching codes in the catalogue
    t_match = time.time()

    matching_img_indices, matching_cat_indices = check_img_codes_for_matches(img_codes, cat_tree, match_tolerance)
    time_to_match_codes = time.time() - t_match
    # adding these times to the diagnostics
    diagnostics.loc[diagnostics.index == 'Match Codes', 'Value'] += time_to_match_codes
    diagnostics.loc[diagnostics.index == 'Match Codes', 'Times Run'] += 1
    
    # for each matching hashcode, convert the hashcode into a quad and then into a WCS object
    for i in range(len(matching_cat_indices)):

        # taking each cat index and img index
        cat_index = matching_cat_indices[i]
        img_index = matching_img_indices[i]

        t_convert = time.time()
        # convert the quads into a list of stars
        cat_stars, cat_quad = index_to_cart_coords(cat_index, cat_quads, cat_data) # this has been changed to ouptut the cat_stars as a projection onto a 2D plane
        img_stars, img_quad = index_to_cart_coords(img_index, img_quads, img_data)
        time_to_convert_index_to_coords = time.time() - t_convert
        # adding these times to the diagnostics
        diagnostics.loc[diagnostics.index == 'Convert Index to Coords', 'Value'] += time_to_convert_index_to_coords
        diagnostics.loc[diagnostics.index == 'Convert Index to Coords', 'Times Run'] += 1


        
        t_WCS = time.time()
        # using the pair of quads to create a WCS object
        w = coords_to_WCS(img_stars, cat_stars)
        time_to_convert_to_WCS = time.time() - t_WCS
        # adding these times to the diagnostics
        diagnostics.loc[diagnostics.index == 'Convert to WCS', 'Value'] += time_to_convert_to_WCS
        diagnostics.loc[diagnostics.index == 'Convert to WCS', 'Times Run'] += 1


        # test the WCS object against the image
        t_test = time.time()

        number_of_matches,normal,cat_test_xy = test_WCS(cat_quad, cat_data, cat_tree_cartesian, image_FOV=image_FOV, w=w, threshold=pixel_threshold, img_quad_xy=img_stars)
        time_to_test_WCS = time.time() - t_test
        # adding these times to the diagnoistics
        diagnostics.loc[diagnostics.index == 'Test WCS', 'Value'] += time_to_test_WCS
        diagnostics.loc[diagnostics.index == 'Test WCS', 'Times Run'] += 1

        """
        plt.imshow(image, cmap='gray')
        plt.plot(img_data['x'][:N], img_data['y'][:N], 'ro', fillstyle='none')
        plt.plot(cat_test_xy[:,0], cat_test_xy[:,1], 'go', fillstyle='none')
        # limit the plot to the image size
        plt.xlim(0, image_size)
        plt.ylim(0, image_size)
        plt.show()
        """

        if number_of_matches >= 11:
            # add the quads to the list of all quads for plotting
            all_cat_quads.append(cat_quad)
            all_img_quads.append(img_quad)
            t_calc = time.time()
            centre,roll = calculate_centre_and_roll(w,image,normal)
            time_to_calculate_results = time.time() - t_calc
            # adding these times to the diagnostics
            diagnostics.loc[diagnostics.index == 'Calculate Results', 'Value'] += time_to_calculate_results
            diagnostics.loc[diagnostics.index == 'Calculate Results', 'Times Run'] += 1

 
            found += 1
            w_store.append(w)

            if found == 3:
                break
            else:
                continue
        else:
            print('WCS failed, ', number_of_matches, ' stars matched')
            continue

    img_quads = []
    img_codes = []
if len(w_store) == 0:
    w = None
    print('no WCS found')

t1 = time.time() - t1
print('time taken: ', t1, ' seconds')
#plotting the results
# in left subplot, image with quads
# in right subplot, catalogue with matching stars overlayed
fig, ax = plt.subplots(1, 3, figsize=(24, 8))
ax[0].imshow(initial_image, cmap='gray')
ax[0].set_title('Image')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
# plot the quads 
p, corners_img = plots.makePolygons(all_img_quads, img_data)
# add p (patch collection) to the plot
ax[0].add_collection(p)
# plot the image data as green circles with no fill
ax[0].plot(img_data['x'][:N_max], img_data['y'][:N_max], 'ro', fillstyle='none')

# plot the catalogue, trim the data to 0.5 eitherside of the target
plot_data = cat_data[(cat_data['RA'] > target[0] - 0.5) & (cat_data['RA'] < target[0] + 0.5) & (cat_data['DE'] > target[1] - 0.5) & (cat_data['DE'] < target[1] + 0.5)]
ax[1].scatter(plot_data['RA'], plot_data['DE'], s=30000/(10**(plot_data['VTmag']/2.5))*2)
ax[1].set_title('Catalogue')
ax[1].set_xlabel('RA')
ax[1].set_ylabel('DE')
ax[1].set_xlim(target[0] - 0.5, target[0] + 0.5)
ax[1].set_ylim(target[1] - 0.5, target[1] + 0.5)
ax[1].invert_xaxis()
ax[1].set_aspect('equal', 'box')
all_cat_quads = [tuple(x) for x in all_cat_quads]
q, corners_cat = plots.makePolygons(all_cat_quads, cat_data)
ax[1].add_collection(q)

# in 3rd subplot, plot all cat_data within 3 deg of target
plot2_data = cat_data[(cat_data['RA'] > target[0] - 3) & (cat_data['RA'] < target[0] + 3) & (cat_data['DE'] > target[1] - 3) & (cat_data['DE'] < target[1] + 3)]
ax[2].scatter(plot2_data['RA'], plot2_data['DE'], s=10000/(10**(plot2_data['VTmag']/2.5))*2)
ax[2].invert_xaxis()
# plot the image superimposed on the catalogue in the correct WC
# The extent should be in world coordinates. The corners of the image give the extent.
if w != None:
    w = w_store[0]
    # find the image corners in RA and DE
    width = np.shape(image)[0]
    height = np.shape(image)[1]
    bot_left = w.all_pix2world(0, 0, 1)
    top_left = w.all_pix2world(0, height, 1)
    top_right = w.all_pix2world(width, height, 1)
    bot_right = w.all_pix2world(width, 0,   1)
    # create a closed polygon of the image corners
    corners = np.array([top_right, bot_right, bot_left, top_left])
    poly = Polygon(corners, closed=True, fill=False, edgecolor='r', linewidth=1)
    poly2 = Polygon(corners, closed=False, fill=False, edgecolor='g', linewidth=1)
    # add the polygon to the plot
    ax[2].add_patch(poly)
    ax[2].add_patch(poly2)
    ax[2].set_aspect('equal', 'box')
    ax[2].set_xlim(target[0] + 2, target[0] - 2)
    ax[2].set_ylim(target[1] - 2, target[1] + 2)
    ax[2].set_xlabel('RA')
    ax[2].set_ylabel('DE')


error_RA = np.abs(target[0] - centre[0])
error_DE = np.abs(target[1] - centre[1])




# adding the quads
q, corners_cat = plots.makePolygons(all_cat_quads, cat_data)
ax[2].add_collection(q)

ax[2].set_title('Image with WCS')
plt.show()

# additional diagnostics
import socket
device = socket.gethostname()

RA_Min = np.round(np.min(cat_data['RA']),0)
RA_Max = np.round(np.max(cat_data['RA']),0)
DE_Min = np.round(np.min(cat_data['DE']),0)
DE_Max = np.round(np.max(cat_data['DE']),0)

RA_bounds = str(RA_Min) + "-" + str(RA_Max)
DE_bounds = str(DE_Min) + "-" + str(DE_Max)

# add the total time taken to the diagnostics
diagnostics.loc[diagnostics.index == 'Total Time', 'Value'] += t1
diagnostics.loc[diagnostics.index == 'Total Time', 'Times Run'] += 1

# adding key values to the diagnostics
diagnostics.loc[diagnostics.index == 'N', 'Value'] = N
diagnostics.loc[diagnostics.index == 'Found', 'Value'] = found
# adding the image name to the diagnostics
diagnostics.loc[diagnostics.index == 'Image Name', 'Value'] = image_file
diagnostics.loc[diagnostics.index == 'Image_FOV', 'Value'] = image_FOV
diagnostics.loc[diagnostics.index == 'Pixel Threshold', 'Value'] = pixel_threshold
diagnostics.loc[diagnostics.index == 'Match Tolerance', 'Value'] = match_tolerance
diagnostics.loc[diagnostics.index == 'Target RA', 'Value'] = target[0]
diagnostics.loc[diagnostics.index == 'Target DE', 'Value'] = target[1]
diagnostics.loc[diagnostics.index == 'Error RA', 'Value'] = error_RA
diagnostics.loc[diagnostics.index == 'Error DE', 'Value'] = error_DE
diagnostics.loc[diagnostics.index == 'Catalogue RA bounds', 'Value'] = RA_bounds
diagnostics.loc[diagnostics.index == 'Catalogue DE bounds', 'Value'] = DE_bounds
diagnostics.loc[diagnostics.index == 'Device Name', 'Value'] = device


RA_Min = np.round(np.min(cat_data['RA']),0)
RA_Max = np.round(np.max(cat_data['RA']),0)






print(diagnostics)
#append the values of the time_frame to a csv file
values = diagnostics['Value'].values
# append to row of csv file
with open('diagnostics.csv', 'a') as f:
    np.savetxt(f, values.reshape(1, len(values)), delimiter=',', fmt='%s')



