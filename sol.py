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

def index_to_coords(ind, quads, data):
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
        coords = data.loc[list(quad),['RA','DE']].values
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
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    # THIS SECTION REGARDS THE CREATION OF THE CD MATRIX
    # calculate the scale
    img_scale = np.linalg.norm(np.subtract(img_B, img_A))
    cat_scale = np.linalg.norm(np.subtract(cat_B, cat_A))
    scale = cat_scale / img_scale # degrees per pixel

    # calculate the rotation matrix
    img_AB = np.subtract(img_B, img_A)
    cat_AB = np.subtract(cat_B, cat_A)
    theta = np.arccos(np.dot(img_AB, cat_AB) / (np.linalg.norm(img_AB) * np.linalg.norm(cat_AB)))

    # calculate the CD matrix
    cd11 = scale * np.cos(theta)
    cd12 = -scale * np.sin(theta)
    cd21 = scale * np.sin(theta)
    cd22 = scale * np.cos(theta)
    w.wcs.cd = np.array([[cd11, cd12], [cd21, cd22]])
    return w




print('------------------')

# setting up stores for plotting
all_img_quads = []
all_cat_quads = []


cat_data, cat_quads, cat_codes, cat_tree, cat_tree_cartesian = load_catalogue()
# initialising time of solve
t1 = time.time()
# creating img_data, img_quads, img_codes
img_data, image_size, img_tree, image, target, initial_image, img_quads, img_codes = load_image('test_sets/60arcmin1.fits')

# limiting N to eaither the number of stars detected by blob detection or an input value
N_max = np.min([70,len(img_data)])
# creating a None-Object for WCS incase of failure
w = None
# setting intial found condition to false
found = False

# looping through N, adding one star at a time
for N in range(4,N_max):
    # if a verified match has been found, break the loop
    if found == True:
        break
    # generate the quads and hashcodes for the image (only for the latest star added)
    img_quads, img_codes = img.generateQuads(N, img_data, image_size, img_quads, img_codes, image)

    # printing the nyumber of NEW quads generated
    print('image quads found: ', len(img_quads))
    # check if their are any quads
    if len(img_quads) == 0:
        # if there are no new quads, continue
        continue
    
    # find matching codes in the catalogue
    matching_img_indices, matching_cat_indices = check_img_codes_for_matches(img_codes, cat_tree, 0.01)


    # for each matching hashcode, convert the hashcode into a quad and then into a WCS object
    for i in range(len(matching_cat_indices)):

        # taking each cat index and img index
        cat_index = matching_cat_indices[i]
        img_index = matching_img_indices[i]

        # convert the quads into a list of stars
        cat_stars, cat_quad = index_to_coords(cat_index, cat_quads, cat_data)
        img_stars, img_quad = index_to_coords(img_index, img_quads, img_data)

        # add the quads to the list of all quads for plotting
        all_cat_quads.append(cat_quad)
        all_img_quads.append(img_quad)

        # using the pair of quads to create a WCS object
        w = coords_to_WCS(img_stars, cat_stars)


        # test the WCS object against the image
        # query the quad centre coordinates against the cat_tree_cartesian
        cat_centroid = utils.centroid(cat_data.loc[list(cat_quad),['x','y','z']].values)
        ind = cat_tree_cartesian.query_ball_point(cat_centroid, r = 2 * np.sin(np.radians(2)/2))
        # convert the indices into a list of stars
        cat_test = cat_data.loc[list(cat_data.iloc[ind].index),['RA','DE']].values


        # convert the list of stars into pixel coordinates
        cat_pix = w.all_world2pix(cat_test,1)

        # query the cat_pix against the img_tree
        ind2 = img_tree.query_ball_point(cat_pix, r = 10)

        # find the number of non-empty lists
        non_empty = np.count_nonzero(ind2)

        if non_empty >= 10:
            # find the RA and DE of centre of the image
            width = np.shape(image)[0]
            height = np.shape(image)[1]
            centre = w.all_pix2world(width/2, height/2, 1)

            cd_matrix = w.wcs.cd

            # Using arctan2 to get the angle
            roll_angle_rad = np.arctan2(-cd_matrix[0, 1], cd_matrix[1, 1])

            # Convert the angle to degrees
            roll_angle_deg = np.rad2deg(roll_angle_rad)

            # Adjust the angle to be in the range [0, 360]
            roll_angle_deg = (roll_angle_deg + 180) % 360
            print(img_quad)
            print('centre: ', centre)
            print('roll (degrees E of N): ', roll_angle_deg)
 
            found = True
            break
        else:
            w = None

            continue

    img_quads = []
    img_codes = []
if w == None:
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







# adding the quads
q, corners_cat = plots.makePolygons(all_cat_quads, cat_data)
ax[2].add_collection(q)

ax[2].set_title('Image with WCS')
plt.show()


