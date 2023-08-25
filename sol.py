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
import img
import utils

"""------------------"""

import matplotlib.pyplot as plt

print('------------------')
# loading in cat_data, cat_quads, cat_codes
cat_data = pd.read_pickle('cat_data.pkl')
cat_quads = np.load('quads.npy', allow_pickle=True)
cat_codes = np.load('hashcodes.npy', allow_pickle=True)
print('catalogue data loaded successfully')
print(len(cat_quads), 'catalogue quads')

# creating img_data, img_quads, img_codes
file = 'test_sets/60arcmin1.fits'
img_data, img_quads, img_codes, img_shape, image, target, initial_image = img.img2codes(file)
print('image processed successfully')
print(len(img_quads), 'image quads')

# convert img_codes and cat_codes into a cKDTree
cat_tree = cKDTree(cat_codes)
img_tree = cKDTree(img_data[['x','y']].values)
cat_tree_cartesian = cKDTree(cat_data[['x','y','z']].values)

# find the closest N matching hashcodes in the catalogue for each hashcode in the image
N = 3
distances, indices = cat_tree.query(img_codes, k=N)
# convert to arrays
distances = np.array(distances)
indices = np.array(indices)

# return the indices of any hashcodes that have a distance < 0.1
rows,cols = np.where(distances < 0.1)

# find the indices of the matching hashcodes
matching_indices = indices[rows,cols]

# find the matching quads for the img and cat
matching_img_quads = list([img_quads[row] for row in rows])
matching_cat_quads = list([cat_quads[col] for col in matching_indices])

# create a list of the wcs objects
wcs_list = []

for i in range(len(matching_img_quads)):
    img_coords = img_data.loc[list(matching_img_quads[i]),['x','y']].values
    cat_coords = cat_data.loc[list(matching_cat_quads[i]),['RA','DE']].values
    img_A, img_B, _ = utils.findAB(img_coords)
    cat_A, cat_B, _ = utils.findAB(cat_coords)

   # create a WCS object
    w = WCS(naxis=2)
    w.wcs.crpix = [img_A[0], img_A[1]]
    w.wcs.crval = [cat_A[0], cat_A[1]]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

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

    # --- test the WCS object against the image ---
    # query the quad centre coordinates against the cat_tree_cartesian
    cat_centroid = utils.centroid(cat_data.loc[list(matching_cat_quads[i]),['x','y','z']].values)
    ind = cat_tree_cartesian.query_ball_point(cat_centroid, r = 2 * np.sin(np.radians(0.5)/2))
    # convert the indices into a list of stars
    cat_stars = cat_data.loc[list(cat_data.iloc[ind].index),['RA','DE']].values

    # convert the list of stars into pixel coordinates
    cat_pix = w.all_world2pix(cat_stars,1)

    # query the cat_pix against the img_tree
    ind2 = img_tree.query_ball_point(cat_pix, r = 10)

    # find the number of non-empty lists
    non_empty = np.count_nonzero(ind2)

    # find the RA and DE of centre of the image
    width = img_shape[0]
    height = img_shape[1]
    centre = w.all_pix2world(width/2, height/2, 1)

    cd_matrix = w.wcs.cd

    # Using arctan2 to get the angle
    roll_angle_rad = np.arctan2(-cd_matrix[0, 1], cd_matrix[1, 1])

    # Convert the angle to degrees
    roll_angle_deg = np.rad2deg(roll_angle_rad)

    # Adjust the angle to be in the range [0, 360]
    roll_angle_deg = (roll_angle_deg + 360) % 360


    

    if non_empty > 4:
        wcs_list.append(w)
        print('centre: ', centre)
        print('roll (degrees E of N): ', roll_angle_deg)

if len(wcs_list) == 0:
    print('No solution found')
else:
    print('Solution found')
    print('Number of solutions: ', len(wcs_list))

# plotting the image in left panel
fig, ax = plt.subplots(1,3,figsize=(30,10))
ax[0].imshow(image, cmap='gray')
ax[0].plot(img_data['x'], img_data['y'], 'go', fillstyle='none')
# axis labels
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')

# on the right panel, plot the catalogue stars in green
plot_data = cat_data[(cat_data['RA'] > target[0] - 0.5) & (cat_data['RA'] < target[0] + 0.5) & (cat_data['DE'] > target[1] - 0.5) & (cat_data['DE'] < target[1] + 0.5)]
ax[1].scatter(plot_data['RA'], plot_data['DE'],s=30000/(10**(plot_data['VTmag']/2.5))*2)
# limit the axes to the target
ax[1].set_xlim(target[0] - 0.5, target[0] + 0.5)
ax[1].set_ylim(target[1] - 0.5, target[1] + 0.5)
# invert the x-axis in the right panel
ax[1].invert_xaxis()
# axis labels
ax[1].set_xlabel('RA')
ax[1].set_ylabel('DE')
# grid
ax[1].grid(True)

# plot the intitial image in the middle panel
ax[2].imshow(initial_image, cmap='gray')
ax[2].plot(img_data['x'], img_data['y'], 'go', fillstyle='none')

plt.tight_layout()


plt.show()


