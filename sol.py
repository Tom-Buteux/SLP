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
from scipy.spatial import ConvexHull




print('------------------')
# setting up stores for plotting
img_store = []
cat_store = []

# loading in cat_data, cat_quads, cat_codes
cat_data = pd.read_pickle('cat_data.pkl')
cat_quads = np.load('quads.npy', allow_pickle=True)
cat_codes = np.load('hashcodes.npy', allow_pickle=True)
cat_tree = cKDTree(cat_codes)
cat_tree_cartesian = cKDTree(cat_data[['x','y','z']].values)
print('catalogue data loaded successfully')
print(len(cat_quads), 'catalogue quads')

# creating img_data, img_quads, img_codes
file = 'test_sets/60arcmin3.fits'
t1 = time.time()
img_FOV = 1 # degrees (greatest dimension)
img_data, image_size, img_tree, image, target, initial_image  = img.imgSetUp(file)
img_quads = []
img_codes = []
img_store = []
cat_store = []
wcs_store = []

N_max = np.min([20,len(img_data)])
w = None
found = False
for N in range(4,N_max):
    img_quads = []
    img_codes = []
    if found == True:
        break

    img_quads, img_codes = img.generateQuads(N, img_data, image_size, img_quads, img_codes, image)
    print('N = ', N)
    # test each quad against the catalogue
    try:
        cat_dist, cat_ind = cat_tree.query(img_codes, k = 1)
        print(len(cat_ind), 'matches found')
        print('cat_dist: ', cat_dist)
        print('cat_ind: ', cat_ind)

        
    except:
        print('no quads built')
        img_quads = []
        img_codes = []
        continue
    
    limit = 0.01
    # remove any cat_ind that the dist is above 0.03
    cat_ind = cat_ind[cat_dist < limit]
    img_ind = np.where(cat_dist < limit)
    cat_dist = cat_dist[cat_dist < limit]

    print('cat_dist: ', cat_dist)
    print('cat_ind: ', cat_ind)
    print('img_ind: ', img_ind)

     
    # if there are matches, find the equivalent quads in the catalogue
    if cat_ind.size > 0:
        for i in range(len(cat_ind)):
            print('cat quad: ', tuple(cat_quads[cat_ind[i]]))
            print('img quad: ', tuple(img_quads[img_ind[0][i]]))
            cat_store.append(tuple(cat_quads[cat_ind[i]]))
            img_store.append(tuple(img_quads[img_ind[0][i]]))


    if N < N_max-1:
        continue
    else:
        print('STORES')
        print('img_store: \n', img_store)
        print('cat_store: \n', cat_store)
        exit()


    """ CONTINUE HERE, NEXT WE NEED TO TEST IF 3 OF THE MATCHES ARE WITHIN THE IMAGE FOV OF EACH OTHER """
    # if there are no matches, continue

    # if cat_ind is not empty
    if cat_ind.size > 0:
        img_ind = [i for i, x in enumerate(cat_ind) if x != []]

        if np.array(img_ind).size == 0:
            continue

        cat_ind = cat_ind[img_ind][0]
        for ind in cat_ind:
            w,wldcen,pxlcen = utils.quad2wcs(img_quads[img_ind[0]], img_data, cat_quads[ind], cat_data, image)
            print('crval: ',w.wcs.crval)
            wcs_store.append(w)
            img_store.append(img_quads[img_ind[0]])
            cat_store.append(cat_quads[ind])

    else:
        continue
    # reset the quads and codes
    # if the wcs has 2 or more entries, find any wcs where the crvals are within img_FOV deg of each other
    wcs_ind=[]
    if len(wcs_store) > 2:
        # create a distance matrix of the wcs crvals
        wcs_crvals = np.array([wcs.wcs.crval for wcs in wcs_store])
        print('wcs_crvals: \n', wcs_crvals)
        # convert to radians
        wcs_crvals = np.radians(wcs_crvals)
        
        # convert the crvals to cartesian coordinates on a unit sphere using numpy broadcasting
        wcs_crvals_cart = np.array([np.cos(wcs_crvals[:,0])*np.cos(wcs_crvals[:,1]), np.sin(wcs_crvals[:,0])*np.cos(wcs_crvals[:,1]), np.sin(wcs_crvals[:,1])]).T
        print('wcs_crvals_cart: \n', wcs_crvals_cart)
        # create a distance matrix of the wcs crvals using numpy broadcasting, the distance should be the cartesian distance
        print('length (xyz) of wcs_crvals: ', np.linalg.norm(wcs_crvals_cart, axis=1))
        wcs_dist = np.linalg.norm(wcs_crvals_cart[:,None] - wcs_crvals_cart[None,:], axis=2)
        print('wcs_dist: \n', wcs_dist)
        # find the indices of the wcs that are within img_FOV of each other and not diagonal or repeated
        wcs_ind = np.argwhere((wcs_dist < img_FOV * 1.74109367e-02) & (wcs_dist > 0))
        # find the unique indices
        wcs_ind = np.unique(wcs_ind[:,0])
        if wcs_ind.size == 0:
            continue
        print('wcs_ind: \n', wcs_ind)
        # if there are 2 or more results that agree, break the loop
        if len(wcs_ind) > 1:
            found = True
            break
        else:
            continue
exit()
# filter the img_store and cat_store to only include the wcs that agree
img_store = [img_store[i] for i in wcs_ind]
cat_store = [tuple(cat_store[i]) for i in wcs_ind]

print('img_store: \n', img_store)
print('cat_store: \n', cat_store)

# for each quad in img_store find the a,b,c,d values
# for each quad in cat_store find the a,b,c,d values
img_coords = []
for quad in img_store:
    a,b,c,d,_ = utils.findABCD(img_data.loc[list(quad), ['x', 'y']].values)
    img_coords.append([a,b,c,d])
img_coords = np.array(img_coords)
img_coords = np.vstack(img_coords)
print('img_coords: \n', img_coords)

cat_coords = []
for quad in cat_store:
    a,b,c,d,_ = utils.findABCD(cat_data.loc[list(quad), ['RA', 'DE']].values)
    cat_coords.append([a,b,c,d])
cat_coords = np.array(cat_coords)
cat_coords = np.vstack(cat_coords)
print('cat_coords: \n', cat_coords)

# fidn the index where any duplicates occur
img_ind = np.unique(img_coords, axis=0, return_index=True)[1]
print('img_ duplicates: \n', img_ind)
# remove the duplicates
img_coords = img_coords[img_ind]
cat_coords = cat_coords[img_ind]
print('img_coords: \n', img_coords)
print('cat_coords: \n', cat_coords)

    # in left subplot, plot all the coords in img_coords
for rand in range(len(img_coords)):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    ax[0].scatter(img_coords[:,0], img_coords[:,1])
    ax[0].scatter(img_coords[rand,0], img_coords[rand,1],color='r')
    ax[0].invert_yaxis()

    ax[1].scatter(cat_coords[:,0], cat_coords[:,1])
    ax[1].scatter(cat_coords[rand,0], cat_coords[rand,1],color ='r')
    ax[1].invert_xaxis()
    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.close()

  
# finding the cat_coord with the hightest RA value
world_max = cat_coords[np.argmax(cat_coords[:,0])]
world_min = cat_coords[np.argmin(cat_coords[:,0])]
pixel_max = img_coords[np.argmax(cat_coords[:,0])]
pixel_min = img_coords[np.argmin(cat_coords[:,0])]

print('world and pixel max and mins: \n', world_max, world_min, pixel_max, pixel_min)

exit()

# Prepare matrices for lstsq
A = np.zeros((2*len(pixel_deltas), 4))
B = np.zeros((2*len(pixel_deltas),))

# Fill matrices
A[::2, 0:2] = pixel_deltas
A[1::2, 2:] = pixel_deltas

B[::2] = world_deltas[:, 0]
B[1::2] = world_deltas[:, 1]


# print matrices
print('A: \n', A)
print('B: \n', B)

# Solve for the CD matrix elements
result, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
cd_matrix = result.reshape((2, 2))

print('cd_matrix: \n', cd_matrix)

# create a new wcs object
w_new = WCS(naxis=2)
w_new.wcs.crval = world_ref
w_new.wcs.crpix = pixel_ref
w_new.wcs.cd = cd_matrix
w_new.wcs.ctype = ["RA---TAN", "DEC--TAN"]

# use the new wcs object to find the image centre
img_shape = np.shape(image)
pxlcen = np.array([img_shape[0]/2,img_shape[1]/2])
wldcen = w_new.all_pix2world(pxlcen[0],pxlcen[1],1)
print('wldcen: ', wldcen)

# convert the brightest N_max stars in the image to RA and DE
refine_wld_RA, refine_wld_DEC = w_new.all_pix2world(img_data.loc[:30, ['x', 'y']].values, 1).T
# convert to a list of tuples
refine_wld = [(refine_wld_RA[i], refine_wld_DEC[i]) for i in range(len(refine_wld_RA))]

# set up the catalogue tree
cat_tree = cKDTree(cat_data[['RA', 'DE']].values)
# find the closest N_max stars in the catalogue to the image stars
cat_dist, cat_ind = cat_tree.query(refine_wld, k=1, distance_upper_bound=0.03)
print('cat_ind: ', cat_ind)
# find the indicies of the stars that have a match
img_ind = np.argwhere(cat_dist < 0.03)
cat_ind = cat_ind[img_ind]
print('img_ind: ', img_ind)
print('cat_ind: ', cat_ind)
cat_dist = cat_dist[img_ind]
print('cat_dist: ', cat_dist)

# print the coordinates of cat_ind
cat_ind = cat_ind.flatten()
ras,decs = cat_data.loc[cat_ind, ['RA', 'DE']].values.T
print('ras: ', ras)
print('decs: ', decs)






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
# plot pixel_ref as a black cross   
ax[0].plot(pixel_ref[0], pixel_ref[1], 'kx')
# plot the quads 
p, corners_img = plots.makePolygons(img_store, img_data)
# add p (patch collection) to the plot
ax[0].add_collection(p)

#plot hull points
ax[0].plot(hull_points_pixel[:,0], hull_points_pixel[:,1], 'r--', fillstyle='none')
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
# plot hull points world
ax[1].plot(hull_points_world[:,0], hull_points_world[:,1], 'r--', fillstyle='none')

cat_store = [tuple(x) for x in cat_store]
q, corners_cat = plots.makePolygons(cat_store, cat_data)
ax[1].add_collection(q)
# plot world_ref as a black cross
ax[1].plot(world_ref[0], world_ref[1], 'kx')
# in 3rd subplot, plot all cat_data within 3 deg of target
plot2_data = cat_data[(cat_data['RA'] > target[0] - 3) & (cat_data['RA'] < target[0] + 3) & (cat_data['DE'] > target[1] - 3) & (cat_data['DE'] < target[1] + 3)]
ax[2].scatter(plot2_data['RA'], plot2_data['DE'], s=10000/(10**(plot2_data['VTmag']/2.5))*2)
# plot refine_wld as red circles with no fill
ax[2].plot([x[0] for x in refine_wld], [x[1] for x in refine_wld], 'ro', fillstyle='none')
# plot the matching stars as green circles with no fill
ax[2].plot(ras, decs, 'go', fillstyle='none')
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
q, corners_cat = plots.makePolygons(cat_store, cat_data)
ax[2].add_collection(q)

ax[2].set_title('Image with WCS')
plt.show()


