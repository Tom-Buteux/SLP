# image processing steps:
# 1. load in image
# 2. preprocess
# 3. find conrers in each quadrent
# 4. itterate through to create quads
# 5. convert each quad into a hashcode
# OUTPUT: img_data (dataframe containing all stars in image), img_codes (4D kd tree containing all hashcodes), img_quads (list of quads in image, in data.index format)

# imports
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import cKDTree
from itertools import combinations
import utils
import plots
from matplotlib.patches import Polygon


# loading in the image
def imgSetUp(file):

    """ loading steps """
    # load the file using fits
    hdul = fits.open(file)

    # extract the data
    image = hdul[0].data

    # extract the header
    header = hdul[0].header

    """ producing target """

    # Create the WCS object and handle any exceptions
    try:
        t = WCS(header, naxis=2)
        # Get the image dimensions from the WCS
        try:
            image_width, image_height = t.pixel_shape
        except:
            print('fits WCS has {} axes, expected 2. Can accept 3'.format(t.naxis))
            _, image_width, image_height = t.array_shape
        # Calculate the center pixel coordinates
        center_x = image_width / 2.0
        center_y = image_height / 2.0

        center_coords = t.pixel_to_world(center_x, center_y)

        # Access the RA and Dec values directly from the SkyCoord object
        center_ra = center_coords.ra.deg
        center_dec = center_coords.dec.deg
        target = (float(center_ra), float(center_dec))
        print('target: ', target)

    except Exception as e:
        # Handle any exceptions that may occur during the WCS creation or conversion
        raise Exception("Failed to create WCS object: " + str(e))


    """ processing steps """

    

   # converting image into an 8-bit unsigned integer with values between 0 and 255
    image = image.astype(np.uint8)
    image = 255-image
    if len(np.shape(image)) == 3:
        # convert to greyscale not using cv2
        image = image[0,:,:]
    initial_image = image
    # creating a median smoothed image
    image_median = cv2.medianBlur(image, 101)
    # subtracting the median smoothed image from the original image
    image = cv2.subtract(image_median, image)
    image = 255-image
    median_subtracted = image

    """ insert variance thresholding here """
    std = np.std(image)
    std8 = std*8
    # thresholding the image so that any values above 8*std are set to 255
    image[image > std8] = 255
    # show the image
    # apply a gaussian blur to the image
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    """
    fig, ax = plt.subplots(2, 2, figsize=(15, 5))
    ax[0][0].imshow(initial_image, cmap='gray')
    ax[0][0].set_title('Initial Image')
    ax[0][1].imshow(median_subtracted, cmap='gray')
    ax[0][1].set_title('Median Subtracted Image')
    ax[1][0].imshow(image, cmap='gray')
    ax[1][0].set_title('Thresholded Image')
    ax[1][1].imshow(image_median, cmap='gray')
    ax[1][1].set_title('Median Image')
    plt.show()
    """
    


    """ setting up detector """
    image_size = min([image_height, image_width])
    # Step 3: find the corners of the image
    params = cv2.SimpleBlobDetector_Params()
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 10
    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.5
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.87
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector.create(params)

    """  finding the corners """
    # Detect blobs.
    keypoints = detector.detect(image)
    corners = [[keypoint.pt[0], keypoint.pt[1]] for keypoint in keypoints]
    sizes = [keypoint.size for keypoint in keypoints]

    coords = sorted(zip(sizes, corners), reverse=True)
    sizes, corners = map(list, zip(*coords))


    # creating img_data dataframe containing all stars in image cols = ['x', 'y', 'size']
    img_data = pd.DataFrame(corners, columns=['x', 'y'])    
    img_data['size'] = sizes
    img_data = img_data.sort_values(by=['size'], ascending=False)
    img_data['count'] = 0
    img_tree = cKDTree(img_data[['x', 'y']])
    # resetting the index
    img_data = img_data.reset_index(drop=True)

    return img_data, image_size, img_tree, image, target, initial_image


# creating progressive function
def generateQuads(N, img_data, image_size,img_quads, img_codes, image):
    plot = True
    current_data = img_data.head(N).copy()
    star = N-1
    # find all neigbors of the star that are within 0.35 * image_size
    tree = cKDTree(current_data[['x', 'y']])
    neighbors = tree.query_ball_point(current_data.iloc[star][['x', 'y']], 0.35*image_size)
    # drop the star from the neighbors
    neighbors.remove(star)
    neighbors = sorted(neighbors)
    colours = ['r', 'g', 'b', 'y', 'm', 'c']

    # check if there are at least 4 neighbors
    if len(neighbors) < 3:
        return img_quads, img_codes
    i = 0
    j = 0
    for combination in combinations(neighbors,3):
        if j == 10:
            break
        j += 1
        
        quad = [star, *combination]

        # check if the quad is already in the list
        if tuple(sorted(quad)) in img_quads:
            continue
        # check if the scale of the quad is within 0.35 and 0.25 of the image size

        A,B,C,D,scale = utils.findABCD(current_data.loc[quad,['x','y']].values)
        if (scale > 0.35*image_size) or (scale < 0.05*image_size):

            continue
        # check if C and D lie in the within the A,B,MidAB circle
        midpoint = np.mean([A, B], axis=0)
        # check if C and D are within 0.5*AB of the midpoint
        distance = np.linalg.norm(np.subtract(midpoint,A))
        if (np.linalg.norm(np.subtract(C,midpoint)) > distance) or (np.linalg.norm(np.subtract(D,midpoint)) > distance):
            continue
        # append the quad to the list
        img_quads.append(tuple(sorted(quad)))
        img_codes.append(utils.hashcode(current_data.loc[quad,['x','y']].values))
    
        """ create a plot of all the quads """
        if plot == True:
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            ax.imshow(image, cmap='gray')
            #plot all points in current_data as an green circle with no fill
            plt.plot(current_data['x'], current_data['y'], 'go', fillstyle='none', markersize=10)
            # plot the last entry in current_data as a red circle with no fill
            plt.plot(current_data.iloc[-1]['x'], current_data.iloc[-1]['y'], 'ro', fillstyle='none', markersize=10)
            for quad in img_quads:
                if i == len(colours)-1:
                    i = -1
                i = i+1
                quad  = plots.order_points(current_data.loc[list(quad),['x','y']].values)
                # create a Polygon patch
                rect = Polygon(quad,linewidth=1,edgecolor=colours[i],facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(rect)
            plt.show(block=False)
            plt.waitforbuttonpress()
            plt.close()


    return img_quads, img_codes

"""
img_data, image_size, img_tree, image = imgSetUp('test_sets/60arcmin9.fits')
img_quads = []
img_codes = []
N_max = np.min([30, len(img_data)])
w = []
found = False
for N in range(4,N_max):
    img_quads, img_codes = generateQuads(N, img_data, image_size, img_quads, img_codes, image)
    img_quads = []
"""

