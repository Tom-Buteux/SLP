"""
Catalogue Steps:
1. load in catalogue
2. split into healpixels, only include top N brightest stars in each healpixel
3. itterate through each healpixel create quads between X and Y in scale length. Scale length is determined using the angular seperation of the two furthest stars.
4. for each quad, convert into a hashcode
OUTPUT: cat_data (dataframe containing all stars in catalogue), cat_codes (4D kd tree containing all hashcodes), cat_quads (list of quads in catalogue, in data.index format)
"""

from astropy_healpix import HEALPix, healpy
from astropy.io import fits
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import itertools
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import utils as utils
import plots

def cat2codes(RA_lims, DE_lims, N):
    # loading in the data
    """
    northern_data = fits.open('T2_northern.fit', ignore_missing_simple=True)
    southern_data = fits.open('T2_southern.fit', ignore_missing_simple=True)

    # converting data to pandas dataframe
    northern_df = pd.DataFrame(northern_data[1].data)
    southern_df = pd.DataFrame(southern_data[1].data)

    # concatenating dataframes
    cat_data = pd.concat([northern_df,southern_df],ignore_index=True)
    """

    # loading in gaia data
    gaia0_36 = fits.open('0-36-result.fits', ignore_missing_simple=True, ignore_missing_end=True)
    gaia0_36 = pd.DataFrame(gaia0_36[1].data)
    gaia36_72 = fits.open('36-72-result.fits', ignore_missing_simple=True, ignore_missing_end=True)
    gaia36_72 = pd.DataFrame(gaia36_72[1].data)
    gaia72_90 = fits.open('72-90-result.fits', ignore_missing_simple=True, ignore_missing_end=True)
    gaia72_90 = pd.DataFrame(gaia72_90[1].data)
    gaia90_108 = fits.open('90-108-result.fits', ignore_missing_simple=True, ignore_missing_end=True)
    gaia90_108 = pd.DataFrame(gaia90_108[1].data)
    gaia108_144 = fits.open('108-144-result.fits', ignore_missing_simple=True, ignore_missing_end=True)
    gaia108_144 = pd.DataFrame(gaia108_144[1].data)
    gaia144_180 = fits.open('144-180-result.fits', ignore_missing_simple=True, ignore_missing_end=True)
    gaia144_180 = pd.DataFrame(gaia144_180[1].data)

    # concatenating dataframes
    cat_data = pd.concat([gaia0_36, gaia36_72, gaia72_90, gaia90_108, gaia108_144, gaia144_180], ignore_index=True)
    # resetting index
    cat_data = cat_data.reset_index(drop=True)
    cat_data = cat_data.copy()
    # renaming columns
    cat_data = cat_data.rename(columns={'ra':'_RAJ2000', 'dec':'_DEJ2000', 'phot_g_mean_mag':'VTmag'})

    print(cat_data.columns)
    

    # create a list of healpix pixels for each coordinate in cat_data
    """
    Future work: Here i want to automate the healpix generation for different plate scale (FOV) 
    """
    cat_data['healpix'] = healpy.ang2pix(128, cat_data['_RAJ2000'], cat_data['_DEJ2000'], nest=True, lonlat=True)



    cat_data = cat_data.copy()
    
    
    
    cat_data['count'] = 0
    healpix_count = pd.DataFrame({'healpix': cat_data['healpix'].unique()})
    healpix_count['count'] = 0



    # converting any big-endian values to little-endian
    for column in cat_data.columns:
        if cat_data[column].dtype.byteorder == '>':  # big-endian
            dtype_name = cat_data[column].dtype.name
            little_endian_type = dtype_name.replace(">", "<")
            cat_data[column] = cat_data[column].astype(little_endian_type)

    # filter data
    cat_data = cat_data[cat_data['_RAJ2000'] >= RA_lims[0]]
    cat_data = cat_data[cat_data['_RAJ2000'] <= RA_lims[1]]
    cat_data = cat_data[cat_data['_DEJ2000'] >= DE_lims[0]]
    cat_data = cat_data[cat_data['_DEJ2000'] <= DE_lims[1]]
    print('Length of filtered data: ', len(cat_data))

    # for each healpix pixel, only keep the N brightest stars
    for healpix in cat_data['healpix'].unique():
        healpix_data = cat_data[cat_data['healpix'] == healpix]
        healpix_data = healpix_data.sort_values(by=['VTmag']).head(N)
        healpix_data = healpix_data.iloc[:N]
        cat_data = cat_data.drop(cat_data[cat_data['healpix'] == healpix].index)
        cat_data = pd.concat([cat_data, healpix_data], ignore_index=True)

    # convert each point to a cartesian coordinate on a unit sphere
    
    # relabel columns to 'RA' and 'DE
    cat_data = cat_data.rename(columns={'_RAJ2000': 'RA', '_DEJ2000': 'DE'})
    cat_data['x'] = np.cos(np.radians(cat_data['DE'])) * np.cos(np.radians(cat_data['RA']))
    cat_data['y'] = np.cos(np.radians(cat_data['DE'])) * np.sin(np.radians(cat_data['RA']))
    cat_data['z'] = np.sin(np.radians(cat_data['DE']))

    
    """
    # set any HIP numbers that are -2147483648 to NaN
    cat_data['HIP'] = cat_data['HIP'].replace(-2147483648, np.nan)
    """
    # sort cat_data by healpix pixel
    cat_data = cat_data.sort_values(by=['healpix', 'VTmag'])

    # reset index
    cat_data = cat_data.reset_index(drop=True)
    
    
    # create quads and hashcodes lists
    quads = []
    hashcodes = []

    # create a cKDTRee object for the x, y, z coordinates
    tree = cKDTree(cat_data[['x', 'y', 'z']])

    # setting up Dmin and Dmax
    Dmin = 2 * np.sin(np.radians(0.15)/2)
    Dmax = 2 * np.sin(np.radians(0.35)/2)

    print ('Dmin: ', Dmin)
    print ('Dmax: ', Dmax)



    # run the pass three times
    for i in range(3):
        run_pass(cat_data, tree, Dmax, Dmin, quads, hashcodes, N=7)
        print(' Number of quads: ', len(quads))


    
    #print('ignoring saves when testing\nre-enable saves in cat.py when changes are complete')
    print(cat_data)
    
    # plotting the stars
    fig = plt.figure(figsize=(10,10))
    plt.scatter(cat_data['RA'], cat_data['DE'], s=30000/(10**(cat_data['VTmag']/2.5))*2,color='black')
    # plot the star with index =4
    ind = 0
    plt.plot(cat_data['RA'][ind], cat_data['DE'][ind], 'r+', fillstyle='none')
    # draw a 0.35 degree circle around the star index ind
    circle = plt.Circle((cat_data['RA'][ind], cat_data['DE'][ind]), 0.35, color='r', fill=False)
    ax = plt.gca()
    ax.add_artist(circle)
    # draw a blue circle r=0.25 degrees
    circle = plt.Circle((cat_data['RA'][ind], cat_data['DE'][ind]), 0.25, color='b', fill=False)
    ax.add_artist(circle)
    # draw a green circle r=0.15 degrees
    circle = plt.Circle((cat_data['RA'][ind], cat_data['DE'][ind]), 0.15, color='g', fill=False)
    ax.add_artist(circle)


    plt.show()
    


    
    # save quads and hashcodes as lists
    np.save('quads.npy', quads)
    np.save('hashcodes.npy', hashcodes)

    # save cat_data as a DataFrame
    with open('cat_data.pkl', 'wb') as f:
        pkl.dump(cat_data, f)
    

    




    
def run_pass(cat_data, tree, Dmax, Dmin, quads, hashcodes, N=7):
    # creating quads and hashcodes
    for healpix in cat_data['healpix'].unique():
        #ending the loop if there are 100 quads

        print('healpix: ', healpix)
        found = False
        # creating a list of all stars in the healpix pixel
        stars = cat_data[cat_data['healpix'] == healpix].index.tolist()
        
        # itterate through each star in stars
        for star in stars:
            if found == True:
                break

            # if star has a 'count' greater then or equal to N, continue
            if cat_data.loc[star, 'count'] >= N:
                continue
            # find all stars within Dmax of star
            neighbours = tree.query_ball_point(cat_data.loc[star, ['x', 'y', 'z']], Dmax)
            # drop star from neighbours
            neighbours.remove(star)

            # ensure that there are at least 3 neighbours
            if len(neighbours) < 3:
                continue

            # sort neighbours by VTmag
            neighbours = cat_data.loc[neighbours].sort_values(by=['VTmag']).index.tolist()
            
            # create a combination of 3 stars from neighbours
            for combination in itertools.combinations(neighbours, 3):
                # create a quad from the combination
                quad = [star, combination[0], combination[1], combination[2]]

                # check if any of the stars in quad have a count greater than or equal to N
                if (cat_data.loc[quad, 'count'] >= N).any():
                    continue

                # make quad into a sorted tuple
                quad = tuple(sorted(quad))

                # check if quad is already in quads
                if quad in quads:
                    #print('quad already in quads')
                    continue


                inp = cat_data.loc[quad, ['x', 'y', 'z']].values

                [A,B,C,D],scale = utils.findABCD(inp)
                

                if (scale < Dmin) or (scale > Dmax):
                    #print('scale not in range')
                    continue
                
                midpoint = np.mean([A, B], axis=0)
                # check if C and D are within 0.5*AB of the midpoint
                distance = np.linalg.norm(np.subtract(midpoint,A))
                if (np.linalg.norm(np.subtract(C,midpoint)) > distance) or (np.linalg.norm(np.subtract(D,midpoint)) > distance):
                    continue

                # check which healpixel the centroid of the quad is in
                centroid = cat_data.loc[quad, ['RA','DE']].mean()
                
                centroid_healpix = healpy.ang2pix(128, centroid['RA'], centroid['DE'], nest=True, lonlat=True)

                # if the centroid is not in the same healpixel as the quad, continue
                if centroid_healpix != healpix:
                    #print('centroid not in same healpix as quad')
                    continue
                print('centroid:\n',centroid)
                # here is code to find the cartesian coordinate of the quad centroid
                centroid_x = np.cos(np.radians(centroid['DE'])) * np.cos(np.radians(centroid['RA']))
                centroid_y = np.cos(np.radians(centroid['DE'])) * np.sin(np.radians(centroid['RA']))
                centroid_z = np.sin(np.radians(centroid['DE']))

                # for a unit sphere, the normal of the plane is the centroid of the quad
                plane_normal = [centroid_x, centroid_y, centroid_z]

                def project_point_to_plane(point, n): # n is the normal of the plane
                    # find the distance from the point to the plane
                    # vec is the vector from the point on plane to the point
                    vec = np.subtract(point, np.array([0,0,0]))
                    # d is the distance from the point to the plane
                    d = np.dot(vec, n)
                    # project the point onto the plane
                    projected_point = np.subtract(point, np.multiply(d,n))
                    return projected_point
                
                # project each point in quad onto the plane
                projected_quad = np.array([project_point_to_plane(point, plane_normal) for point in inp])

                # finding the orthogonal set of the plane normal
                u,v = utils.find_orthogonal_set(plane_normal)

                # find the coordinates of the projected quad in the new coordinate system
                new_coordinates = [[np.dot(point, u), np.dot(point, v)] for point in projected_quad] 

                # plot each polygon of new coordinates
                """
                fig, ax = plt.subplots(1, 2, figsize=(24, 12))
                pts = plots.order_points(new_coordinates)
                poly = Polygon(pts, closed=True, fill=False, color='r', linewidth=2)
                ax[0].add_patch(poly)
                
                min_x = min([point[0] for point in new_coordinates])
                max_x = max([point[0] for point in new_coordinates])
                min_y = min([point[1] for point in new_coordinates])
                max_y = max([point[1] for point in new_coordinates])
                ax[0].set_xlim(min_x, max_x)
                ax[0].set_ylim(min_y, max_y)

                # for each quad, plot this againt the catalogue as a polygon
                pts = plots.order_points(cat_data.loc[quad, ['RA', 'DE']].values)
                poly = Polygon(pts, closed=True, fill=False, color='r', linewidth=2)
                ax[1].add_patch(poly)
                ax[1].scatter(cat_data['RA'], cat_data['DE'], s=30000/(10**(cat_data['VTmag']/2.5))*2,color='black')
                plt.show(block=False)
                plt.waitforbuttonpress()
                plt.close()
                """
                
                    


           



                
                





                



                # add quad to quads
                quads.append(quad)
                # create hashcode for quad
                hashcode = tuple(utils.hashcode(new_coordinates))
                hashcodes.append(hashcode)
                # increment count for each star in quad
                cat_data.loc[list(quad), 'count'] += 1
                found = True
                break

        
            

                






# running code
cat2codes([0,360],[0,90],7)
#cat2codes([70,90],[70,90],7)
#cat2codes([36,39],[55,57],10)
#cat2codes([9,14],[9,14],5)

