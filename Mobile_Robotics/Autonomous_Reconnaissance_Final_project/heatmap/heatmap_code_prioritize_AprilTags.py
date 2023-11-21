# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 14:42:22 2022

@author: 12037
"""

# Load the PIL and numpy libraries
import numpy as np
import math
import seaborn as sns
import matplotlib.pylab as plt

ori_col = 384/2 + 7
ori_row = 384/2 - 10

# Interpret this image as a numpy array, and threshold its values to

with open('map.txt') as f:
    lines = f.readlines()
    #if not lines[0:4] == "data":
    grid_in_1d = np.array(lines[-2][7:-2].replace(" ","").replace("\n","").split(",")).astype(np.float)
    grid_in_2d = grid_in_1d.reshape((384, 384)).T
    grid_in_2d = np.flip(grid_in_2d,0)

    

recovered_path = []
with open('AprilTagModel/robot_locations_tags_r3.txt') as f:
    lines = f.readlines()
    for line in lines:
        try:
            x = np.float(line.split(" ")[0])
            y = np.float(line.split(" ")[1])
            new_x, new_y = int(ori_row - (x*20)), int(ori_col - (y*20))
            recovered_path.append((new_x,new_y))
        except:
            pass
        
occupancy_grid = grid_in_2d

occupancy_grid = occupancy_grid.astype(int)

nrow,ncol = occupancy_grid.shape

# AprilTag Coordinates
# For simplicity we have hardcoded the tag locations and pick the ones that have been detected
# We are able to output tag positions after each mapping.
# These hardcoded locations came from the output of tag positions
tag_coords = []
tag_coords.append((int(ori_row - (-0.148*20)), int(ori_col - (1*20))))
tag_coords.append((int(ori_row - (-2.95*20)), int(ori_col - (0.22*20))))
tag_coords.append((int(ori_row - (-0.05*20)), int(ori_col - (-2*20))))
tag_coords.append((int(ori_row - (1.3*20)), int(ori_col - (-0.5*20))))
tag_coords.append((int(ori_row - (0.61*20)), int(ori_col - (-1.51*20))))
tag_coords.append((int(ori_row - (-1.26*20)), int(ori_col - (1.08*20))))

tags_detected = [True, True, True, True, True, True]
radius = 15


# using recovered path compare each point in the grid to each point on the path to determine heatmap value
# use inverse square law
heatmap_vals = np.zeros_like(occupancy_grid)
recovered_path = recovered_path[::10]
for pt in recovered_path:
    print(pt)
    for r in range(nrow):
        for c in range(ncol):
            if occupancy_grid[r,c] == 0:
                d = math.dist(pt,(r,c))
                d_tags = []
                for i in range(len(tags_detected)):
                    if tags_detected[i]:
                        d_tags.append(math.dist((r,c),tag_coords[i]))
                if len(d_tags) > 0:
                    tag_dist = min(d_tags)
                else:
                    tag_dist = radius
                if d == 0:
                    d = 1
                intensity = 1000/(np.pi*(d**(1/2)))
                heatmap_vals[r,c] += intensity
                
                tag_proximity = intensity * ((radius - tag_dist)/radius)
                heatmap_vals[r,c] -= tag_proximity/2
            

# add influence of previous heatmaps
heatmap_prev_round = np.load('AprilTagModel/heatmap_tags_r2.npy')
heatmap_vals = heatmap_vals + heatmap_prev_round

# normalize heatmap values between 0 and 1
heatmap_vals = heatmap_vals/np.max(heatmap_vals)



mask = occupancy_grid == 1
inv_mask = occupancy_grid == 0
min_intensity = np.where(heatmap_vals == min(heatmap_vals[inv_mask]))
initialization = [min_intensity[0][0],min_intensity[1][0]]
crop_top = 100
crop_bottom = 250
crop_right = 260
crop_left = 130
heatmap_cropped = heatmap_vals[crop_top:crop_bottom,crop_left:crop_right]
initialization_cropped = [initialization[0] - crop_top,initialization[1] - crop_left]
mask_cropped = mask[crop_top:crop_bottom,crop_left:crop_right]
ax = sns.heatmap(heatmap_cropped, mask=mask_cropped, vmin = 0.15, vmax=1, square=True,  cmap="jet")
plt.plot(initialization_cropped[1],initialization_cropped[0],'w*',linewidth=20, markersize=20)
plt.savefig('AprilTagModel/heatmap_tags_r3.png',dpi=800)
plt.show()

np.save('AprilTagModel/heatmap_tags_r3.npy',heatmap_vals)


new_x = (ori_row - initialization[0])/20
new_y = (ori_col - initialization[1])/20


