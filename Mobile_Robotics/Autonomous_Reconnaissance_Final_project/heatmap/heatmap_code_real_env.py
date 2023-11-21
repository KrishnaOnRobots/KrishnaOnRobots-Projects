# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 14:42:22 2022

@author: 12037
"""

# Load the PIL and numpy libraries
from PIL import Image,ImageDraw
import numpy as np
import math
import seaborn as sns
import matplotlib.pylab as plt

ori_col = 384/2 + 7
ori_row = 384/2 - 10

# Interpret this image as a numpy array, and threshold its values to

with open('RealEnvModel/real_env_map.csv') as f:
    lines = f.readlines()
    #if not lines[0:4] == "data":
    grid_in_1d = np.array(lines[-2][121:-3].replace(" ","").replace("\n","").split(",")).astype(np.float)
    grid_in_2d = grid_in_1d.reshape((672, 384)).T
    grid_in_2d = np.flip(grid_in_2d,0)

plt.imshow(grid_in_2d, cmap='hot', interpolation='nearest')
plt.plot(390,270,'b*',linewidth=5,markersize=10)
plt.plot(550,355,'y*',linewidth=5,markersize=10)
plt.plot(510,260,'r*',linewidth=5,markersize=10)
plt.plot(460,350,'g*',linewidth=5,markersize=10)
plt.show(block=False)

#window door elevator

## Attempting to find affine transformation between occupancy grid and robot
#  coordinate frame
door2hallway = math.dist((-9.277, 17.44),(-4.218, 9.87))

mri2elevator = math.dist((-4.288, 15.854),(-8.49, 13.25))

y = [[390, 550, 510],[270, 355, 260],[1,1,1]]
x = [[-4.218, -9.277, -8.49],[9.87, 17.44, 13.25],[1,1,1]]

a = y @ np.linalg.inv(x)

a1 = np.array([-4.218,9.87])

A = a[0:2,0:2]

a1 @ A + a[0:2,2]
    

recovered_path = []
with open('robot_locations_r3.txt') as f:
    lines = f.readlines()
    for line in lines:
        x = np.float(line.split(" ")[0])
        y = np.float(line.split(" ")[1])
        new_x, new_y = int(ori_row - (x*20)), int(ori_col - (y*20))
        recovered_path.append((new_x,new_y))
        
occupancy_grid = grid_in_2d
plt.imshow(occupancy_grid)
occupancy_grid = occupancy_grid.astype(int)
plt.show()

nrow,ncol = occupancy_grid.shape


# using recovered path compare each point in the grid to each point on the path to determine heatmap value
# use inverse square law
heatmap_vals = np.zeros_like(occupancy_grid)
recovered_path = recovered_path[::10]
for pt in recovered_path:
    print(pt)
    for r in range(nrow):
        for c in range(ncol):
            if (r,c) != pt and occupancy_grid[r,c] == 0:
                d = math.dist(pt,(r,c))
                intensity = 1000/(np.pi*(d**(1/2)))
                heatmap_vals[r,c] += intensity
heatmap_prev_round = np.load('heatmap_r2.npy')
heatmap_vals = heatmap_vals + heatmap_prev_round
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
plt.savefig('heatmap_r3.png',dpi=800)
plt.show()

np.save('heatmap_r3.npy',heatmap_vals)


new_x = (ori_row - initialization[0])/20
new_y = (ori_col - initialization[1])/20


