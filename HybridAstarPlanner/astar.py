import heapq
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import sys
sys.path.append("/home/liilu/Desktop/COURSE/AI/plan_planning/MotionPlanning")

import glob
from CurvesGenerator.quintic_polynomial import non_holonomic_simulation


VIS_PLOT = True

class Node:
    def __init__(self, x, y, cost, pind):
        self.x = x  # x position of node
        self.y = y  # y position of node
        self.cost = cost  # g cost of node
        self.pind = pind  # parent index of node

def calc_holonomic_heuristic_with_obstacle(obsmap, goal_x, goal_y):
    print(f'---------------------Starting to compute holonomic_heuristic_with_obstacle .....----------------------')
    find_ret = len(glob.glob("holonomic_heuristic_with_obstacle.npy")) == 1
    if find_ret:
        print(f'---------------------Finished loading holonomic_heuristic_with_obstacle !----------------------')
        return np.load("holonomic_heuristic_with_obstacle.npy")
    assert obsmap[goal_x, goal_y] == 0
    qdist = np.full(obsmap.shape, np.inf)
    visited = np.full(obsmap.shape, False)
    qdist[goal_x, goal_y] = 0
    H, W = obsmap.shape
    queue = [[goal_x, goal_y]]
    while len(queue) > 0:
        new_queue = []
        for point in queue:
            if visited[point[0], point[1]]:
                continue
            visited[point[0], point[1]] = True

            if point[0]+1<H:
                qdist[point[0], point[1]] = min(qdist[point[0], point[1]], qdist[point[0]+1, point[1]]+1)
            if point[0]-1>-1:
                qdist[point[0], point[1]] = min(qdist[point[0], point[1]], qdist[point[0]-1, point[1]]+1)
            if point[1]+1<W:
                qdist[point[0], point[1]] = min(qdist[point[0], point[1]], qdist[point[0], point[1]+1]+1)
            if point[1]-1>-1:
                qdist[point[0], point[1]] = min(qdist[point[0], point[1]], qdist[point[0], point[1]-1]+1)

            assert qdist[point[0], point[1]] < np.inf
            if point[0]+1<H and qdist[point[0]+1, point[1]] == np.inf and obsmap[point[0]+1, point[1]] == 0 and  not visited[point[0]+1, point[1]]:
                new_queue.append([point[0]+1, point[1]])
            if point[0]-1>-1 and qdist[point[0]-1, point[1]] == np.inf and obsmap[point[0]-1, point[1]] == 0 and not visited[point[0]-1, point[1]]:
                new_queue.append([point[0]-1, point[1]])
            if point[1]+1<W and qdist[point[0], point[1]+1] == np.inf and obsmap[point[0], point[1]+1] == 0 and not visited[point[0], point[1]+1]:
                new_queue.append([point[0], point[1]+1])
            if point[1]-1>-1 and qdist[point[0], point[1]-1] == np.inf and obsmap[point[0], point[1]-1] == 0 and not visited[point[0], point[1]-1]:
                new_queue.append([point[0], point[1]-1])

        queue = new_queue
    if VIS_PLOT:
        plt.imshow(qdist[:,:])
        plt.colorbar()
        # plt.show()
    print(f'---------------------Finished computing holonomic_heuristic_with_obstacle .....----------------------')
    np.save("holonomic_heuristic_with_obstacle.npy", qdist)
    return qdist

def cal_non_holonomic_without_obstacles(obsmap, gx, gy, gyaw = np.deg2rad(60), yaw_reso = 3):
    print(f'---------------------Starting to compute non_holonomic_without_obstacles .....----------------------')
    find_ret = len(glob.glob("non_holonomic_yaw_traj.npy")) == 1 and len(glob.glob("non_holonomic_yaw_traj_length.npy")) == 1
    if find_ret:
        print(f'---------------------Finished loading non_holonomic_without_obstacles .....----------------------')
        return np.load("non_holonomic_yaw_traj.npy"), np.load("non_holonomic_yaw_traj_length.npy")
    H, W = obsmap.shape
    yaw_set = np.arange(0, 360, yaw_reso)
    yaw_set = np.deg2rad(yaw_set)
    yaw_set_len = yaw_set.shape[0]
    MAX_TRACK = 1000
    YAW_TRAJ = np.zeros([H, W, yaw_set_len, 4, MAX_TRACK])
    YAW_TRAJ_LENGTH = np.zeros([H, W, yaw_set_len])
    for x in range(H):
        for y in range(W):
            for yaw_idx, yaw in enumerate(yaw_set):
                x_traj, y_traj, v_traj, yaw_traj = non_holonomic_simulation(sx=x, sy=y, syaw=yaw, gx=gx, gy=gy, gyaw=gyaw, sv = min(10/(x+1e-6), 1.5), gv = min(10/(x+1e-6), 1.5))
                YAW_TRAJ[x, y, yaw_idx, 0, :len(x_traj)] = np.array(x_traj, dtype=np.float128)
                YAW_TRAJ[x, y, yaw_idx, 1, :len(y_traj)] = np.array(y_traj, dtype=np.float128)
                YAW_TRAJ[x, y, yaw_idx, 2, :len(y_traj)] = np.array(yaw_traj, dtype=np.float128)
                YAW_TRAJ[x, y, yaw_idx, 3, :len(y_traj)] = np.array(v_traj, dtype=np.float128)
                traj_x_diff = np.diff(np.array(x_traj, dtype=np.float128), axis=0)
                traj_x_2 = traj_x_diff * traj_x_diff
                traj_y_diff = np.diff(np.array(y_traj, dtype=np.float128), axis=0)
                traj_y_2 = traj_y_diff * traj_y_diff
                traj_x_y = np.sqrt(traj_x_2 + traj_y_2)
                total_length = np.sum(traj_x_y)
                # print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ The total path length is {total_length} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``')
                YAW_TRAJ_LENGTH[x, y, yaw_idx] = total_length

    np.save("non_holonomic_yaw_traj.npy", YAW_TRAJ)
    np.save("non_holonomic_yaw_traj_length.npy", YAW_TRAJ_LENGTH)

    print(f'---------------------Finished computing non_holonomic_without_obstacles .....----------------------')

    return YAW_TRAJ, YAW_TRAJ_LENGTH

def vis_non_holonomic_without_obstacles(obsmap, yaw_traj_length, yaw_reso):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 10}

    mpl.rc('font', **font)

    H, W = obsmap.shape
    # Make data.
    X = np.arange(0, H, 1)
    Y = np.arange(0, W, 1)
    X, Y = np.meshgrid(X, Y)

    Z = yaw_traj_length
    np.set_printoptions(threshold=200)
    print(f'the shape of Z is {Z.shape}')
    for i in range(Z.shape[2]):
        # Plot the surface.
        print(f'shape:{Z[...,i].shape}')
        surf = ax.plot_surface(X, Y, Z[...,i], cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        idx = np.where(Z[...,i]==np.max(Z[...,i]))
        print(f'the max index is idx {idx}')
    # Customize the z axis.
    ax.set_zlim(0, 200)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    ax.text2D(0.05, 0.95, "$Non-holonomic \; heuristics \; without \; obstacles \; with \; yaw \; variations$", transform=ax.transAxes)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # plt.show()

