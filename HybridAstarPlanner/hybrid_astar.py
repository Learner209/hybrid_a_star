import os
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.kdtree as kd
from collections import namedtuple
from queue import PriorityQueue
from loguru import logger
import tqdm
import cv2
import dubins
from scipy.signal import savgol_filter

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")
from PIL import Image



# HybridAstarPlanner
from HybridAstarPlanner.traj_solver import TRAJSolver
from HybridAstarPlanner.astar import *

import CurvesGenerator.reeds_shepp as rs
from CurvesGenerator.quintic_polynomial import non_holonomic_simulation
from voronoi.geometry import *

# import utils
from utils.solver import make_lr_scheduler, make_optimizer
from utils.data_loader import make_data_loader
from utils.trainer import *
from utils.logger import setup_logger

from dl_ext.average_meter import AverageMeter
from dl_ext.pytorch_ext.dist import *
from dl_ext.pytorch_ext.optim import OneCycleScheduler


from torch import nn
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import torch
import torch.nn.functional as F

from mpl_toolkits.axes_grid1 import Grid
from scipy.special import comb



def find_index_of_nearest_xy(y_array, x_array, y_point, x_point):
    distance = (y_array-y_point)**2 + (x_array-x_point)**2
    idyx = np.where(distance==np.min(distance))
    return idyx[0], None if len(idyx)==1 else (idyx[0],idyx[1])

def softmax(x):
    return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())

class Path:
    def __init__(self, x, y, yaw, direction, cost):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.direction = direction
        self.cost = cost

class AstarPath:
    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.direction = []
        self.cost = []


class AstarPathPlanner():

    A_STAR_NDOE = namedtuple('Node', ['x', 'y', "g", "heuristics", "cost", "prev"])


    def __init__(self) -> None:
        pass

    def astar(self, sx, sy, gx, gy, obsmap):
        print(f"----------------------------Entering into the astar algorithm path planner !------------------------------------")
        H, W = obsmap.shape
        print(f'the H is {H} and the W is {W}')
        dist = np.full([H, W], np.inf)
        visited = np.full([H, W], False)

        open_list = PriorityQueue()
        ox, oy = np.where(obsmap==1)[0], np.where(obsmap==1)[1]
        closed_set = list()

        start_node_heuristics = math.sqrt((sx-gx)*(sx-gx) + (sy-gy)*(sy-gy))
        # start_node_heuristics = abs(sx-gx) + abs(sy-gy)

        start_node = AstarPathPlanner.A_STAR_NDOE(sx, sy, 0, start_node_heuristics, start_node_heuristics, None)
        open_list.put((0 + start_node_heuristics, start_node))

        expanded = []
        cnt = 0
        while True:

            curr_node = open_list.get()
            _, cur = curr_node
            if visited[cur.x, cur.y]:
                continue
            if cur.x == gx and cur.y  == gy:
                break
            assert obsmap[cur.x, cur.y] == 0
            visited[cur.x, cur.y] = True
            expanded.append([cur.x, cur.y])
            dist[cur.x, cur.y] = cur.g
            closed_set.append(cur)

            cnt += 1
            print(f'the new node into the regime is x:{cur.x}, y:{cur.y}, g:{cur.g} heuristics:{cur.heuristics} overall_cost:{cur.cost}')

            # if cnt == 50:
            #     break
            directions = [[0,1],[0,-1],[1,0],[-1,0]]
            directions = [[-1,0],[1,0],[0,-1],[0,1]]
            for i in range(len(directions)):
                ret, node_x, node_y, node_cost = self.find_next_node(cur, directions[i], H, W, obsmap)
                # print(f'the ret is {ret}, the node_x is {node_x}, the node_y is {node_y}, the node_cost is {node_cost}')
                if not ret:
                    continue
                else:
                    node_heuristics = math.sqrt((node_x-gx)*(node_x-gx) + (node_y-gy)*(node_y-gy))
                    # node_heuristics = abs(node_x-gx) + abs(node_y-gy)
                    new_node = AstarPathPlanner.A_STAR_NDOE(node_x, node_y, node_cost, node_heuristics, node_heuristics + node_cost, cur)
                    if not visited[node_x, node_y]:
                        if dist[node_x, node_y] == np.inf: # If haven't been put into the open_list
                            open_list.put((node_heuristics + node_cost, new_node))
                        else:
                            if dist[node_x, node_y] > node_cost: # If already existed in the open list
                                # continue
                                dist[node_x, node_y] = node_cost
                                open_list.put((node_heuristics + node_cost, new_node))

        fig, ax = plt.subplots()

        path = AstarPath()
        while(cur is not None):
            path.x.append(cur.x)
            path.y.append(cur.y)
            cur = cur.prev
        path.x = path.x[::-1]
        path.y = path.y[::-1]

        # self.extract_and_vis_path(cur, np.array(expanded), ox, oy, ax, None, None)
        return path.x, path.y
    

    def extract_and_vis_path(self, cur, expanded_list, ox, oy, ax, alpha, potential_field_weight):
        color_map = np.random.rand(expanded_list.shape[0], 3)
        color_map = np.sort(color_map, axis = 1)
        ax.scatter(np.array(expanded_list)[...,0], np.array(expanded_list)[...,1], c = color_map,  s=7.5)

        path = AstarPath()
        while(cur is not None):
            path.x.append(cur.x)
            path.y.append(cur.y)
            cur = cur.prev
        path.x = path.x[::-1]
        path.y = path.y[::-1]
        ax.scatter(path.x, path.y, color = "red", s=12)
        ax.scatter(ox,oy, color = "gray")
        if alpha is not None and potential_field_weight is not None:
            ax.set_title(f'$\\beta={{:.2f}},\\alpha={{:.2f}}$'.format(alpha, potential_field_weight))
        elif alpha is not None:
            ax.set_title(f'$\\beta={{:.2f}}$'.format(alpha))
        elif potential_field_weight is not None:
            ax.set_title(f'$\\alpha={{:.2f}}$'.format(potential_field_weight))
        else:
            pass
        
        # px, py = np.where(potential_field>=50)[0], np.where(potential_field>=50)[1]
        # ax.scatter(px,py, color = "pink")
        return path.x, path.y, path.yaw

    def find_next_node(self, cur, dire, H, W, obsmap):
        new_node_x = cur.x + dire[0]
        new_node_y = cur.y + dire[1]

        # Out of canvas
        if new_node_x <= 0 or new_node_x >= H-1 or new_node_y <= 0 or new_node_y >= W-1:
            return False, None, None, None

        # Too close to be obstacle points or in the obstacles themselves
        if obsmap[new_node_x, new_node_y]:
            return False, None, None, None
        
        new_cost = cur.g + 1
        return True, new_node_x, new_node_y, new_cost


class ImprovedAstarPathPlanner(AstarPathPlanner):

    IMPROVED_A_STAR_NDOE = namedtuple('Node', ['x', 'y', "yaw", "g", "heuristics", "cost", "prev"])

    def __init__(self) -> None:
        super().__init__()

    def set_voronoi_potential_field(self, value, H, W) -> None:
        self.potential_voronoi_field = value
        potential_voronoi_field = Image.fromarray((self.potential_voronoi_field))
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Resizing the Voronoi_potential field to resolution: {(H, W)}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        potential_voronoi_field = potential_voronoi_field.resize((H, W), Image.BILINEAR)
        self.potential_voronoi_field = np.array(potential_voronoi_field).T
        
        VIZ_POTENTIAL_FIELD = False
        if VIZ_POTENTIAL_FIELD:
            print(f'the shape of the self.potenttial_voronoi_field is {self.potential_voronoi_field.shape} max:{np.max(self.potential_voronoi_field)} min:{np.min(self.potential_voronoi_field)}')
            cv2.imshow("heatmap", np.tile((255-self.potential_voronoi_field)[::-1,:, None], (1,1,3)))
            cv2.waitKey(0)

        # print(f'the norm is {np.linalg.norm(self.potential_voronoi_field)}')
        # self.potential_voronoi_field /= np.linalg.norm(self.potential_voronoi_field) 


    def astar(self, sx, sy, syaw, gx, gy, gyaw, yaw_reso, obsmap, alpha = 1, VIZ_PLOT = False, **kwargs):
        print(f"----------------------------Entering into the astar algorithm path planner !------------------------------------")
        H, W = obsmap.shape

        potential_field_weight = kwargs.get('potential_field_weight')
        steering_penalty_weight = kwargs.get("steering_penalty_weight", 0)
        

        self.__holonomic_heuristic_with_obstacle = calc_holonomic_heuristic_with_obstacle(obsmap, gx, gy)
        self.__yaw_traj, self.__non_holonomic_heuristic_without_obstacle = cal_non_holonomic_without_obstacles(obsmap, gx, gy, gyaw, yaw_reso)
        # vis_non_holonomic_without_obstacles(obsmap, self.__non_holonomic_heuristic_without_obstacle, yaw_reso)
        self.yaw_reso = yaw_reso
        self.yaw_set = np.arange(0, 360, yaw_reso)
        self.yaw_set = np.deg2rad(self.yaw_set)

        print(f'the H is {H} and the W is {W}')
        print(f'the choseable yaw_set is {self.yaw_set}')
        dist = np.full([H, W], np.inf)
        visited = np.full([H, W], False)

        open_list = PriorityQueue()
        ox, oy = np.where(obsmap==1)[0], np.where(obsmap==1)[1]
        kdtree = kd.KDTree([[x, y] for x, y in zip(ox, oy)])
        closed_set = list()

        start_node_heuristics = math.sqrt((sx-gx)*(sx-gx) + (sy-gy)*(sy-gy))
        yaw_id, _ = find_index_of_nearest_xy(self.yaw_set, 0, syaw, 0)
        start_node_heuristics, _, _ = self.query_heuristics(sx, sy, yaw_id, alpha, 0, 0, potential_field_weight)
        start_node = ImprovedAstarPathPlanner.IMPROVED_A_STAR_NDOE(sx, sy, yaw_id, start_node_heuristics, 0, start_node_heuristics, None)
        open_list.put((0 + start_node_heuristics, start_node))

        expanded = []
        cnt = 0
        while True:

            curr_node = open_list.get()
            _, cur = curr_node
            if visited[math.floor(cur.x), math.floor(cur.y)]:
                continue
            if distance(np.array([cur.x, cur.y]), np.array([gx, gy])) < 9e-1:
                break
            assert obsmap[math.floor(cur.x), math.floor(cur.y)] == 0
            visited[math.floor(cur.x), math.floor(cur.y)] = True
            expanded.append([cur.x, cur.y])
            dist[math.floor(cur.x), math.floor(cur.y)] = cur.g
            closed_set.append(cur)

            cnt += 1
            print(f'the new node into the regime is x:{cur.x}, y:{cur.y}, g:{cur.g} heuristics:{cur.heuristics} overall_cost:{cur.cost}')

            # if cnt == 100000:
            #     break

            directions = [[math.cos(yaw), math.sin(yaw)] for yaw in self.yaw_set]
            for i in range(len(directions)):
                ret, node_x, node_y, node_cost, node_heuristics, id = self.find_next_node(cur, i, directions[i], H, W, obsmap, kdtree, alpha, steering_penalty_weight, potential_field_weight)
                # print(f'the ret is {ret}, the node_x is {node_x}, the node_y is {node_y}, the node_cost is {node_cost}')
                if not ret:
                    continue
                else:
                    new_node = ImprovedAstarPathPlanner.IMPROVED_A_STAR_NDOE(node_x, node_y, i, node_cost, node_heuristics, node_heuristics + node_cost, cur)
                    if not visited[id[0], id[1]]:
                        if dist[id[0], id[1]] == np.inf: # If haven't been put into the open_list
                            open_list.put((node_heuristics + node_cost, new_node))
                        else:
                            if dist[id[0], id[1]] > node_cost: # If already existed in the open list
                                dist[id[0], id[1]] = node_cost
                                open_list.put((node_heuristics + node_cost, new_node))

        print(f"----------------------------Finished the astar algorithm path planner !------------------------------------")
        del self.__holonomic_heuristic_with_obstacle, self.__yaw_traj, self.__non_holonomic_heuristic_without_obstacle
        if VIZ_PLOT:
            fig, ax = plt.subplots()
            x, y, yaw = self.extract_and_vis_path(cur, np.array(expanded), ox, oy, ax, None, None, None)
            plt.show()
            print(f'the shape of the {np.concatenate([np.array(x)[...,None], np.array(y)[...,None]], axis = 1).shape}')
            np.save("trajectory.npy", np.concatenate([np.array(x)[...,None], np.array(y)[...,None]], axis = 1))
            return np.concatenate([np.array(x)[...,None], np.array(y)[...,None]], axis = 1)
        return cur, np.array(expanded), ox, oy

    def extract_and_vis_path(self, cur, expanded_list, ox, oy, ax, alpha, potential_field_weight, steering_cost = None, VIS_BEZIER_CURVE = True, VIZ_EXPANDED = False):
        color_map = np.random.rand(expanded_list.shape[0], 3)
        color_map = np.sort(color_map, axis = 1)
        if VIZ_EXPANDED:
            ax.scatter(np.array(expanded_list)[...,0], np.array(expanded_list)[...,1], c = color_map,  s=7.5)

        path = AstarPath()
        while(cur is not None):
            path.x.append(cur.x)
            path.y.append(cur.y)
            cur = cur.prev
        path.x = path.x[::-1]
        path.y = path.y[::-1]
        init_traj = np.concatenate([np.array(path.x)[...,None], np.array(path.y)[...,None]], axis = 1)
        if VIS_BEZIER_CURVE:
            ax.plot(init_traj[...,0], init_traj[...,1], c='blue', linewidth = 2)
            yhat = savgol_filter(init_traj[...,1], 51, 3) # window size 51, polynomial order 3
            ax.plot(init_traj[...,0], yhat, c='pink', linewidth = 2)
            init_traj[...,1] = yhat
            # assert False
            xvals, yvals = bezier_curve(init_traj, nTimes=1000)
            init_traj = np.concatenate([xvals[..., None], yvals[...,None]], axis=1)
            ax.plot(init_traj[...,0], init_traj[...,1], c='red', linewidth = 2)
        else:
            ax.scatter(path.x, path.y, color = "red", s=12)

        ax.scatter(ox,oy, color = "gray")
        message = f''
        if alpha is not None:
            message += f' $\\beta={{:.2f}}$'.format(alpha)
        if potential_field_weight is not None:
            message += f' $\\gamma={{:.2f}}$'.format(potential_field_weight)
        if steering_cost is not None:
            message += f' $\\lambda={{:.2f}}$'.format(steering_cost)
        if len(message) > 0:
            ax.set_title(message)
        
        # px, py = np.where(potential_field>=50)[0], np.where(potential_field>=50)[1]
        # ax.scatter(px,py, color = "pink")
        return path.x, path.y, path.yaw

    def query_heuristics(self, new_node_x, new_node_y, yaw_id, alpha, steering_penalty_weight, steering_penalty_loss, voronoi_potential_field_weight):
        left_nearest = math.floor(new_node_x)
        right_nearest = math.floor(new_node_x) + 1 if (new_node_x - math.floor(new_node_x)) < 1e-2 else math.ceil(new_node_x)
        down_nearest = math.floor(new_node_y)
        up_nearest = math.floor(new_node_y) + 1 if (new_node_y - math.floor(new_node_y)) < 1e-2 else math.ceil(new_node_y)

        _0 = np.array([new_node_x,new_node_y])
        _1 = np.array([left_nearest, down_nearest])
        _2=  np.array([left_nearest, up_nearest])
        _3 = np.array([right_nearest, up_nearest])
        _4 = np.array([right_nearest, down_nearest])

        dist_1 = 1 / (distance(_0, _1) + 1e-6)
        dist_2 = 1 / (distance(_0, _2) + 1e-6)
        dist_3 = 1 / (distance(_0, _3) + 1e-6)
        dist_4 = 1 / (distance(_0, _4) + 1e-6)

        dist_weight = np.array([dist_1, dist_2, dist_3, dist_4])
        # dist_weight = np.linalg.norm(dist_weight)
        dist_weight = softmax(dist_weight)
        assert np.allclose(np.sum(dist_weight), 1)
        # print(f'the dist weight is {dist_weight}')
        
        if hasattr(self, "potential_voronoi_field"):
            voronoi_potential_field_weights = np.array([self.potential_voronoi_field[_1[0], _1[1]],
                                                        self.potential_voronoi_field[_2[0], _2[1]],
                                                        self.potential_voronoi_field[_3[0], _3[1]],
                                                        self.potential_voronoi_field[_4[0], _4[1]]]) * 5

        holonomic_heuristic_with_obstacle_weight = np.array([self.__holonomic_heuristic_with_obstacle[_1[0], _1[1]],
                                                             self.__holonomic_heuristic_with_obstacle[_2[0], _2[1]],
                                                             self.__holonomic_heuristic_with_obstacle[_3[0], _3[1]],
                                                             self.__holonomic_heuristic_with_obstacle[_4[0], _4[1]]])
        
        non_holonomic_without_obstacles_weight = np.array([self.__non_holonomic_heuristic_without_obstacle[_1[0], _1[1], yaw_id],
                                                             self.__non_holonomic_heuristic_without_obstacle[_2[0], _2[1], yaw_id],
                                                             self.__non_holonomic_heuristic_without_obstacle[_3[0], _3[1], yaw_id],
                                                             self.__non_holonomic_heuristic_without_obstacle[_4[0], _4[1], yaw_id]])

        holonomic_heuristic_with_obstacle_interpolation = np.sum(dist_weight * holonomic_heuristic_with_obstacle_weight)
        non_holonomic_without_obstacles_interpolation = np.sum(dist_weight * non_holonomic_without_obstacles_weight)
        if hasattr(self, "potential_voronoi_field"):
            voronoi_potential_field_interpolation = np.sum(dist_weight * voronoi_potential_field_weights)

        new_heuristics = (alpha) * non_holonomic_without_obstacles_interpolation + (1-alpha) * holonomic_heuristic_with_obstacle_interpolation
        new_heuristics += steering_penalty_loss * steering_penalty_weight
        if voronoi_potential_field_weight is not None:
            new_heuristics = (1-voronoi_potential_field_weight) * new_heuristics + voronoi_potential_field_weight * voronoi_potential_field_interpolation
        return new_heuristics, holonomic_heuristic_with_obstacle_interpolation, non_holonomic_without_obstacles_interpolation
    
    def get_steering_penalty(self, yaw_reso, yaw_id_1, yaw_id_2):
        if not hasattr(self, "yaw_penalty_corr"):
            yaw_set = np.arange(0, 360, yaw_reso)
            yaw_set = np.deg2rad(yaw_set)
            yaw_penalty_corr = np.identity(len(yaw_set))
            for yaw_id_1 in range(len(yaw_set)):
                for yaw_id_2 in range(len(yaw_set)):
                    vec_1 = np.array([math.cos(yaw_set[yaw_id_1]), math.sin(yaw_set[yaw_id_1])])
                    vec_2 = np.array([math.cos(yaw_set[yaw_id_2]), math.sin(yaw_set[yaw_id_2])])
                    yaw_penalty_corr[yaw_id_1, yaw_id_2] = inner(vec_1, vec_2)
            print(f'the yaw_penalty_corr is {yaw_penalty_corr}')
            yaw_penalty_corr = np.power(50 *(-yaw_penalty_corr+1), 1.25)
            print(f'the softmaxed yaw_penalty_corr is {yaw_penalty_corr}')
            self.yaw_penalty_corr = yaw_penalty_corr
        return self.yaw_penalty_corr[yaw_id_1, yaw_id_2]

    def find_next_node(self, cur, yaw_id, dire, H, W, obsmap, kdtree, alpha, steering_penalty_weight, potential_field_weight):

        new_node_x = cur.x + dire[0]
        new_node_y = cur.y + dire[1]

        # Out of canvas
        if new_node_x <= 0 or new_node_x >= H-1 or new_node_y <= 0 or new_node_y >= W-1:
            return False, None, None, None, None, None
        nearest_dist, obs_idx = kdtree.query([new_node_x, new_node_y], k=1)

        left_nearest = math.floor(new_node_x)
        down_nearest = math.floor(new_node_y)

        # Too close to be obstacle points or in the obstacles themselves
        if nearest_dist < 2 or obsmap[left_nearest, down_nearest]:
            return False, None, None, None, None, None

        steering_penalty_loss = self.get_steering_penalty(self.yaw_reso, yaw_id, cur.yaw)

        new_heuristics, _, _ = self.query_heuristics(new_node_x, new_node_y, yaw_id, alpha, steering_penalty_weight, steering_penalty_loss, potential_field_weight)

        new_cost = cur.g + math.sqrt(dire[0] * dire[0] + dire[1] * dire[1])

        return True, new_node_x, new_node_y, new_cost, new_heuristics, [left_nearest, down_nearest]


    def reproduce_heuristics_relation(self):
        sx = 5
        sy = 5
        gx = 110
        gy = 110
        hor = 2
        ver = 5
        fig = plt.figure()
        grid = Grid(fig, rect=111, nrows_ncols=(2, 5),
                axes_pad=0.2, label_mode='L',
                )
        # fig, axs = plt.subplots(hor, ver, figsize=(8,8))
        alpha = np.arange(0, 1.0, 0.1)
        cnt = 0
        for i in range(hor):
            for j in range(ver):
                idx = i * ver + j
                print(f'we have been trying the {i} {j}')
                cur, expanded, ox, oy = self.astar(sx = sx, sy = sy, syaw = np.deg2rad(90), gx = gx, gy = gy, gyaw=np.deg2rad(120), yaw_reso=5, obsmap = np.load('map.npy'), alpha = alpha[idx])
                self.extract_and_vis_path(cur, expanded, ox, oy, grid[cnt], alpha[idx], None)
                # grid[cnt].set_xlim(0,1)
                # grid[cnt].set_ylim(0,1)
                cnt += 1
        plt.tight_layout()
        plt.show()
        
    def reproduce_potential_voronoi_field_variation(self):
        sx = 5
        sy = 5
        gx = 110
        gy = 110
        obsmap = np.load('map.npy')
        potential_voronoi_field = np.load("voronoi_potential_field.npy")
        self.set_voronoi_potential_field(potential_voronoi_field, obsmap.shape[0], obsmap.shape[1])
        potential_voronoi_weights = np.arange(0, 1, 0.1)
        hor = 2
        ver = 5
        fig = plt.figure()
        grid = Grid(fig, rect=111, nrows_ncols=(2, 5),
                axes_pad=0.3, label_mode='L',
                )
        cnt = 0
        alpha = 0.5
        potential_voronoi_field = np.arange(0, 1, 0.1)
        for i in range(hor):
            for j in range(ver):
                idx = i * ver + j
                kwargs = {
                    "integrate_voronoi_field": True,
                    "potential_field_weight": potential_voronoi_weights[idx],
                }
                cur, expanded, ox, oy = self.astar(sx = sx, sy = sy, syaw = np.deg2rad(90), gx = gx, gy = gy, gyaw=np.deg2rad(120), yaw_reso=30, obsmap = np.load('map.npy'), alpha = alpha, **kwargs)
                self.extract_and_vis_path(cur, expanded, ox, oy, grid[cnt], alpha, potential_voronoi_weights[idx])
                cnt += 1
        plt.tight_layout()
        plt.show()
        
    def reproduce_steering_cost(self):
        sx = 5
        sy = 5
        gx = 110
        gy = 110
        obsmap = np.load('map.npy')
        potential_voronoi_field = np.load("voronoi_potential_field.npy")
        self.set_voronoi_potential_field(potential_voronoi_field, obsmap.shape[0], obsmap.shape[1])
        potential_voronoi_weights = np.arange(0, 0, 0.1)
        hor = 2
        ver = 5
        fig = plt.figure()
        grid = Grid(fig, rect=111, nrows_ncols=(2, 5),
                axes_pad=0.3, label_mode='L',
                )
        cnt = 0
        alpha = 0.4
        steering_cost = np.arange(0, 1, 0.1)
        for i in range(hor):
            for j in range(ver):
                idx = i * ver + j
                kwargs = {
                    "potential_field_weight": 0.6,
                    "steering_penalty_weight":steering_cost[idx]
                }
                cur, expanded, ox, oy = self.astar(sx = sx, sy = sy, syaw = np.deg2rad(90), gx = gx, gy = gy, gyaw=np.deg2rad(120), yaw_reso=30, obsmap = np.load('map.npy'), alpha = alpha, **kwargs)
                self.extract_and_vis_path(cur, expanded, ox, oy, grid[cnt], alpha, kwargs["potential_field_weight"], steering_cost[idx])
                cnt += 1
                # plt.show()
                # assert False
        plt.tight_layout()
        plt.show()
       
    
class HybridAstarPathPlanner(ImprovedAstarPathPlanner):
    
    IMPROVED_A_STAR_NDOE = namedtuple('Node', ['x', 'y', "yaw", "g", "heuristics", "cost", "prev"])

    distance_trash = 0.002

    def __init__(self, ret_val: namedtuple, obsmap, alpha, d_o_max):
        
        self.__alpha = torch.as_tensor(alpha)
        self.__d_o_max = torch.as_tensor(d_o_max)
        self.__triangles = torch.zeros(ret_val.triangles.shape[0], 3, 2)
        for idx in range(ret_val.triangles.shape[0]):
            tri_vertices = ret_val.triangles[idx].points
            self.__triangles[idx] = torch.from_numpy(tri_vertices)
        self.__points = torch.from_numpy(ret_val.points).float()
        # Get rid of the staring point and the end point
        self.__points = self.__points[:-2,...]
        self.__points_polygon = torch.from_numpy(ret_val.points_polygon).float()
        self.__vor_vertices = torch.from_numpy(ret_val.vertices)
        self.__ridge_vertices = torch.from_numpy(ret_val.ridge_vertices)
        self.__obsmap = obsmap
        self.H, self.W = self.__obsmap.shape
        self.__ridges_kdtree = kd.KDTree([[x, y] for x, y in zip(self.__vor_vertices[...,0], self.__vor_vertices[...,1])])
        self.__obs_kdtree = kd.KDTree([[x, y] for x, y in zip(self.__points[...,0], self.__points[...,1])])

    def voronoi_potential_field_loss(self, point:torch.Tensor, last:torch.Tensor) -> float:
        flag = True
        # find the nearest obstacle point
        assert point[0] <= 1 and point[1] <= 1 and point[0] >= 0 and point[1] >= 0, f'the point is {point}'
        batch_d_o = torch.cdist(self.__points[None].to(point.device), point[None], p=2.0)[0].double()
        d_o = torch.min(batch_d_o)
        # print(f'the d_o is {d_o}')

        if d_o >= self.__d_o_max:
            return flag, 0, 0, torch.tensor(0)
        
        if is_in_polygon(self.__triangles, point):
            flag = False

        # find two nearest ridge vertices
        nearest_dist, rid_idx = self.__ridges_kdtree.query(point.data.cpu().numpy(), k = 2)

        # batch_d_v = torch.cdist(self.__vor_vertices[None].to(point.device), point[None], p=2.0)[0].double()
        # print(f'the nearrest distance: {nearest_dist}')
        d_v = distance_between_line_point(torch.index_select(self.__vor_vertices.to(point.device), 0, torch.from_numpy(rid_idx).to(point.device)), point)
        # print(f'the d_v is {d_v}')

        p = ((self.__alpha) / (self.__alpha + d_o + 1e-6)) * \
            ((d_v) / (d_v + d_o + 1e-6)) * \
            (torch.pow(d_o - self.__d_o_max, 2)  / (torch.pow(self.__d_o_max, 2) + 1e-6))
        assert isinstance(p, torch.Tensor)
        if torch.isnan(p):
            p = torch.tensor(0)

        # print(f'do:{d_o.data.cpu().numpy()}, d_v:{d_v.data.cpu().numpy()},p:{p.data.cpu().numpy()}')
        return flag, d_o, d_v, p * 1
    
    def obstacle_collision_field_loss(self, point:torch.Tensor) -> float:
        # find the nearest obstacle point
        batch_d_o = torch.cdist(self.__points[None].to(point.device), point[None], p=2.0)[0].double()
        d_o = torch.min(batch_d_o)
        # print(f'do:{d_o.data.cpu().numpy()}, d_o_max:{self.__d_o_max}')
        return torch.pow(abs(d_o - self.__d_o_max), 2) * 100

    def curvature_constrain_loss(self, traj:torch.Tensor, index: int, curvature_max:torch.Tensor) -> float:
        N, _ = traj.shape
        if index == 0 or index == N-1:
            return torch.tensor(0)
        diff = torch.diff(traj, n=1, dim=0).double()

        epsilon = 1e-7
        delta_phi_i_1 =  torch.atan2(diff[index, 1], diff[index, 0] + epsilon if diff[index, 0] == 0 else diff[index, 0])
        delta_phi_i =  torch.atan2(diff[index-1, 1], diff[index-1, 0] + epsilon if diff[index-1, 0] == 0 else diff[index-1, 0])

        delta_phi = torch.abs(delta_phi_i_1 - delta_phi_i)
        delta_xi = torch.norm(diff[index-1, ...])
        # print(f'delta_phi_i_1:{delta_phi_i_1}, delta_phi_i:{delta_phi_i}, delta_xi:{delta_xi}')
        return torch.pow(abs(delta_phi / delta_xi - curvature_max), 2) * 100
    
    def traj_balance_loss(self, traj:torch.Tensor, index: int) -> float:
        N, _ = traj.shape
        if index == 0 or index == N-1:
            return torch.tensor(0)
        diff = torch.diff(traj, n=1, dim=0)
        delta_xi = torch.norm(diff[index-1, ...])
        delta_xi_1 = torch.norm(diff[index, ...])
        return (delta_xi - delta_xi_1) * (delta_xi - delta_xi_1) * 100

    def steering_loss(self, traj:torch.Tensor, index: int, curvature_gauge: float) -> float:
        """
        Add steering loss as a term of regularization
        """
        N, _ = traj.shape
        if index == 0 or index == N-1:
            return torch.tensor(0)
        diff = torch.diff(traj, n=1, dim=0).double()

        epsilon = 1e-7
        delta_phi_i_1 =  torch.atan2(diff[index, 1], diff[index, 0] + epsilon if diff[index, 0] == 0 else diff[index, 0])
        delta_phi_i =  torch.atan2(diff[index-1, 1], diff[index-1, 0] + epsilon if diff[index-1, 0] == 0 else diff[index-1, 0])
        delta_phi = torch.abs(delta_phi_i_1 - delta_phi_i)

        return torch.pow(torch.abs(delta_phi - curvature_gauge), 2) * 150

    def run(self, traj, resolution) -> list:
        traj = torch.from_numpy(np.load("trajectory.npy")).float()
        plt.plot(traj[...,0].data.cpu().numpy(), traj[...,1].data.cpu().numpy())
        plt.show()
        print(f'the shape of the traj is {traj.shape}')
        N, _ = traj.shape
        for i in range(N):
            print(f'the self.traj_balance_loss is {self.traj_balance_loss(traj, index = i)}')
            print(f'the self.curvature_constrain_loss is {self.curvature_constrain_loss(traj, index = i, curvature_max=0.04)}')
            pass
        
        pot_field = torch.zeros([self.H, self.W], dtype=torch.float64)
        # print(f'x_min:{x_min}, x_max:{x_volume}, y_min:{y_min}, y_max:{y_volume}')
        x_min, x_max, y_min, y_max, x_volume, y_volume = 0, 1, 0, 1, int(1 / resolution[0]), int(1 / resolution[1])
        self.x_volume = x_volume
        self.y_volume = y_volume
        pot_field = torch.zeros([x_volume, y_volume], dtype=torch.float64)
        terrain_field = torch.zeros([x_volume, y_volume], dtype=torch.float64)
        # print(f'x_min:{x_min}, x_max:{x_volume}, y_min:{y_min}, y_max:{y_volume}')
        last = np.array(0)
        cnt = 0
        for idx, x in enumerate(torch.arange(x_min, x_max, resolution[0])):
            for idy, y in enumerate(torch.arange(y_min, y_max, resolution[1])):
                bar_free, d_o, d_v, res = self.voronoi_potential_field_loss(torch.as_tensor([x,y]).float(), last)
                obs_loss = self.obstacle_collision_field_loss(torch.as_tensor([x,y]).float())
                print(f'the obs loss is {obs_loss}')
                pot_field[idx, idy] = res
                last = pot_field[idx, idy]
                cnt += 1
                # if cnt > 10:
                #     assert False

        pot_field = torch.clamp(pot_field, min=0, max=1)
        self.__pot_field = pot_field
        heatmap = self.__pot_field
        print(f'the pot_field is {heatmap}')
        normalized_heatmap = (heatmap - torch.min(heatmap)) / (torch.max(heatmap) - torch.min(heatmap))
        normalized_heatmap = (normalized_heatmap.data.cpu().numpy() * 255).astype(np.uint8)
        # print(normalized_heatmap)
        print(f'the shape of the normalized heatmap is {normalized_heatmap.shape} and the min is {np.min(normalized_heatmap)} and the maxis {np.max(normalized_heatmap)}')
        self.__pot_field = normalized_heatmap

        terrain = np.zeros([self.x_volume, self.y_volume, 3])
        terrain = terrain.astype(np.uint8)
        print(f'the input type of the np.tile si {np.tile(self.__pot_field[::-1,:, None], (1,1,3)).dtype} and another is {terrain.dtype}')
        dst = cv2.addWeighted(terrain, 1, np.tile((255-self.__pot_field)[::-1,:, None], (1,1,3)), 1, 0)
        cv2.imshow('sas', dst)
        cv2.imshow('ori', terrain)
        cv2.imshow("heatmap", np.tile((255-self.__pot_field)[::-1,:, None], (1,1,3)))
        cv2.waitKey(0)

        print(f'---------------------Finished computing Voronoi Potential Field.....----------------------')
        return self.__pot_field

    def fit(self):
        os.makedirs(self.output_dir, exist_ok=True)
        num_epochs = self.num_epochs
        begin = time.time()
        for epoch in range(self.begin_epoch, num_epochs):
            self.train(epoch)
            synchronize()
        if is_main_process():
            logger.info('Training finished. Total time %s' % (format_time(time.time() - begin)))

    def main(self, cfg):

        # Alpha = 1 means using the non-holomomic-without_obstacles 
        # Alpha = 0 mean using the holomomic-with_obstacles 
        sx = cfg.sx
        sy = cfg.sy
        syaw = np.deg2rad(cfg.syaw)
        gx = cfg.gx
        gy = cfg.gy
        gyaw = np.deg2rad(cfg.gyaw)
        yaw_reso = cfg.yaw_reso
        obsmap = np.load(cfg.obsmap)
        alpha = cfg.alpha
        kw = {"potential_field_weight": cfg.potential_field_weight,"steering_penalty_weight":cfg.steering_penalty_weight}

        cur, _, ox, oy = self.astar(sx, sy, syaw, gx, gy, gyaw, yaw_reso, obsmap, alpha, **kw)
        path = AstarPath()
        while(cur is not None):
            path.x.append(cur.x)
            path.y.append(cur.y)
            cur = cur.prev
        path.x = path.x[::-1]
        path.y = path.y[::-1]
        init_traj = np.concatenate([np.array(path.x)[...,None],np.array(path.y)[...,None]], axis =1)

        # init_traj = np.load("trajectory.npy")
        # np.save('trajectory.npy', init_traj)
        print(f'the shape of the init_traj is {init_traj.shape}')
        yhat = savgol_filter(init_traj[...,1], 51, 3) # window size 51, polynomial order 3
        init_traj[...,1] = yhat

        plt.plot(init_traj[...,0], init_traj[...,1], c='green', linewidth = 2)
        # assert False
        self.ox, self.oy = np.where(obsmap==1)[0], np.where(obsmap==1)[1]
        xvals, yvals = bezier_curve(init_traj, nTimes=1000)
        init_rs_traj = self.cal_reeds_shepp_optimal_path(init_traj, cfg, 1)
        # plt.plot(init_rs_traj[...,0], init_rs_traj[...,1], c='red', linewidth = 2)
        init_rs_traj = np.concatenate([xvals[..., None], yvals[...,None]], axis=1)
        # plt.plot(init_rs_traj[...,0], init_rs_traj[...,1], c='orange', linewidth = 2)
        # plt.scatter(ox, oy, c = 'gray')
        # plt.show()    
        # assert False
        torch.random.manual_seed(2023)
        cfg.mode = 'train'
        cfg.freeze()
        output_dir = cfg.output_dir
        os.makedirs(output_dir, exist_ok=True)

        logger = setup_logger(output_dir)
        logger.info("Running with config:\n{}".format(cfg))

        self.cfg = cfg
        self.init_traj = init_traj
        data = torch.from_numpy(init_traj).float()

        logger.info("~~~~~~~~~~~~~~~~~~~~~Creating the model...~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        self.model: nn.Module = TRAJSolver(cfg.solver, self, data, 1, self.H, self.W).to(torch.device(self.cfg.model.device))
        self.train_dl = make_data_loader(self.cfg, is_train=True)
        logger.info("~~~~~~~~~~~~~~~~~~~~~Training dataset generated~~~~~~~~~~~~~~~~~~~~~~~~")
        self.optimizer = make_optimizer(self.cfg, self.model)
        logger.info("~~~~~~~~~~~~~~~~~~~~~Optimizer generated~~~~~~~~~~~~~~~~~~~~~~~~")
        self.scheduler = make_lr_scheduler(self.cfg, self.optimizer,
                                           self.cfg.solver.num_epochs * len(self.train_dl))
        logger.info("~~~~~~~~~~~~~~~~~~~~~Scheduler generated~~~~~~~~~~~~~~~~~~~~~~~~")
        
        self.output_dir = cfg.output_dir
        self.num_epochs = cfg.solver.num_epochs
        self.begin_epoch = 0
        self.max_lr = cfg.solver.max_lr
        self.epoch_time_am = AverageMeter()

        self.fit()

    def train(self, epoch):
        loss_meter = AverageMeter()
        self.model.train()
        metric_ams = {}
        bar = tqdm.tqdm(self.train_dl, leave=False) if is_main_process() and len(self.train_dl) > 1 else self.train_dl
        begin = time.time()
        for batchid, batch in enumerate(bar):
            self.optimizer.zero_grad()
            batch = to_cuda(batch)
            output, loss_dict = self.model(batch)
            loss = sum(v for k, v in loss_dict.items())
            loss.backward()
            if self.cfg.solver.do_grad_clip:
                if self.cfg.solver.grad_clip_type == 'norm':
                    clip_grad_norm_(self.model.parameters(), self.cfg.solver.grad_clip)
                else:
                    clip_grad_value_(self.model.parameters(), self.cfg.solver.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None and isinstance(self.scheduler, OneCycleScheduler):
                self.scheduler.step()
            reduced_loss = reduce_loss(loss)
            metrics = {}
            for k, v in output.items():
                reduced_s = reduce_loss(v)
                metrics[k] = reduced_s
            if is_main_process():
                loss_meter.update(reduced_loss.item())
                lr = self.optimizer.param_groups[0]['lr']

                bar_vals = {'epoch': epoch, 'phase': 'train', 'loss': loss_meter.avg, 'lr': lr}

                if isinstance(bar, tqdm.tqdm):
                    bar.set_postfix(bar_vals)

        torch.cuda.synchronize()
        epoch_time = format_time(time.time() - begin)
        if is_main_process() and epoch % self.cfg.solver.log_interval == 0:
            loss_msgs = ['traj_balance_loss: %.4f, curvature_constrain_loss: %.4f, voronoi_potential_field_loss: %.4f, obstacle_collision_field_loss: %.4f steering_loss:%.4f' % (
                output['traj_balance_loss'], output['curvature_constrain_loss'], output['voronoi_potential_field_loss'], output['obstacle_collision_field_loss'], output['steering_loss'])]
            logger.info(loss_msgs)
            metric_msgs = ['epoch %d, train, loss %.4f, time %s' % (
                epoch, loss_meter.avg, epoch_time)]
            s = ', '.join(metric_msgs)
            logger.info(s)
        if is_main_process() and epoch % self.cfg.solver.show_interval == 0:
            identifier = "traj_2"
            if not os.path.exists(f"traj_solver/{identifier}"):
                os.makedirs(f"traj_solver/{identifier}", exist_ok=True)
            plt.plot(self.init_traj[...,0], self.init_traj[...,1],c = 'green', linewidth=2)
            plt.plot(self.model.traj[...,0].data.cpu().numpy(), self.model.traj[...,1].data.cpu().numpy(), c = 'red', linewidth=2)
            plt.scatter(self.ox, self.oy, c = 'gray')
            len_fig = len(glob.glob(f"traj_solver/{identifier}/*.png"))
            plt.savefig(f"traj_solver/{identifier}/{len_fig}.png")
            # plt.show()

            plt.close('all')
        if self.scheduler is not None and not isinstance(self.scheduler, OneCycleScheduler):
            self.scheduler.step()
        return metric_ams

    def is_collision_rs_path(self, rspath):
        for point in zip(rspath.x, rspath.y):
            point = torch.tensor(point).double()
            point = point / torch.tensor([self.H, self.W])
            # find the nearest obstacle point
            # assert point[0] <= 1 and point[1] <= 1 and point[0] >= 0 and point[1] >= 0, f'the point is {point}'
            batch_d_o = torch.cdist(self.__points[None].to(point.device), point[None], p=2.0)[0].double()
            d_o = torch.min(batch_d_o)
            if d_o >= self.__d_o_max:
                continue
            
            if is_in_polygon(self.__triangles, point):
                print(f'the proposed path have been found to be contained in a triangle!')
                return True
            
        return False

    def calc_reeds_shepp_path_cost(self, rspath, cfg):
        # if self.is_collision_rs_path(rspath):
        #     return 100000
        
        cost = 0.0

        for lr in rspath.lengths:
            if lr >= 0:
                cost += 1
            else:
                cost += 10000

        for i in range(len(rspath.lengths) - 1):
            if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:
                cost += cfg.reeds_shepp.gear_cost

        for ctype in rspath.ctypes:
            if ctype != "S":
                cost += cfg.reeds_shepp.steer_angle_cost* abs(cfg.reeds_shepp.max_curvature)

        nctypes = len(rspath.ctypes)
        ulist = [0.0 for _ in range(nctypes)]

        for i in range(nctypes):
            if rspath.ctypes[i] == "R":
                ulist[i] = -cfg.reeds_shepp.max_curvature
            elif rspath.ctypes[i] == "WB":
                ulist[i] = cfg.reeds_shepp.max_curvature

        for i in range(nctypes - 1):
            cost += cfg.reeds_shepp.steer_change_cost * abs(ulist[i + 1] - ulist[i])

        return cost

    def cal_reeds_shepp_optimal_path(self, init_traj, cfg, stop_sign):
        if stop_sign == 0:
            return init_traj
        sampled_init_traj = init_traj[::cfg.reeds_shepp.sample_rate, ...]
        sampled_init_traj = np.concatenate([sampled_init_traj, init_traj[-1][None,...]], axis=0)
        sampled_init_traj_diff = np.diff(sampled_init_traj, n=1, axis=0)
        epsilon = 1e-7


        print(f'the smapled_init_traj is {sampled_init_traj.shape}')
        seq_len, _ = sampled_init_traj.shape
        opt_path_x, opt_path_y = [], []

        for idx in range(seq_len-2):
            seg_syaw = np.arctan2(sampled_init_traj_diff[idx, 1], sampled_init_traj_diff[idx, 0] + epsilon if sampled_init_traj_diff[idx, 0] == 0 else sampled_init_traj_diff[idx, 0])
            seg_gyaw = np.arctan2(sampled_init_traj_diff[idx+1, 1], sampled_init_traj_diff[idx+1, 0] + epsilon if sampled_init_traj_diff[idx+1, 0] == 0 else sampled_init_traj_diff[idx+1, 0])
            print(f'the starting yaw is {seg_syaw} and the ending yaw is {seg_gyaw}')

            q0 = (sampled_init_traj[idx][0], sampled_init_traj[idx][1], seg_syaw)
            q1 = (sampled_init_traj[idx+1][0], sampled_init_traj[idx+1][1], seg_syaw)
            turning_radius = 0.5
            step_size = 0.01

            path = dubins.shortest_path(q0, q1, turning_radius)
            configurations, _ = path.sample_many(step_size)
            segment_path_x = [x for (x, y, theta) in configurations]
            segment_path_y = [y for (x, y, theta) in configurations]
            opt_path_x += segment_path_x
            opt_path_y += segment_path_y

        opt_path = np.concatenate([np.array(opt_path_x)[...,None], np.array(opt_path_y)[...,None]], axis=1)
        return self.cal_reeds_shepp_optimal_path(opt_path, cfg, stop_sign - 1)

    def cal_reeds_shepp_segment_paths(self, sx, sy, syaw, gx, gy, gyaw, max_curvature, step_size):
        q0 = [sx, sy, syaw]
        q1 = [gx, gy, gyaw]

        paths = rs.generate_path(q0, q1, max_curvature)

        for path in paths:
            x, y, yaw, directions = \
                rs.generate_local_course(path.L, path.lengths,
                                    path.ctypes, max_curvature, step_size * max_curvature)

            # convert from local coordinate to global coordinate
            path.x = [math.cos(-q0[2]) * ix + math.sin(-q0[2]) * iy + q0[0] for (ix, iy) in zip(x, y)]
            path.y = [-math.sin(-q0[2]) * ix + math.cos(-q0[2]) * iy + q0[1] for (ix, iy) in zip(x, y)]
            path.yaw = [rs.pi_2_pi(iyaw + q0[2]) for iyaw in yaw]
            path.directions = directions
            path.lengths = [l / max_curvature for l in path.lengths]
            path.L = path.L / max_curvature

        return paths


def distance_between_line_point(line:torch.Tensor, point:torch.Tensor):
    _1 = line[1] - line[0]
    _2 = point - line[0]
    _1_l = torch.norm(_1)
    _2_l = torch.norm(_2)
    _d = torch.dot(_1, _2)
    if _d > _1_l * _1_l or _d < 0.0:
        return min(_2_l, torch.norm(point - line[1]))
    
    _c = torch.acos(_d / (_1_l * _2_l + 1e-6))
    return torch.sin(_c) * _2_l

# test if point is in triangle
def is_in_polygon(batch_tri_vertices, point, distance_trash = 0.002) -> bool:
    # if point is close enough, filter out
    batch_size, _, _ = batch_tri_vertices.shape
    for i in range(batch_size):
        if test_distance_trash(batch_tri_vertices[i].to(point.device), point, distance_trash):
            return True
        if test_point_convex(batch_tri_vertices[i].to(point.device), point):
            return True
    return False

# test if a point is close enough to vertex
# test value is _distance_trash
def test_distance_trash(points, test_point, distance_trash) -> bool:
    dis = torch.cdist(test_point.unsqueeze(0), points.unsqueeze(0))[0]
    return torch.any(dis <= distance_trash)


def cross_2d(point_1, point_2):
    s = torch.cat([point_1[None], point_2[None]], dim=0)
    s_pad = F.pad(s, (0, 1))
    # compute the cross product 
    A = torch.cross(s_pad[0], s_pad[1], dim=0)
    # use the last dim which is the same as if the cross product would be done in 2D
    A = A[..., 2]
    return A


def bernstein_poly(i, n, t):
    """
    The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def test_point_convex(points, test_point) -> bool:
    for i in range(3):
        # vector ab
        l_1 = points[(i+1)%3] - points[i]
        # vector ac
        l_2 = points[(i+2)%3] - points[i]
        # vector ap
        l_3 = test_point - points[i]

        # ab x ap
        c_1 = cross_2d(l_1, l_3)
        # ap x ac
        c_2 = cross_2d(l_3, l_2)

        # if (ab x ap) * (ap x ac) < 0 than the point is in convex
        if c_1 * c_2 <= 0.0:
            return False
    return True

if __name__ == "__main__":
    import numpy as np
    from scipy.special import comb

    def bernstein_poly(i, n, t):
        """
        The Bernstein polynomial of n, i as a function of t
        """

        return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


    def bezier_curve(points, nTimes=1000):

        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])

        t = np.linspace(0.0, 1.0, nTimes)

        polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)

        return xvals, yvals


    from matplotlib import pyplot as plt

    nPoints = 4
    points = np.random.rand(nPoints,2)*200
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    points = np.load('trajectory.npy')
    xvals, yvals = bezier_curve(points, nTimes=1000)
    plt.plot(xvals, yvals)

    plt.show()