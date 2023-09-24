
import matplotlib.pyplot as plt
from queue import PriorityQueue
from matplotlib.collections import LineCollection

from voronoi.dictionary import IndexDict
from voronoi.geometry import *
from collections import namedtuple

import numba as nb
import cv2
from threading import Thread
import glob
import scipy.spatial.kdtree as kd

class VoronoiField:
    

    def __init__(self, ret_val: namedtuple, alpha: np.ndarray, d_o_max: np.ndarray) -> None:
        
        self.__alpha = alpha
        self.__d_o_max = d_o_max
        self.__triangles = ret_val.triangles
        print(f'she shape of the self.__triangles are {self.__triangles.shape}')
        self.__boundaries = ret_val.boundaries
        print(f'she shape of the self.__boundaries are {self.__boundaries.shape}')
        self.__points = ret_val.points
        # Get rid of the staring point and the end point
        self.__points = self.__points[:-2]
        print(f'she shape of the self.__points are {self.__points.shape}')
        self.__points_polygon = ret_val.points_polygon
        print(f'she shape of the self.__points_polygon are {self.__points_polygon.shape}')
        self.__vor_vertices = ret_val.vertices
        print(f'she shape of the self.__vertices are {self.__vor_vertices.shape}')
        self.__ridge_vertices = ret_val.ridge_vertices
        print(f'she shape of the self.__ridge_vertices are {self.__ridge_vertices.shape}')
        self.__obs_kdtree = kd.KDTree([[x, y] for x, y in zip(self.__points[...,0], self.__points[...,1])])
        self.__ridges_kdtree = kd.KDTree([[x, y] for x, y in zip(self.__vor_vertices[...,0], self.__vor_vertices[...,1])])
        self.__inner_polygen_points = []


    def query(self, point:np.ndarray, last:np.ndarray, filter_inner_polygen_threshold = 0.013) -> float:
        flag = True
        # find the nearest obstacle point
        point = np.array(point) if not isinstance(point, np.ndarray) else point
        nearest_dist,  obs_idx = self.__obs_kdtree.query(point, k=1)
        d_o = distance(_1 = self.__points[obs_idx], _2 = point)
        assert np.allclose(0, nearest_dist - distance(_1 = self.__points[obs_idx], _2 = point))

        if d_o >= self.__d_o_max:
            return flag, 0, 0, 0
        
        # filter the inner points of every polygen
        for tri in self.__triangles:
            if tri.is_in_polygon(point):
                flag = False
                if d_o >= filter_inner_polygen_threshold:
                    self.__inner_polygen_points.append(point)

        # find two nearest ridge vertices
        nearest_dist, rid_idx = self.__ridges_kdtree.query(point, k = 2)
        assert [rid_idx[0], rid_idx[1]] in self.__ridge_vertices or [rid_idx[1], rid_idx[0]] in self.__ridge_vertices
        d_v = distance_between_line_point(np.take(self.__vor_vertices, rid_idx, axis = 0), point)
        p = np.array(((self.__alpha) / (self.__alpha + d_o + 1e-6)) * \
            ((d_v) / (d_v + d_o + 1e-6)) * \
            (pow(d_o - self.__d_o_max, 2)  / (pow(self.__d_o_max, 2) + 1e-6)))
        if np.isnan(p):
            p = np.array(0)
        print(f'do:{d_o}, d_v:{d_v},p:{p}')
        return flag, d_o, d_v, np.array(p)
        
    def run(self,resolution, start, end) -> list:

        print(f'---------------------Starting to compute Voronoi Potential Field .....----------------------')
        x_min, x_max, y_min, y_max, x_volume, y_volume = 0, 1, 0, 1, int(1 / resolution[0]), int(1 / resolution[1])
        self.x_volume = x_volume
        self.y_volume = y_volume

        find_ret = len(glob.glob("voronoi_potential_field.npy")) == 1
        if find_ret:
            print(f'---------------------Finished loading Voronoi Potential Field .....----------------------')
            self.__pot_field = np.load("voronoi_potential_field.npy")
            return self.__pot_field
        

        pot_field = np.zeros([x_volume, y_volume], dtype=np.float64)
        terrain_field = np.zeros([x_volume, y_volume], dtype=np.int64)
        # print(f'x_min:{x_min}, x_max:{x_volume}, y_min:{y_min}, y_max:{y_volume}')
        last = np.array(0)
        for idx, x in enumerate(np.arange(x_min, x_max, resolution[0])):
            for idy, y in enumerate(np.arange(y_min, y_max, resolution[1])):
                bar_free, d_o, d_v, res = self.query(point=np.array([x,y]), last=last)
                if not bar_free:
                    terrain_field[idx, idy] = 1
                pot_field[idx, idy] = res
                last = pot_field[idx, idy]
        pot_field = np.clip(pot_field, 0, 1)
        self.__pot_field = pot_field
        self.__terrain_field = terrain_field
        heatmap = self.__pot_field
        normalized_heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        normalized_heatmap = (normalized_heatmap * 255).astype(np.uint8)
        # print(normalized_heatmap)
        print(f'the shape of the normalized heatmap is {normalized_heatmap.shape} and the min is {np.min(normalized_heatmap)} and the maxis {np.max(normalized_heatmap)}')
        self.__pot_field = normalized_heatmap
        self.__inner_polygen_points = np.array(self.__inner_polygen_points)

        np.save("voronoi_potential_field.npy", self.__pot_field)
        print(f'---------------------Finished computing Voronoi Potential Field.....----------------------')

        return self.__pot_field

    def generate_plot(self) -> None:
        terrain = np.zeros([self.x_volume, self.y_volume, 3])
        for point in self.__inner_polygen_points:
            cv2.circle(terrain, center = (int((point[1]) * self.x_volume), int((1-point[0]) * self.y_volume)), radius = 1, color = (0, 0, 255), thickness=1, lineType=8, shift=0)
        terrain = terrain.astype(np.uint8)
        print(f'the input type of the np.tile si {np.tile(self.__pot_field[::-1,:, None], (1,1,3)).dtype} and another is {terrain.dtype}')
        dst = cv2.addWeighted(terrain, 1, np.tile((255-self.__pot_field)[::-1,:, None], (1,1,3)), 1, 0)
        cv2.imshow('sas', dst)
        cv2.imshow('ori', terrain)
        cv2.imshow("heatmap", np.tile((255-self.__pot_field)[::-1,:, None], (1,1,3)))
        cv2.waitKey(0)

    def terrain_on_resolution(self):
        print(f'----------------------Generaign the new terrain under the new RESOLUTION!----------------------------')
        np.save("resolution_map.npy", self.__terrain_field)
        cv2.imshow("heatmap", (255-255*self.__terrain_field).astype(np.uint8))
        cv2.waitKey(0)
        print(f'----------------------Finished generating new terrain under the new RESOLUTION!----------------------------')
        return self.__terrain_field


    def show(self) -> None:
        plt.show()