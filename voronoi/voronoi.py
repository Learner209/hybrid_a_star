"""
***************************************************************************************/
*    Title: Voronoi Generalized Diagram Implementation (Generate Voronoi field)
*    Author: ross 1573
*    Date: 2022
*    Availability: https://github.com/ross1573/generalized_voronoi_diagram.git
*
***************************************************************************************/
"""

import numpy as np
import matplotlib.pyplot as plt
from rdp import rdp
from enum import Enum
from collections import deque
from scipy.spatial import Voronoi, voronoi_plot_2d

from voronoi.dictionary import IndexDict
from voronoi.geometry import *
import copy
from collections import namedtuple
import cv2


class run_type(Enum):
    non_lined = 0,
    non_deleted = 1,
    non_optimized = 2,
    optimized = 3
    

# polygon based voronoi diagram
class GeneralizedVoronoi:
    rdp_epsilon: float

    def __init__(self) -> None:
        self.__points = []
        self.__lines = []
        self.__triangles = []
        self.__boundaries = []

        self.__triangle_points = []
        self.__boudnary_points = []
        self.__line_points = []

        self.__triangle_lined_points = []
        self.__boundary_lined_points = []
        self.__line_lined_points = []

        self.__chains = []
        self.__start = None
        self.__end = None

    def add_point(self, point: list) -> None:
        self.__points.append(np.array(point))

    def add_points(self, points: list) -> None:
        self.__start = points[0]
        self.__end = points[1]
        for point in points:
            self.add_point(point)

    def add_line(self, line: Line) -> None:
        self.__lines.append(line)
        self.__line_lined_points += line.generate_line()
        for ele in line.points:
            self.__line_points.append(ele)
    
    def add_lines(self, lines) -> None:
        for line in lines:
            self.add_line(line)
    
    def add_triangle(self, triangle: Triangle) -> None:
        self.__triangles.append(triangle)
        self.__triangle_lined_points += triangle.generate_line()
        for ele in triangle.points:
            self.__triangle_points.append(ele)

    def add_triangles(self, triangles) -> None:
        for triangle in triangles:
            self.add_triangle(triangle)

    def add_boundary(self, boundary: Line) -> None:
        self.__boundaries.append(boundary)
        self.__boundary_lined_points += boundary.generate_line()
        for ele in boundary.points:
            self.__boudnary_points.append(ele)
    
    def add_boundaries(self, boundaries) -> None:
        for boundary in boundaries:
            self.add_boundary(boundary)
    
    def add_polygon(self, polygon: list) -> None:
        triangles = triangulation(polygon)
        for vertices in triangles:
            self.add_triangle(Triangle(vertices))
    
    def add_polygons(self, polygons: list) -> None:
        for polygon in polygons:
            self.add_polygon(polygon)

    def __run_voronoi(self, points) -> None:
        self.__vor = Voronoi(points=points)


    def run_non_optimized(self, generate_result=True) -> namedtuple:
        # run voronoi
        self.__run_voronoi(self.__boundary_lined_points + 
                           self.__triangle_lined_points +
                           self.__line_lined_points +
                           self.__points)

        # calculate unreachable vertices and ridges
        unreachable_vertices = self.__vertices_in_polygon()
        if len(unreachable_vertices.flatten().tolist()) > 0:
            print(f'Currently pruning the unreachable vertices....')
            ridge_to_delete = self.__ridges_to_delete(unreachable_vertices)
            self.__delete_vertex(unreachable_vertices)
            self.__delete_ridge(ridge_to_delete)
            self.__reorganize_ridge(unreachable_vertices)

        # calculate dead_ends vertices
        dead_ends_vertices = self.__vertices_dead_ends(deg = 1)
        while len(dead_ends_vertices.flatten().tolist()) > 0:
            print(f'Currently pruning the dead ends: {len(dead_ends_vertices.flatten().tolist())}')
            ridge_to_delete = self.__ridges_to_delete(dead_ends_vertices)
            self.__delete_vertex(dead_ends_vertices)
            self.__delete_ridge(ridge_to_delete)
            self.__reorganize_ridge(dead_ends_vertices)
            dead_ends_vertices = self.__vertices_dead_ends()

        # Clean away the zero degree vertices 
        dead_ends_vertices = self.__vertices_dead_ends(deg=0)
        ridge_to_delete = self.__ridges_to_delete(dead_ends_vertices)
        self.__delete_vertex(dead_ends_vertices)
        self.__delete_ridge(ridge_to_delete)
        self.__reorganize_ridge(dead_ends_vertices)

        if generate_result:
            return self.__generate_result()

    def run_optimized(self) -> namedtuple:
        # run voronoi
        self.run_non_optimized(True)
        while True:
            if not self.__optimize_line():
                break
            self.__delete_unfinished()

        return self.__generate_result()

    def run(self, type = run_type.optimized, plot = False) -> namedtuple:
        if type == run_type.non_optimized: result = self.run_non_optimized()
        elif type == run_type.optimized:     result = self.run_optimized()
        if plot: self.generate_plot()
        return result

    # optimize line using Ramer-Douglas-Peucker algorithm
    def __optimize_line(self) -> bool:
         # generate chains in voronoi diagram
        self.__chains = self.__generate_chains()

        if len(self.__chains) == 0:
            return False
        print(f'the length of the generated chain is {len(self.__chains)}')
        # generate vertex chains based on chains
        vertex_chains = self.__generate_vertex_chains(self.__chains)

        # unoptimizable case
        if self.__chains == []: return self.__vor

        # optimize line
        optimized_chains = self.__optimize_line_base(vertex_chains)

        # regenerate ridges
        self.__regenerate_voronoi(optimized_chains)

        return True

    def __optimize_line_base(self, chains) -> list:
        optimized_chains = []
        for chain in chains:
            optimized_chains.append(rdp(chain, epsilon=self.rdp_epsilon))
        return optimized_chains
    
    # regenerate voronoi based on optimized valued
    def __regenerate_voronoi(self, chains) -> None:
        vertices = []
        ridge_vertices = []

        for chain in chains:
            if chain[0] not in vertices:
                vertices.append(chain[0])

            for i in range(len(chain)-1):
                idx = [vertices.index(chain[i]), -1]
                
                if chain[i+1] not in vertices:
                    vertices.append(chain[i+1])
                    idx[1] = len(vertices)-1
                else:
                    idx[1] = vertices.index(chain[i+1])
                
                ridge_vertices.append(np.array(idx))
        
        self.__vor.vertices = np.array(vertices)
        self.__vor.ridge_vertices = np.array(ridge_vertices)


    # generate vertex chain by line unit
    def __generate_chains(self):
        ridge_vertices = copy.deepcopy(self.__vor.ridge_vertices)
        dic = IndexDict(ridge_vertices)

        # ignition value must be dead end point(which has only 1 neighbor)
        ignition_idx = -1
        for key, value in dic.items():
            if len(value) == 1:
                ignition_idx = key
                break

        # voronoi diagram with no dead end point cannot be optimized
        if (ignition_idx == -1):
            return []

        # generate chains
        feature_point = []
        chains = []
        start_point = deque()
        start_point.append([-1, ignition_idx])
        while len(start_point) > 0:
            chains.append(self.__generate_chain(dic, start_point, feature_point))

        return chains

    # generate vertex chain based on index chain
    def __generate_vertex_chains(self, chains):
        vertex_chains = []
        for chain in chains:
            vertex_chain = []
            for ele in chain:
                vertex_chain.append(self.__vor.vertices[ele])
            vertex_chains.append(vertex_chain)
        return vertex_chains

    # generate chain
    def __generate_chain(self, dic, start, feature) -> None:
        chain = []

        # get new starting point
        idx = start.pop()

        # case of dead end point
        if idx[0] != -1: chain.append(idx[0])

        # ignite chain
        new_start = self.__chain_(dic, idx, chain, feature)

        # add chain start and end to feature points
        feature.append(chain[0])
        feature.append(chain[-1])

        # add new starting points to queue
        for ele in new_start:
            start.append(ele)
        
        return np.array(chain)

    # generate chain by finding next vertex recursively
    def __chain_(self, dic, idx, chain, feature) -> list:
        # append current point to chain
        chain.append(idx[1])

        # search neighbor on index dictionary
        neighbor = dic.find(idx[1])
        neighbor_count = len(neighbor)

        # visited is selected base on feature point(diverging point) and previous point
        visited = feature + idx[:1]

        # case 1, dead end point
        if neighbor_count == 1:
            if neighbor[0] == idx[0]: return []
            return self.__chain_(dic, [idx[1], neighbor[0]], chain, feature)

        # case 2, middle of line
        elif neighbor_count == 2:
            has_visited = [False, False]
            if neighbor[0] in visited: has_visited[0] = True
            if neighbor[1] in visited: has_visited[1] = True

            # if both neighbor is visited, it's end of line
            if has_visited[0] and has_visited[1]:
                if   idx[0] == neighbor[0]: chain.append(neighbor[1])
                elif idx[0] == neighbor[1]: chain.append(neighbor[0])
                return []

            # prevent going back
            elif has_visited[0]: next_idx = 1
            elif has_visited[1]: next_idx = 0
            else: raise ValueError("at least one neighbor has to be visited")

            # find next vertex
            return self.__chain_(dic, [idx[1], neighbor[next_idx]], chain, feature)
        
        # case more than 2, diverging point
        # line must end on diverging point
        # new starting points must be added for full construction
        new_start_points = []
        for i in range(neighbor_count):
            if neighbor[i] in visited: continue
            new_start_points.append([idx[1], neighbor[i]])
        return new_start_points

    def __delete_unfinished(self) -> None:
        # regenerate chain by optimized value
        self.__chains = self.__generate_chains()

        # calculate unfinished vertices and ridges
        unfinised_vertices = self.__unfinished_vertices()
        ridge_to_delete = self.__ridges_to_delete(unfinised_vertices)

        # delete unfinished vertices and ridges
        self.__delete_vertex(unfinised_vertices)
        self.__delete_ridge(ridge_to_delete)
        self.__reorganize_ridge(unfinised_vertices)
    
    # calculate vertices which is unfinished
    def __unfinished_vertices(self) -> list:
        dic = IndexDict(self.__vor.ridge_vertices)
        unfinished = []

        for key, value in dic.items():
            if len(value) == 1:
                unfinished.append(key)

        chain_vertices = np.array([], dtype=int)
        for ele in unfinished:
            for chain in self.__chains:
                if ele == chain[0]: 
                    chain_vertices = np.append(chain_vertices, chain[:-1])
                    break
                elif ele == chain[-1]:
                    chain_vertices = np.append(chain_vertices, chain[1:])
                    break
        
        chain_vertices = np.sort(chain_vertices)
        return chain_vertices
    
    # calculate vertices which are in convex
    def __vertices_in_polygon(self) -> list:
        in_polygon = []

        for i in range(len(self.__vor.vertices)):
            for tri in self.__triangles:
                if tri.is_in_polygon(self.__vor.vertices[i]):
                    in_polygon.append(i)
                    break

        return np.array(in_polygon)
    
    # calculate vertices which are in branches
    def __vertices_dead_ends(self, deg = 1) -> list:
        # find all the dead ends in the structure
        idx = np.full([len(self.__vor.vertices),1], 0)
        for i in range(len(self.__vor.ridge_vertices)):
            rv = self.__vor.ridge_vertices[i]
            idx[rv[0]] += 1
            idx[rv[1]] += 1
        if np.any(np.where(idx>2)):
            vertices = np.concatenate((np.argwhere(idx==deg)[...,0],), axis=0)
            return vertices
        else:
            return np.array([])
    
    # calculate ridges which are related to deleted vertices and outside
    def __ridges_to_delete(self, vertex_vec) -> list:
        to_delete = []
        vertices = self.__vor.vertices

        for i in range(len(self.__vor.ridge_vertices)):
            rv = self.__vor.ridge_vertices[i]

            # if ridge heads outside, delete ridge
            if rv[0] == -1 or rv[1] == -1:
                to_delete.append(i)
                continue

            # if ridge contains deleted vertex, delete ridge
            deleted = False
            for ver in vertex_vec:
                if rv[0] == ver or rv[1] == ver:
                    to_delete.append(i)
                    deleted = True
                    break
            if deleted: continue

            # if ridge intersects with line, delete ridge
            for line in self.__lines:
                l_2 = [vertices[rv[0]], vertices[rv[1]]]
                if line.is_intersecting(l_2):
                    to_delete.append(i)
                    break
        
        return to_delete
    
    # delete vertices
    def __delete_vertex(self, to_delete) -> None:
        self.__vor.vertices = np.delete(self.__vor.vertices, to_delete, 0)
    
    # delete unused ridge
    def __delete_ridge(self, to_delete) -> None:
        self.__vor.ridge_vertices = np.delete(self.__vor.ridge_vertices, to_delete, 0)

    # reorganize ridge
    def __reorganize_ridge(self, deleted_vertices) -> None:
        for i in range(len(self.__vor.ridge_vertices)):
            _0, _1 = self.__vor.ridge_vertices[i][0], self.__vor.ridge_vertices[i][1]
            _0_i, _1_i = find_closest(deleted_vertices, _0), find_closest(deleted_vertices, _1)
            self.__vor.ridge_vertices[i] = np.array([_0 - _0_i, _1 - _1_i], int)

    def __generate_result(self) -> namedtuple:
        # print('the Voroni result is being returned!>>>>>>>>>>>>>>>>>>>>>')
        ret_val = namedtuple('result', ['triangles', 'boundaries', 'points', 'points_polygon', 
                                        'vertices', 'ridge_vertices', 'start', 'end',"chains"])
        ret_val.triangles = np.array(self.__triangles, dtype=Triangle, copy=False)
        ret_val.boundaries = np.array(self.__boundaries, dtype=Line, copy=False)
        ret_val.points = np.array(self.__vor.points, dtype=float, copy=False)
        ret_val.points_polygon = np.array(self.__triangle_lined_points, dtype=float, copy=False)
        ret_val.vertices = np.array(self.__vor.vertices, dtype=float, copy=True)
        ret_val.ridge_vertices = np.array(self.__vor.ridge_vertices, dtype=int, copy=True)
        ret_val.start = np.array(self.__start, dtype=float, copy=True)
        ret_val.end = np.array(self.__start, dtype=float, copy=True)
        if len(self.__chains) > 0: 
            ret_val.chains = np.array(self.__chains, dtype=list, copy=True)
        return ret_val
    
    def generate_plot(self) -> None:
        kwarg = {"show_points":True, "show_vertices":True}
        voronoi_plot_2d(self.__vor, **kwarg)
        # kwarg = {"show_points":False, "show_vertices":False}
        # voronoi_plot_2d(self.__vor, **kwarg)
        # kwarg = {"show_points":True, "show_vertices":False}
        # voronoi_plot_2d(self.__vor, **kwarg)
        # kwarg = {"show_points":False, "show_vertices":True}
        # voronoi_plot_2d(self.__vor, **kwarg)
    
    def generate_plot_only_points(self) -> None:
        fig, ax = plt.subplots()
        points= self.__vor.points
        ax.plot(points[:,0], points[:,1], '.')

    
    def show(self) -> None:
        plt.show()

# delaunay triangulation
def triangulation(contours) -> list:
    triangles = []
    # print(f'the type of the contours are {contours}')
    triangulation = tripy.earclip(contours)
    for triangle in triangulation:
        vertices = []
        for vertex in triangle:
            vertices.append(list(vertex))
        triangles.append(Triangle(vertices))

    return triangles


# Copied from the official matplotlib library
def _adjust_bounds(ax, points):
    margin = 0.1 * points.ptp(axis=0)
    xy_min = points.min(axis=0) - margin
    xy_max = points.max(axis=0) + margin
    ax.set_xlim(xy_min[0], xy_max[0])
    ax.set_ylim(xy_min[1], xy_max[1])
    
