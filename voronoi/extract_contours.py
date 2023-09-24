"""
***************************************************************************************/
*    Title: Voronoi Generalized Diagram Implementation (Extract contours from the obstacle map)
*    Author: ross 1573
*    Date: 2022
*    Availability: https://github.com/ross1573/generalized_voronoi_diagram.git
*
***************************************************************************************/
"""


from typing import Any
import cv2
import numpy as np
import os

import matplotlib.pyplot as plt
import tripy
from voronoi.geometry import Triangle

    

class ContourDetector:
    rdp_epsilon: float
    area_threshold: int
    gray_thresh_boundary: int

    def __init__(self, path):
        self.__path = path
        assert os.path.exists(path)
        terrain = np.load(path)
        # Clear away the contours
        terrain[0,...] = 0
        terrain[-1,...] = 0
        terrain[...,0] = 0
        terrain[...,-1] = 0
        self.__terrain = terrain

        self.__gray = (terrain * 255).astype(np.uint8)
        ret, thresh = cv2.threshold(self.__gray, 200, 255, 0)
        self.__contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(len(self.__contours))
        contour_canvas = (np.ones_like(self.__gray) * 255).astype(np.uint8)
        self.__contoured_img = cv2.drawContours(contour_canvas, self.__contours, -1, (122,255,67), 2)

    def run(self, bound = [1.0, 1.0], triangulation = True):
        if bound[0] > 0.0 and bound[1] > 0.0:
            self.normalize(bound)
        else:
            self.normalize(bound = [self.__gray.shape[0], self.__gray.shape[1]])

        if triangulation:
            self.__result = self.triangulation()
        else:
            self.__result = self.__contours
        
        return self.__result

    # delaunay triangulation
    def triangulation(self) -> list:
        triangles = []
        for contour in self.__contours:
            triangulation = tripy.earclip(contour)
            for triangle in triangulation:
                vertices = []
                for vertex in triangle:
                    vertices.append(list(vertex))
                triangles.append(vertices)

        return triangles

    # normalize values between 0 and bound parameter
    # normalized based on image size
    def normalize(self, bound) -> None:
        size = self.__gray.shape
        multiplier = [bound[0] / size[1],
                      bound[1] / size[0]]
        
        contours = []
        for contour in self.__contours:
            points = []
            for point in contour:
                x = point[0][0] * multiplier[0]
                y = point[0][1] * multiplier[1]
                points.append([x,y])
            contours.append(points)
        self.__contours = contours

    # convert result to Triangle class which are acceptable in polygon voronoi
    def convert_result(self):
        triangles = []
        for vertices in self.__result:
            triangles.append(Triangle(vertices))
        return triangles

    def vis_contour(self):
        _, ax = plt.subplots()
        ax.imshow(self.__contoured_img.T)

    def vis_ori_gray(self):
        _, ax = plt.subplots()
        ax.imshow(self.__gray.T, cmap='gray')

    def show(self):
        plt.show()
    
    def get_terrain(self) -> Any:
        return self.__terrain

if __name__ == "__main__":
    pass