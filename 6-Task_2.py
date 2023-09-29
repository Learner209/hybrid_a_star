import sys
import os
import numpy as np
import matplotlib.pyplot as plt

MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '3-map/map.npy')

### START CODE HERE ###
# This code block is optional. You can define your utility function and class in this block if necessary.
  

from voronoi.voronoi import GeneralizedVoronoi, run_type
from voronoi.geometry import Line, Triangle
from voronoi.astar import Astar
from voronoi.extract_contours import ContourDetector
from voronoi.potential_voronoi_field import VoronoiField
from config.defaults import _C as cfg
import numpy as np

from HybridAstarPlanner.hybrid_astar import AstarPathPlanner, ImprovedAstarPathPlanner, HybridAstarPathPlanner, AstarPath

###  END CODE HERE  ###


def Improved_A_star(world_map, start_pos, goal_pos):
    """
    Given map of the world, start position of the robot and the position of the goal, 
    plan a path from start position to the goal using improved A* algorithm.

    Arguments:
    world_map -- A 120*120 array indicating map, where 0 indicating traversable and 1 indicating obstacles.
    start_pos -- A 2D vector indicating the start position of the robot.
    goal_pos -- A 2D vector indicating the position of the goal.

    Return:
    path -- A N*2 array representing the planned path by improved A* algorithm.
    """

    ### START CODE HERE ###

    # adjustable values
    Line.point_distance = 0.01
    Triangle.distance_trash = 0.002

    GeneralizedVoronoi.rdp_epsilon = 0.005
    ContourDetector.rdp_epsilon = 0.005
    ContourDetector.area_threshold = 400
    ContourDetector.gray_thresh_boundary = 20

    #point
    start = cfg.start
    end = cfg.end

    # polygon detector
    pd = ContourDetector('map.npy')
    pd.run(bound = [1.0, 1.0])
    triangles = pd.convert_result()
    # pd.vis_ori_gray()
    # pd.vis_contour()
    # pd.show()
    # terrain = pd.get_terrain()

    # boundary
    b1 = Line([[0.0, 0.0], [1.0, 0.0]])
    b2 = Line([[1.0, 0.0], [1.0, 1.0]])
    b3 = Line([[1.0, 1.0], [0.0, 1.0]])
    b4 = Line([[0.0, 0.0], [0.0, 1.0]])

    # # Voronoi Diagram
    vor = GeneralizedVoronoi()
    vor.add_triangles(triangles)
    vor.add_boundaries([b1, b2, b3, b4])
    vor.add_points([start, end])
    vor_result = vor.run(run_type.optimized)
    # vor.generate_plot()
    # vor.show()

    # calculate Voronoi potential field
    vorfield = VoronoiField(vor_result, alpha=np.array(10), d_o_max=np.array(.10))
    potential_voronoi_field = vorfield.run(np.array([cfg.x_resolution, cfg.y_resolution]), start, end)
    # vorfield.generate_plot()
    # terrain_field = vorfield.terrain_on_resolution()
    # vorfield.show()

    cfg.start[0] = start_pos[0] / world_map.shape[0]
    cfg.start[1] = start_pos[1] / world_map.shape[0]
    cfg.end[0] = goal_pos[0] / world_map.shape[1]
    cfg.end[1] = goal_pos[1] / world_map.shape[1]

    sx = int(cfg.start[0] / cfg.x_resolution)
    sy = int(cfg.start[1] / cfg.y_resolution)
    gx = int(cfg.end[0] / cfg.x_resolution)
    gy = int(cfg.end[1] / cfg.y_resolution)


    sx = start_pos[0]
    sy = start_pos[1]
    gx = goal_pos[0]
    gy = goal_pos[1]

    obsmap = world_map
    potential_voronoi_field = np.load("voronoi_potential_field.npy")
    print(f'the shape of the potential voronoi field is {potential_voronoi_field.shape}')
    improved_astar_path_planner = ImprovedAstarPathPlanner()
    improved_astar_path_planner.set_voronoi_potential_field(potential_voronoi_field, obsmap.shape[0], obsmap.shape[1])
    kwargs = {
        "potential_field_weight": 0.2,
        "steering_penalty_weight":1.0
    }
    # improved_astar_path_planner.reproduce_heuristics_relation()
    # improved_astar_path_planner.reproduce_potential_voronoi_field_variation()
    # improved_astar_path_planner.reproduce_steering_cost()
    # assert False
    # improved_astar_path_planner.get_steering_penalty(30,0,1)
    # # Alpha = 1 means using the non-holomomic-without_obstacles 
    # # Alpha = 0 mean using the holomomic-with_obstacles 
    alpha = 0.75
    # alpha = .6, potential_field_weight = .5 steering_penalty_weight = .8
    cur, _, _, _ = improved_astar_path_planner.astar(sx = sx, sy = sy, syaw = np.deg2rad(120), gx = gx, gy = gy, gyaw=np.deg2rad(120), yaw_reso=5, obsmap = np.load('map.npy'), alpha = alpha, **kwargs)
    path = AstarPath()
    while(cur is not None):
        path.x.append(cur.x)
        path.y.append(cur.y)
        cur = cur.prev
    path.x = path.x[::-1]
    path.y = path.y[::-1]
    path = [[path_x, path_y] for (path_x, path_y) in zip(path.x, path.y)]

    # ###  END CODE HERE  ###
    return path





if __name__ == '__main__':

    # Get the map of the world representing in a 120*120 array, where 0 indicating traversable and 1 indicating obstacles.
    map = np.load(MAP_PATH)

    # Define goal position of the exploration
    goal_pos = [100, 100]

    # Define start position of the robot.
    start_pos = [10, 10]

    # Plan a path based on map from start position of the robot to the goal.
    path = Improved_A_star(map, start_pos, goal_pos)

    # Visualize the map and path.
    obstacles_x, obstacles_y = [], []
    for i in range(120):
        for j in range(120):
            if map[i][j] == 1:
                obstacles_x.append(i)
                obstacles_y.append(j)

    path_x, path_y = [], []
    for path_node in path:
        path_x.append(path_node[0])
        path_y.append(path_node[1])

    plt.plot(path_x, path_y, "-r")
    plt.plot(start_pos[0], start_pos[1], "xr")
    plt.plot(goal_pos[0], goal_pos[1], "xb")
    plt.plot(obstacles_x, obstacles_y, ".k")
    plt.grid(True)
    plt.axis("equal")
    plt.show()

