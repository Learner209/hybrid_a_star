from voronoi.voronoi import GeneralizedVoronoi, run_type
from voronoi.geometry import Line, Triangle
from voronoi.astar import Astar
from voronoi.extract_contours import ContourDetector
from voronoi.potential_voronoi_field import VoronoiField
from config.defaults import _C as cfg
import numpy as np

from HybridAstarPlanner.hybrid_astar import AstarPathPlanner, ImprovedAstarPathPlanner, HybridAstarPathPlanner

if __name__ == '__main__':
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

    # # calculate Voronoi potential field
    # vorfield = VoronoiField(vor_result, alpha=np.array(10), d_o_max=np.array(.10))
    # potential_voronoi_field = vorfield.run(np.array([cfg.x_resolution, cfg.y_resolution]), start, end)
    # vorfield.generate_plot()
    # terrain_field = vorfield.terrain_on_resolution()
    # vorfield.show()

    sx = int(cfg.start[0] / cfg.x_resolution)
    sy = int(cfg.start[1] / cfg.y_resolution)
    gx = int(cfg.end[0] / cfg.x_resolution)
    gy = int(cfg.end[1] / cfg.y_resolution)

    sx = 5
    sy = 5
    gx = 110
    gy = 110

    print(f'the start is {sx, sy} and the end is {gx, gy}')
    # astar_path_planner = AstarPathPlanner()
    # astar_path_planner.astar(sx = sx, sy = sy, gx = gx, gy = gy, obsmap = np.load('map.npy'))a
    # assert False
    obsmap = np.load('map.npy')
    potential_voronoi_field = np.load("voronoi_potential_field.npy")
    improved_astar_path_planner = ImprovedAstarPathPlanner()
    improved_astar_path_planner.set_voronoi_potential_field(potential_voronoi_field, obsmap.shape[0], obsmap.shape[1])
    kwargs = {
        "potential_field_weight": 0.5,
        "steering_penalty_weight":0.0
    }
    improved_astar_path_planner.reproduce_heuristics_relation()
    # improved_astar_path_planner.reproduce_potential_voronoi_field_variation()
    # improved_astar_path_planner.get_steering_penalty(30,0,1)
    # Alpha = 1 means using the non-holomomic-without_obstacles 
    # Alpha = 0 mean using the holomomic-with_obstacles 
    # traj = improved_astar_path_planner.astar(sx = sx, sy = sy, syaw = np.deg2rad(120), gx = gx, gy = gy, gyaw=np.deg2rad(120), yaw_reso=5, obsmap = np.load('map.npy'), alpha = 0.5, **kwargs)
    # hybrid_astar_path_planner = HybridAstarPathPlanner(vor_result, obsmap = obsmap, alpha=np.array(10), d_o_max=np.array(cfg.d_o_max))
    # hybrid_astar_path_planner.set_voronoi_potential_field(potential_voronoi_field, obsmap.shape[0], obsmap.shape[1])
    # hybrid_astar_path_planner.run(None, np.array([cfg.x_resolution, cfg.y_resolution]))
    # hybrid_astar_path_planner.main(cfg)
