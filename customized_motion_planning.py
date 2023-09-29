from voronoi.voronoi import GeneralizedVoronoi, run_type
from voronoi.geometry import Line, Triangle
from voronoi.astar import Astar
from voronoi.extract_contours import ContourDetector
from voronoi.potential_voronoi_field import VoronoiField
from config.defaults import _C as cfg
import numpy as np


if __name__ == '__main__':
    # adjustable values
    Line.point_distance = 0.02
    Triangle.distance_trash = 0.002

    GeneralizedVoronoi.rdp_epsilon = 0.005
    ContourDetector.rdp_epsilon = 0.01
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
    terrain = pd.get_terrain()
    # boundary
    b1 = Line([[0.0, 0.0], [1.0, 0.0]])
    b2 = Line([[1.0, 0.0], [1.0, 1.0]])
    b3 = Line([[1.0, 1.0], [0.0, 1.0]])
    b4 = Line([[0.0, 0.0], [0.0, 1.0]])

    # voronoi
    vor = GeneralizedVoronoi()
    vor.add_triangles(triangles)
    vor.add_boundaries([b1, b2, b3, b4])
    vor.add_points([start, end])
    vor_result = vor.run(run_type.optimized)
    # vor.generate_plot()
    # vor.show()

    vorfield = VoronoiField(vor_result, alpha=np.array(0.02), d_o_max=np.array(.15))
    vorfield.run(np.array([cfg.x_resolution, cfg.y_resolution]))
    # vorfield.generate_plot()
    vorfield.terrain_on_resolution()
    vorfield.show()

