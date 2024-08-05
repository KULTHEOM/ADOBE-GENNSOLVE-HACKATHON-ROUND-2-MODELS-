


import numpy as np
import csv
import svgwrite
from scipy.interpolate import splprep, splev

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def is_circle(XY, tolerance=20):
    center = np.mean(XY, axis=0)
    radii = np.sqrt((XY[:, 0] - center[0])**2 + (XY[:, 1] - center[1])**2)
    return np.std(radii) / np.mean(radii) < tolerance

def is_straight_line(XY, tolerance=0.5):
    if len(XY) < 2:
        return False
    dx = XY[1:, 0] - XY[:-1, 0]
    dy = XY[1:, 1] - XY[:-1, 1]
    slopes = dy / dx
    return np.std(slopes) < tolerance

def is_star(coords, tolerance=0.5):
    if len(coords) != 5:
        return None
    
    def distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    centroid = np.mean(coords, axis=0)
    angles = np.arctan2(coords[:,1] - centroid[1], coords[:,0] - centroid[0])
    sorted_indices = np.argsort(angles)
    sorted_coords = coords[sorted_indices]

    d = []
    for i in range(5):
        d.append(distance(sorted_coords[i], sorted_coords[(i + 2) % 5]))

    side = d[0]
    if not all(abs(side - dist) < tolerance for dist in d):
        return None

    return True

def is_square(coords, tolerance=10):
    if len(coords) != 4:
        return False

    def distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    d = []
    for i in range(4):
        for j in range(i + 1, 4):
            d.append(distance(coords[i], coords[j]))

    d.sort()

    side = d[0]
    if not all(abs(side - d[i]) < tolerance for i in range(4)):
        return None

    diagonal = d[4]
    if not abs(diagonal - d[5]) < tolerance:
        return None
    
    if not abs(diagonal - np.sqrt(2) * side) < tolerance:
        return None

    return True

def is_rectangle(coords, tolerance=10):
    if len(coords) != 4:
        return False

    def distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    def angle_between(v1, v2):
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    side_lengths = [distance(coords[i], coords[(i + 1) % 4]) for i in range(4)]
    diagonal_lengths = [distance(coords[i], coords[(i + 2) % 4]) for i in range(2)]

    if not (np.abs(side_lengths[0] - side_lengths[2]) < tolerance and 
            np.abs(side_lengths[1] - side_lengths[3]) < tolerance):
        return False

    if not np.abs(diagonal_lengths[0] - diagonal_lengths[1]) < tolerance:
        return False

    angles = [] 
    for i in range(4):
        v1 = coords[(i + 1) % 4] - coords[i]
        v2 = coords[(i + 2) % 4] - coords[(i + 1) % 4]
        angles.append(angle_between(v1, v2))

    if all(80 < angle < 100 for angle in angles):
        return True

    return False

def identify_shape(XY):
    if is_straight_line(XY):
        return XY
    if is_circle(XY):
        return XY
    if len(XY) == 4 and is_square(XY):
        return XY
    if len(XY) == 4 and is_rectangle(XY):
        return XY
    if len(XY) == 5 and is_star(XY):
        return XY
    return None

def polyline_to_bezier(XY, s=0.9):
    tck, u = splprep([XY[:, 0], XY[:, 1]], s=s)
    u_new = np.linspace(0, 1, 100)
    x_new, y_new = splev(u_new, tck)
    return np.array([x_new, y_new]).T

def create_svg(paths_XYs, svg_path):
    dwg = svgwrite.Drawing(svg_path, profile='tiny')
    for i, XYs in enumerate(paths_XYs):
        for XY in XYs:
            shape = identify_shape(XY)
            if shape is not None:
                bezier_curve = polyline_to_bezier(XY)  # It should use XY, not shape
                path_data = "M{},{} ".format(bezier_curve[0, 0], bezier_curve[0, 1])
                for j in range(1, len(bezier_curve) - 1, 3):
                    path_data += "C{},{} {},{} {},{} ".format(bezier_curve[j, 0], bezier_curve[j, 1],
                                                              bezier_curve[j + 1, 0], bezier_curve[j + 1, 1],
                                                              bezier_curve[j + 2, 0], bezier_curve[j + 2, 1])
                dwg.add(dwg.path(d=path_data, fill='none', stroke='black', stroke_width=2))
    dwg.save()


# Example usage
csv_path = "C:/Users/OM/Downloads/problems/problems/isolated.csv"
svg_path = "C:/Users/OM/Downloads/problems/problems/output_shapes.svg"
path_XYs = read_csv(csv_path)
create_svg(path_XYs, svg_path)


