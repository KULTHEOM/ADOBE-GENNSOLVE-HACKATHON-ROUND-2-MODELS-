
import numpy as np
import matplotlib.pyplot as plt
import csv

import cv2

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


def is_straight_line(XY, dynamic_tolerance_factor=1e-6):
    if len(XY) < 2:
        return False 
    
    dx = XY[1:, 0] - XY[:-1, 0]
    dy = XY[1:, 1] - XY[:-1, 1]
    if np.any(dx == 0):  
        return False
    
    slopes = dy / dx
    
    slope_std = np.std(slopes)
    tolerance = dynamic_tolerance_factor * slope_std
    
    is_straight = np.all(np.abs(slopes - slopes[0]) < tolerance)
    
    return is_straight

def is_circle(XY, tolerance=9):
    center = np.mean(XY, axis=0)
    radii = np.sqrt((XY[:, 0] - center[0])**2 + (XY[:, 1] - center[1])**2)
    return np.all(np.abs(radii - radii[0]) < tolerance)



def is_rectangle(coords, tolerance=1):
    if len(coords) != 4:
        return False

    def distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    def angle_between(v1, v2):
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    # Calculate the distances between consecutive points
    side_lengths = [distance(coords[i], coords[(i + 1) % 4]) for i in range(4)]
    # Calculate the lengths of the diagonals
    diagonal_lengths = [distance(coords[i], coords[(i + 2) % 4]) for i in range(2)]

    # Check if opposite sides are equal within a tolerance
    if not (np.abs(side_lengths[0] - side_lengths[2]) < tolerance and 
            np.abs(side_lengths[1] - side_lengths[3]) < tolerance):
        return False

    # Check if the diagonals are equal within a tolerance
    if not np.abs(diagonal_lengths[0] - diagonal_lengths[1]) < tolerance:
        return False

    # Calculate the angles between consecutive sides
    angles = [] 
    for i in range(4):
        v1 = coords[(i + 1) % 4] - coords[i]
        v2 = coords[(i + 2) % 4] - coords[(i + 1) % 4]
        angles.append(angle_between(v1, v2))
    # Check if all angles are approximately 90 degrees (allowing a small tolerance)
    if all(80 < angle < 100 for angle in angles):
        return True

    return False


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

    return coords

def is_star(coords, tolerance=1.5):
    if len(coords) != 5:
        return None
    
    def distance(p1, p2):
        return np.linalg.norm(p1 - p2)

    # Sort points based on angle from centroid
    centroid = np.mean(coords, axis=0)
    angles = np.arctan2(coords[:,1] - centroid[1], coords[:,0] - centroid[0])
    sorted_indices = np.argsort(angles)
    sorted_coords = coords[sorted_indices]

    # Check star distances (alternate connections)
    d = []
    for i in range(5):
        d.append(distance(sorted_coords[i], sorted_coords[(i + 2) % 5]))

    side = d[0]
    if not all(abs(side - dist) < tolerance for dist in d):
        return None

    return sorted_coords



def fit_ellipse(XY):
    if len(XY) < 5:
        return None
    points = XY.reshape((-1, 1, 2)).astype(np.float32)
    ellipse = cv2.fitEllipse(points)
    return ellipse

def star_to_bezier(coords):
    bezier_curves = []
    for i in range(5):
        p0 = coords[i]
        p3 = coords[(i + 2) % 5]
        p1 = p0 + (p3 - p0) / 3
        p2 = p0 + 2 * (p3 - p0) / 3
        bezier_curves.append(np.array([p0, p1, p2, p3]))
        # print(f"Bezier Points for segment {i+1}: {np.array([p0, p1, p2, p3])}")
    return bezier_curves



def check_for_polygon(XY, tolerance=4):
    num_points = len(XY)
    for start in range(0, num_points - 3):
        for i in range(start + 1, num_points - 2):
            for j in range(i + 1, num_points - 1):
                for k in range(j + 1, num_points):
                    coords = np.array([XY[start], XY[i], XY[j], XY[k]])
                    if is_rectangle(coords, tolerance):
                        return coords
                    square_coords=is_square(coords,tolerance)
                    if square_coords is not None:
                        return square_coords
                    if fit_ellipse(coords):
                        return coords
                    
    return None

def check_for_star(XY, tolerance=1.5):
    num_points = len(XY)
    for a in range(0, num_points - 4):
        for b in range(a + 1, num_points - 3):
            for c in range(b + 1, num_points - 2):
                for d in range(c + 1, num_points - 1):
                    for e in range(d + 1, num_points):
                        coords = np.array([XY[a], XY[b], XY[c], XY[d], XY[e]])
                        star_coords=is_star(coords,tolerance)
                        if star_coords is not None:
                            return star_coords

    return None




def line_to_bezier(XY):
    p0 = XY[0]
    p3 = XY[-1]
    p1 = p0 + (p3 - p0) / 3
    p2 = p0 + 2 * (p3 - p0) / 3
    return np.array([p0, p1, p2, p3])


def circle_to_bezier(radius):
    kappa = 0.5522847498307936  
    r = radius
    
    # Control points for the first quadrant
    p0 = np.array([r, 0])
    p1 = np.array([r, kappa * r])
    p2 = np.array([kappa * r, r])
    p3 = np.array([0, r])

    # Reflect and rotate points to get the full circle
    control_points = [
        p0, p1, p2, p3,  # First quadrant
        [0, r], [-kappa * r, r], [-r, kappa * r], [-r, 0],  
        [-r, 0], [-r, -kappa * r], [-kappa * r, -r], [0, -r], 
        [0, -r], [kappa * r, -r], [r, -kappa * r], [r, 0]  
    ]
    bezier_curves = []
    for i in range(0, len(control_points), 4):
        bezier_curves.append(control_points[i:i+4])
        
    return bezier_curves

def rectangle_to_bezier(XY):
    bezier_curves = []
    for i in range(4):
        p0 = XY[i]
        p3 = XY[(i+1)%4]
        p1 = p0 + (p3 - p0) / 3
        p2 = p0 + 2 * (p3 - p0) / 3
        bezier_curves.append(np.array([p0, p1, p2, p3]))
    return bezier_curves

def square_to_bezier(XY):
    bezier_curves = []
    for i in range(4):
        p0 = XY[i]
        p3 = XY[(i+1) % 4]
        p1 = p0 + (p3 - p0) / 3
        p2 = p0 + 2 * (p3 - p0) / 3
        bezier_curves.append(np.array([p0, p1, p2, p3]))
        # print(f"Bezier Points for side {i+1}: {np.array([p0, p1, p2, p3])}")
    return bezier_curves




def ellipse_to_bezier(ellipse):
    bezier_points = []
    center, axes, angle = ellipse[0], ellipse[1], ellipse[2]
    
    a, b = axes[0] / 2.0, axes[1] / 2.0

    c = 0.551915024494  # Magic number for ellipse approximation

    p0 = np.array([a, 0]) + center
    p1 = np.array([a, c * b]) + center
    p2 = np.array([c * a, b]) + center
    p3 = np.array([0, b]) + center

    p4 = np.array([-a, 0]) + center
    p5 = np.array([-a, c * b]) + center
    p6 = np.array([-c * a, b]) + center
    p7 = np.array([0, -b]) + center

    bezier_points.append([p0, p1, p2, p3])
    bezier_points.append([p3, p6, p5, p4])
    bezier_points.append([p4, -p1 + 2*center, -p2 + 2*center, -p3 + 2*center])
    bezier_points.append([p7, p2, p1, p0])
    
    return bezier_points



def plot_all_bezier_curves_ellipse(control_points):
    for i in range(0, len(control_points), 4):
        bezier_points = control_points[i:i+4]
        plot_bezier_curve(bezier_points)


def plot_bezier_curve(bezier_points, num_points=100):
    t = np.linspace(0, 1, num_points)
    t = t[:, None]  # for broadcasting

    p0, p1, p2, p3 = bezier_points
    curve = (1 - t)**3 * p0 + \
            3 * (1 - t)**2 * t * p1 + \
            3 * (1 - t) * t**2 * p2 + \
            t**3 * p3

    plt.plot(curve[:, 0], curve[:, 1])



def save_shapes_to_csv(shapes, csv_path):
    try:
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for shape in shapes:
                writer.writerows(shape)  # Write the points directly
        print(f"CSV file created successfully at {csv_path}")
    except Exception as e:
        print(f"Error writing CSV file: {e}")



# Example usage
paths_XYs = read_csv('C:/Users/OM/Downloads/problems/problems/isolated.csv')
# main code 

shapes = []
for XYs in paths_XYs:
   
    
    for XY in XYs:
        if is_straight_line(XY):
        
            shapes.append(XY.tolist())
            bezier_curve = line_to_bezier(XY)
            plot_bezier_curve(bezier_curve)
            shapes.append(bezier_curve.tolist())

        # #to plot and detect circles 
        if is_circle(XY):
         
            center = np.mean(XY, axis=0)
            radius = np.mean(np.sqrt((XY[:, 0] - center[0])**2 + (XY[:, 1] - center[1])**2))
            bezier_control_points = circle_to_bezier(radius)
            for bezier_points in bezier_control_points:
                plot_bezier_curve(np.array(bezier_points))  
                shapes.append(np.array(bezier_points).tolist())  

        # #to plot and detect rectangles 
        rectangle_coords=check_for_polygon(XY)
        if rectangle_coords is not None:
           bezier_curves=rectangle_to_bezier(XY)
        #    print(bezier_curves)    
        for bezier_curve in bezier_curves:
               plot_bezier_curve(bezier_curve)
               shapes.append(bezier_curve.tolist())

        # to plot and detect squares 
        square_coords=check_for_polygon(XY)
        # print(f"Detected Square Coordinates: {square_coords}")
        if square_coords is not None:
            bezier_curves=square_to_bezier(square_coords)
            for bezier_curve in bezier_curves:
                plot_bezier_curve(bezier_curve)
                shapes.append(bezier_curve.tolist())
        else:
            print("No square found")
        # to plot and detect ellipse 
        ellipse_coords = check_for_polygon(XY)
        if ellipse_coords is not None:
                ellipse = fit_ellipse(ellipse_coords)
                if ellipse is not None:
                        control_points = ellipse_to_bezier(ellipse)
                        plot_all_bezier_curves_ellipse(control_points)
                        plt.gca().set_aspect('equal', adjustable='box')
                        plt.show()
                else:
                    print("No valid ellipse could be fitted")
        #to plot and detect star 
        star_coords=check_for_star(XY)
        if star_coords is not None:
            bezier_curves = star_to_bezier(star_coords)
            for bezier_curve in bezier_curves:
                plot_bezier_curve(bezier_curve)
                shapes.append(bezier_curve.tolist())

        else:
            print("no shapes detected ")

plt.show()
csv_path = 'C:/Users/OM/Downloads/problems/identified1_shapes.csv'
save_shapes_to_csv(shapes, csv_path)




#helper codes 


# XY = np.array([
#     [4.0, 1.0], [5.0, 2.0], [5.5, 3.5], [4.5, 4.0], [3.0, 3.5], [2.0, 2.0]
# ])

# ellipse_coords = check_for_polygon(XY)
# if ellipse_coords is not None:
#     ellipse = fit_ellipse(ellipse_coords)
#     if ellipse is not None:
#         control_points = ellipse_to_bezier(ellipse)
#         plot_all_bezier_curves_ellipse(control_points)
#         plt.gca().set_aspect('equal', adjustable='box')
#         plt.show()
#     else:
#         print("No valid ellipse could be fitted")
# else:
#     print("No shapes detected")


# points for star 


# coords = np.array([
#     [0, 1],
#     [0.5878, 0.809],
#     [0.9511, 0.309],
#     [0.5878, -0.809],
#     [0, -1],
#     [-0.5878, -0.809],
#     [-0.9511, 0.309],
#     [-0.5878, 0.809],
#     [0, 1]
# ])


#points for square 

# square_vertices = np.array([
#     [0, 1],  # top-left
#     [1, 1],  # top-right
#     [1, 0],  # bottom-right
#     [0, 0]   # bottom-left
# ])

#points for rectangle

# Example usage
# XY = np.array([
#     [7.98000002, 0.83999997],
#     [8.98950958, 0.84788334],
#     [9.99900818, 0.85685998],
#     [11.00849438, 0.86692989],
#     [12.01796722, 0.878093]
# ])



# # shapes = []
# # for XYs in paths_XYs:
# #     for XY in XYs:
# #         rectangle_coords=check_for_rectangles(XY)
# #         if rectangle_coords is not None:
# #             bezier_curves = rectangle_to_bezier(XY)
# #             for bezier_curve in bezier_curves:
# #                 plot_bezier_curve(bezier_curve)
# #                 shapes.append(bezier_curve.tolist())
# #         else:
# #             print("No rectangle found")
            

# # plt.show()

# shapes = []
# for XYs in paths_XYs:
#     for XY in XYs:
#         square_coords=check_for_polygon(XY)
#         if square_coords is not None:
#             bezier_curves=square_to_bezier(square_coords)
#             for bezier_curve in bezier_curves:
#                 plot_bezier_curve(bezier_curve)
#                 shapes.append(bezier_curve.tolist())
#         else:
#             print("No rectangle found")
            

# plt.show()


# # shapes = []
# # for XYs in paths_XYs:
# #     for XY in XYs:
# #         if is_circle(XY):
# #             print(f"Circle: {XY.tolist()}")
# #             center = np.mean(XY, axis=0)
# #             radius = np.mean(np.sqrt((XY[:, 0] - center[0])**2 + (XY[:, 1] - center[1])**2))
# #             bezier_control_points = circle_to_bezier(radius, center)
# #             for bezier_points in bezier_control_points:
# #                 plot_bezier_curve(np.array(bezier_points))  # Ensure bezier_points is a NumPy array
# #                 shapes.append(np.array(bezier_points).tolist())  # Convert to NumPy array then to list
# #         else:
# #             print("no")




# def circle_to_bezier(radius, center):
#     k = 4 * (np.sqrt(2) - 1) / 3
#     control_points = []

#     for i in range(4):
#         angle = np.pi / 2 * i
#         p0 = np.array([np.cos(angle), np.sin(angle)]) * radius + center
#         p1 = np.array([np.cos(angle + np.pi / 4) * k, np.sin(angle + np.pi / 4) * k]) * radius + center
#         p2 = np.array([np.cos(angle + 3 * np.pi / 4) * k, np.sin(angle + 3 * np.pi / 4) * k]) * radius + center
#         p3 = np.array([np.cos(angle + np.pi / 2), np.sin(angle + np.pi / 2)]) * radius + center
#         control_points.append([p0, p1, p2, p3])

#     return control_points


# def plot_paths(paths_XYs):
#     fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))

#     for i, XYs in enumerate(paths_XYs):
#         for XY in XYs:
#             ax.plot(XY[:, 0], XY[:, 1], linewidth=2)

#     ax.set_aspect('equal')
#     plt.show()



# # plt.show()
# print("Shapes detected:")
# for shape in shapes:
#     print(shape)

