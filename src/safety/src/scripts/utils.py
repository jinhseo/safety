import numpy as np
import os
from geodesy.utm import fromLatLong as proj

front_h, front_w = 10, 3
left_h,  left_w  = 5, 3
right_h, right_w = 5, 3
interval = 10
front_offset = 2
left_offset  = 1
right_offset = 1

center = proj(35.64838, 128.40105, 0)
center = np.array([center.easting, center.northing, center.altitude])

def generate_front_zone(front_offset, wide_zone):
    if not wide_zone:
        front_x = np.linspace(front_offset, front_offset + front_h, front_h * interval)
        front_y = np.linspace(-front_w/2, front_w/2, front_w * interval)
        front_X, front_Y = np.meshgrid(front_x, front_y)
        f_zone = np.concatenate((front_X.flatten(), front_Y.flatten()), axis=0).reshape(2, front_h * interval * front_w * interval)
    else:
        #theta = np.linspace(0, np.pi, 180)
        #r = 10
        #x1 = r*np.cos(theta)
        #x2 = r*np.sin(theta)
        #circle_zone = np.reshape(np.concatenate((x1, x2), axis=0), (180, 2), order='F')
        #import IPython; IPython.embed()
        front_x = np.linspace(front_offset, front_offset + 15, 15 * 3) ### 5 is interval
        front_y = np.linspace(-15, 15, 15 * 3)
        front_X, front_Y = np.meshgrid(front_x, front_y)
        x0, y0, radius = 0.0, 0.0, 15
        r = np.sqrt((front_X - x0)**2 + (front_Y - y0)**2)
        shp = np.concatenate((front_X[r<15], front_Y[r<15]), axis=0).shape
        f_zone = np.concatenate((front_X[r<15], front_Y[r<15]), axis=0).reshape(2, int(shp[0]/2))
    return f_zone

def generate_left_zone(left_offset):
    left_x = np.linspace(-left_h + left_offset, 0 + left_offset, left_h * interval)
    left_y = np.linspace(0, left_w, left_w * interval)
    left_X, left_Y = np.meshgrid(left_x, left_y)
    l_zone = np.concatenate((left_X.flatten(), left_Y.flatten()), axis=0).reshape(2, left_h * interval * left_w * interval)
    return l_zone

def generate_right_zone(right_offset):
    right_x = np.linspace(-right_h + right_offset, 0 + right_offset, right_h * interval)
    right_y = np.linspace(-right_w, 0, right_w * interval)
    right_X, right_Y = np.meshgrid(right_x, right_y)
    r_zone = np.concatenate((right_X.flatten(), right_Y.flatten()), axis=0).reshape(2, right_h * interval * right_w * interval)
    return r_zone

def gps_to_utm(latitude, longitude, altitude):
    pos = proj(latitude, longitude, altitude)
    pos = np.array([pos.easting, pos.northing, pos.altitude])
    pos[:2] -= center[:2]
    return pos

def to_narray(geometry_msg):
    points, points_x, points_y, points_z = [], [], [], []
    for msg in geometry_msg:
        points.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    return np.array(points)

def load_cw(cw_path, cw_txt):

    cross_walk = open(os.path.join(cw_path, cw_txt), 'r')
    crosswalk_node = cross_walk.readlines()
    cw_node = []
    for cw_n in crosswalk_node:
        lat, long = list(map(float, cw_n.split(',')))
        cw_utm = gps_to_utm(lat, long, 0)
        cw_node.append(cw_utm)
    return np.array(cw_node)

