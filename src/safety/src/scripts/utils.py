import numpy as np
import os
from geodesy.utm import fromLatLong as proj

front_h, front_w = 13, 5
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
'''def generate_windows(detected_obj_msg, ped_cam, ped_move, car_cam):
    if len(detected_obj_msg.detections) != 0:
            for d_object in detected_obj_msg.detections:
                #if 0 < d_object.results[0].score < 25:
                    if d_object.results[0].id == 0:
                        ped_cam.pop(0)
                        ped_move.pop(0)
                        ped_cam.append(True)
                        ped_move.append(round(d_object.bbox.center.x, 2))
                        car_cam.pop(0)
                        car_cam.append(False)
                    if d_object.results[0].id == 2:
                        car_cam.pop(0)
                        car_cam.append(True)
                        ped_cam.pop(0)
                        ped_move.pop(0)
                        ped_cam.append(False)
                        ped_move.append(-1)
                else:
                    ped_cam.pop(0)
                    ped_cam.append(False)
                    ped_move.pop(0)
                    ped_move.append(-1)
                    car_cam.pop(0)
                    car_cam.append(False)
    elif len(detected_obj_msg.detections) == 0:
        ped_cam.pop(0)
        ped_cam.append(False)
        ped_move.pop(0)
        ped_move.append(-1)
        car_cam.pop(0)
        car_cam.append(False)
    return ped_cam, ped_move, car_cam
'''
def generate_windows_v2(detected_obj_msg, ped_cam, car_cam, on_one_lane):
    if len(detected_obj_msg.detections) != 0:
            for d_object in detected_obj_msg.detections:
                if 0 < d_object.results[0].score < 20:
                    if d_object.results[0].id == 0:
                        ped_cam.pop(0)
                        ped_cam.append(True)
                        car_cam.pop(0)
                        car_cam.append(False)
                    if d_object.results[0].id == 2:
                        if on_one_lane:
                            if d_object.bbox.center.x > 0.4 and d_object.bbox.center.x < 0.8:
                                car_cam.pop(0)
                                car_cam.append(True)
                                ped_cam.pop(0)
                                ped_cam.append(False)
                            else:
                                car_cam.pop(0)
                                car_cam.append(False)
                                ped_cam.pop(0)
                                ped_cam.append(False)
                        else:
                            car_cam.pop(0)
                            car_cam.append(True)
                            ped_cam.pop(0)
                            ped_cam.append(False)
                else:
                    ped_cam.pop(0)
                    ped_cam.append(False)
                    car_cam.pop(0)
                    car_cam.append(False)
    elif len(detected_obj_msg.detections) == 0:
        ped_cam.pop(0)
        ped_cam.append(False)
        car_cam.pop(0)
        car_cam.append(False)
    return ped_cam, car_cam

def target_object(detected_obj_msg, valid_car, valid_person):
    if len(detected_obj_msg.detections) != 0:
        for d_object in detected_obj_msg.detections:
            if 0 < d_object.results[0].score < 20:
                if d_object.results[0].id == 2 or d_object.results[0].id == 7:
                    valid_car.append(d_object.results[0].score)
                if d_object.results[0].id == 0:
                    valid_person.append(d_object.results[0].score)
    return valid_car, valid_person
