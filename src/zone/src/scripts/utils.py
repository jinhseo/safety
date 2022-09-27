import numpy as np

front_h, front_w = 10, 3
left_h,  left_w  = 5, 3
right_h, right_w = 5, 3
interval = 10
front_offset = 2
left_offset  = 1
right_offset = 1

def generate_front_zone(front_offset):
    front_x = np.linspace(front_offset, front_offset + front_h, front_h * interval)
    front_y = np.linspace(-front_w/2, front_w/2, front_w * interval)
    front_X, front_Y = np.meshgrid(front_x, front_y)
    f_zone = np.concatenate((front_X.flatten(), front_Y.flatten()), axis=0).reshape(2, front_h * interval * front_w * interval)
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
