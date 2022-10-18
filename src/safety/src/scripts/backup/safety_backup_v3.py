#!/home/imlab/.miniconda3/envs/carla/bin/python
import rospy
import ros_numpy
import numpy as np
import time
import message_filters as mf
from std_msgs.msg import String, Header, Float64, Bool, Float64MultiArray
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Vector3
from scipy.spatial import distance
from novatel_gps_msgs.msg import NovatelHeading2
from ackermann_msgs.msg import AckermannDriveStamped
from utils import generate_front_zone, gps_to_utm, to_narray, load_cw
from geodesy.utm import fromLatLong as proj
CLS =  {0: {'class': 'person',     'color': [220, 20, 60]},
        1: {'class': 'bicycle',    'color': [220, 20, 60]},
        2: {'class': 'car',        'color': [  0,  0,142]},
        3: {'class': 'motorcycle', 'color': [220, 20, 60]},
        5: {'class': 'bus',        'color': [  0,  0,142]},
        7: {'class': 'truck',      'color': [  0,  0,142]}
        }

class Safety:
    def __init__(self, pub_rate, freespace, target_waypoint, waypoint, car, points_ring, heading, heading_v2, can_data,
                 dist_obj_0, dist_obj_1, dist_obj_2, frame_id):
        self.frame_id = frame_id
        self.rate = rospy.Rate(pub_rate)

        self.pub_zone = rospy.Publisher('target_safety_zone', PointCloud2, queue_size=10)
        self.pub_target_speed = rospy.Publisher('target_speed', Float64, queue_size = 10)
        self.pub_dist = rospy.Publisher('safety_dist', Float64, queue_size=10)
        self.pub_lane_change = rospy.Publisher('lane_change', Bool, queue_size=10)

        self.sub_target_waypoint = mf.Subscriber(target_waypoint, MarkerArray)
        self.sub_car_waypoint = mf.Subscriber(car, Marker)

        self.sub_freespace = mf.Subscriber(freespace, PointCloud2)
        self.sub_points_ring = mf.Subscriber(points_ring, MarkerArray)
        self.sub_heading_v2 = mf.Subscriber(heading_v2, Float64) ###TODO: to heading_v2
        #self.sub_can_data = mf.Subscriber(can_data, AckermannDriveStamped)
        self.sub_detected_obj_0 = mf.Subscriber(dist_obj_0, Detection2DArray)
        self.sub_detected_obj_1 = mf.Subscriber(dist_obj_1, Detection2DArray)
        self.sub_detected_obj_2 = mf.Subscriber(dist_obj_2, Detection2DArray)
        #self.sub_dist_obj_1 = mf.Subscriber(dist_obj_1, Detection2DArray)

        self.subs = []
        self.callbacks = []

        self.front_h, self.front_w = 10, 3

        self.interval = 10
        self.wp_interval = 4
        self.radius = 1**2

        self.front_offset = 2

        self.ped_1_cam, self.ped_0_cam, self.ped_2_cam = [False] * 10, [False] * 10, [False] * 10
        self.ped_0_move, self.ped_1_move, self.ped_2_move = [-1] * 10, [-1] * 10, [-1] * 10
        self.ped_0_dist, self.ped_1_dist, self.ped_2_dist = [-1] * 10, [-1] * 10, [-1] * 10
        self.ped_0_diff, self.ped_1_diff, self.ped_2_diff = np.array([]), np.array([]), np.array([])
        self.ped_diff = 10

        self.ped_front_cam = False
        self.ped_all_cam = False
        self.car_all_cam = False

        self.ped_signal = False

        self.car_1_cam, self.car_0_cam, self.car_2_cam = [False] * 10, [False] * 10, [False] * 10

        self.stop_speed = 0.0
        self.slow_speed = 5.0
        self.orig_speed = 30.0

        self.ped_stop_time = 0
        self.ped_stop_start = False
        self.car_stop_time = 0
        self.car_stop_start = False
        self.lane_change_time = 0
        self.lane_change_start = False

        self.anti_clock_nt = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)], [np.sin(np.pi/2), np.cos(np.pi/2)]])
        self.cw_node = load_cw('cross_node.txt')

        subs = [self.sub_target_waypoint, self.sub_car_waypoint, self.sub_freespace, self.sub_points_ring,
                self.sub_heading_v2, self.sub_detected_obj_0, self.sub_detected_obj_1, self.sub_detected_obj_2]
        callbacks = [self.callback_target_waypoint, self.callback_car_waypoint, self.callback_freespace, self.callback_points_ring,
                     self.callback_heading_v2, self.callback_detected_obj_0, self.callback_detected_obj_1, self.callback_detected_obj_2]

        for sub, callback in zip(subs, callbacks):
            if sub is not None:
                self.subs.append(sub)
                self.callbacks.append(callback)

        self.ts = mf.ApproximateTimeSynchronizer(self.subs, 10, 0.1, allow_headerless=True)
        self.ts.registerCallback(self.callback)

    def callback_target_waypoint(self, target_waypoint_msg):
        self.target_waypoint_msg = target_waypoint_msg
    def callback_waypoint(self, waypoint_msg):
        self.waypoint_msg = waypoint_msg
    def callback_car_waypoint(self, car_msg):
        self.car_waypoint_msg = car_msg
    def callback_freespace(self, freespace_msg):
        self.freespace_msg = freespace_msg
    def callback_points_ring(self, points_ring_msg):
        self.points_ring_msg = points_ring_msg
    def callback_heading(self, heading_msg):
        self.heading_msg = heading_msg
    def callback_heading_v2(self, heading_v2_msg):
        self.heading_v2_msg = heading_v2_msg
    def callback_can_data(self, can_data_msg):
        self.can_data_msg = can_data_msg
    def callback_detected_obj_0(self, detected_obj_0_msg):
        self.detected_obj_0_msg = detected_obj_0_msg
    def callback_detected_obj_1(self, detected_obj_1_msg):
        self.detected_obj_1_msg = detected_obj_1_msg
    def callback_detected_obj_2(self, detected_obj_2_msg):
        self.detected_obj_2_msg = detected_obj_2_msg

    def set_front(self, lidar_xyz, front_x, front_y, theta, dist):
        front_theta = np.logical_and(180 <= theta, theta <= 360)

        #if wide_zone:
        #    front_dist = dist < 15
        #    front_zone = np.where(front_theta * front_dist)
        #else:
        front_zone_x = np.logical_and(self.front_offset < front_x, front_x <= self.front_h+self.front_offset)
        front_zone_y = np.logical_and(-self.front_w/2 <= front_y, front_y <= self.front_w/2)
        front_zone = np.where(front_theta * front_zone_x * front_zone_y)

        zone_x, zone_y = lidar_xyz[front_zone][:,0], lidar_xyz[front_zone][:,1]

        predefined_f_zone = generate_front_zone(self.front_offset, wide_zone=False)
        empty_zone = np.round(predefined_f_zone.T[(distance.cdist(np.array([zone_x,zone_y]).T, predefined_f_zone.T).min(0) >= 1.0)])
        empty_point = np.unique(empty_zone, axis=0)
        obstacle = np.round(np.dot(self.anti_clock_nt, empty_point.T).T)

        if obstacle.shape[0] != 0:
            obs_dist = distance.cdist(np.array([[self.front_offset, 0]]), obstacle).min()
            obs_ind  = distance.cdist(np.array([[self.front_offset, 0]]), obstacle).argmin()
            obs_theta = np.rad2deg(np.arctan2(np.array([0,1]), obstacle[distance.cdist(np.array([[0,1]]), obstacle).argmin()]))
            obs_theta[obs_theta >= 180] -= 180
            obs_theta = obs_theta.max()
        else:
            obs_theta = -999
            obs_dist  = -999
        return front_zone, obstacle, obs_theta, obs_dist

    def callback(self, *args):
        safety_zone_msg = PointCloud2()
        target_speed = self.orig_speed

        for i, callback in enumerate(self.callbacks):
            callback(args[i])

        ### collect detection results ###
        if len(self.detected_obj_1_msg.detections) != 0:
            for d_object in self.detected_obj_1_msg.detections:
                self.ped_1_cam.pop(0)
                self.ped_1_move.pop(0)
                self.car_1_cam.pop(0)
                cam_1_pid = True if d_object.results[0].id == 0 else False
                cam_1_cid = True if d_object.results[0].id == 2 else False
                self.ped_1_cam.append(cam_1_pid)
                self.car_1_cam.append(cam_1_cid)
                cam_1_move = round(d_object.bbox.center.x,2) if d_object.results[0].id == 0 else -1
                self.ped_1_move.append(cam_1_move)
        elif len(self.detected_obj_1_msg.detections) == 0:
            self.ped_1_cam.pop(0)
            self.ped_1_cam.append(False)
            self.ped_1_move.pop(0)
            self.ped_1_move.append(-1)
            self.car_1_cam.pop(0)
            self.car_1_cam.append(False)

        if len(self.detected_obj_0_msg.detections) != 0:
            for d_object in self.detected_obj_0_msg.detections:
                self.ped_0_cam.pop(0)
                self.ped_0_move.pop(0)
                self.car_0_cam.pop(0)
                cam_0_pid = True if d_object.results[0].id == 0 else False
                cam_0_cid = True if d_object.results[0].id == 2 else False
                self.ped_0_cam.append(cam_0_pid)
                self.car_0_cam.append(cam_0_cid)
                cam_0_move = round(d_object.bbox.center.x,2) if d_object.results[0].id == 0 else -1
                self.ped_0_move.append(cam_0_move)
        elif len(self.detected_obj_0_msg.detections) == 0:
            self.ped_0_cam.pop(0)
            self.ped_0_cam.append(False)
            self.ped_0_move.pop(0)
            self.ped_0_move.append(-1)
            self.car_0_cam.pop(0)
            self.car_0_cam.append(False)

        if len(self.detected_obj_2_msg.detections) != 0:
            for d_object in self.detected_obj_2_msg.detections:
                self.ped_2_cam.pop(0)
                self.ped_2_move.pop(0)
                self.car_2_cam.pop(0)
                cam_2_pid = True if d_object.results[0].id == 0 else False
                cam_2_cid = True if d_object.results[0].id == 2 else False
                self.ped_2_cam.append(cam_2_pid)
                self.car_2_cam.append(cam_2_cid)
                cam_2_move = round(d_object.bbox.center.x,2) if d_object.results[0].id == 0 else -1
                self.ped_2_move.append(cam_2_move)
        elif len(self.detected_obj_2_msg.detections) == 0:
            self.ped_2_cam.pop(0)
            self.ped_2_cam.append(False)
            self.ped_2_move.pop(0)
            self.ped_2_move.append(-1)
            self.car_0_cam.pop(0)
            self.car_0_cam.append(False)

        self.ped_all_cam = np.array([np.array(self.ped_0_cam).any(), np.array(self.ped_1_cam).any(), np.array(self.ped_2_cam).any()]).any()
        self.car_all_cam = np.array([np.array(self.car_0_cam).any(), np.array(self.car_1_cam).any(), np.array(self.car_2_cam).any()]).any()
        self.ped_all_move = np.array([sum(np.array(self.ped_0_move) > 0) > 7,
                            sum(np.array(self.ped_1_move) > 0) > 7,
                            sum(np.array(self.ped_2_move) > 0) > 7]).any()

        if self.ped_all_move:
            self.ped_0_diff = np.array(self.ped_0_move)[np.array(self.ped_0_move) > 0]
            self.ped_1_diff = np.array(self.ped_1_move)[np.array(self.ped_1_move) > 0]
            self.ped_2_diff = np.array(self.ped_2_move)[np.array(self.ped_2_move) > 0]
            self.ped_diff = abs(np.diff(self.ped_0_diff)).sum() + abs(np.diff(self.ped_1_diff)).sum() + abs(np.diff(self.ped_2_diff)).sum()

        #print(len(self.detected_obj_1_msg.detections))
        #print(len(self.detected_obj_1_msg.detections[0].results))

        ### collect detection results ###
        if self.freespace_msg is not None and self.target_waypoint_msg.markers:
            lidar_pc = ros_numpy.point_cloud2.pointcloud2_to_array(self.freespace_msg)
            lidar_xyz = ros_numpy.point_cloud2.get_xyz_points(lidar_pc, remove_nans=True)
            freespace = lidar_xyz[:,:2]

            ### find waypoint ###
            car_position = np.array([[self.car_waypoint_msg.pose.position.x, self.car_waypoint_msg.pose.position.y, self.car_waypoint_msg.pose.position.z]])[:,:2]
            target_waypoint = to_narray(self.target_waypoint_msg.markers)[:,:2] - car_position

            cw_waypoint = self.cw_node[:, :2] - car_position

            heading_degree = np.radians(self.heading_v2_msg.data)
            heading_vector = np.dot(np.array([[np.cos(heading_degree), np.sin(heading_degree)], [-np.sin(heading_degree), np.cos(heading_degree)]]), np.array([0,1]))

            y_axis = heading_vector
            x_axis = np.dot(self.anti_clock_nt, heading_vector)
            transformed_waypoint = np.dot(target_waypoint, np.array([x_axis, y_axis]))
            transformed_cw = np.dot(cw_waypoint, np.array([x_axis, y_axis]))
            transformed_freespace = np.dot(self.anti_clock_nt, freespace.T).T

            car_cw_dist = distance.cdist(car_position, self.cw_node[:,:2]).min()
            min_dist = distance.cdist(transformed_freespace, transformed_waypoint).min(0)

            #target_near_cw = self.cw_node[:, :2][distance.cdist(to_narray(self.target_waypoint_msg.markers)[:,:2], self.cw_node[:, :2]).min(0) < 5]
            #neighbour_cw = transformed_cw[(distance.cdist(target_near_cw, self.cw_node[:, :2]).min(0) < 5)]

            if car_cw_dist < 10 and self.ped_all_cam and self.ped_all_move:
                target_speed = self.stop_speed
                if not self.ped_stop_start:
                    self.ped_stop_time = rospy.Time.now().secs
                    self.ped_stop_start = True
            if self.ped_stop_start and rospy.Time.now().secs - self.ped_stop_time < rospy.Time(5).secs: ## greater? lower?
                target_speed = self.stop_speed
                self.ped_stop_start = False
            elif car_cw_dist < 10 and self.ped_diff < 0.05:
                target_speed = self.orig_speed

            #target_near_cw = self.cw_node[:, :2][distance.cdist(to_narray(self.target_waypoint_msg.markers)[:,: 2], self.cw_node[:, :2]).min(0) < 5]
            #neighbour_cw = transformed_cw[(distance.cdist(target_near_cw, self.cw_node[:, :2]).min(0) < 5)]
            #target_cw = self.cw_node[:, :2][distance.cdist(target_near_cw, self.cw_node[:, :2]).min(0) < 5]
            #neighbour_cw = transformed_cw[(distance.cdist(car_position, self.cw_node[:, :2]) < 15)[0]]
            #if neighbour_cw.shape[0] != 0:
            #    ped_cw_dist = distance.cdist(transformed_freespace, neighbour_cw).min(0)
            #    go_crosswalk = (ped_cw_dist < 1).nonzero()[0]
            #    if go_crosswalk.shape[0] != 0:
            #        print(go_crosswalk.max())
            #ped_cw_dist = distance.cdist(transformed_freespace, transformed_cw)


            go_waypoint = np.array([0])
            go_waypoint = (min_dist < 1).nonzero()[0]
            if go_waypoint.shape[0] != 0:
                to_where = go_waypoint.max()
                print("to where:", to_where)
            else:
                to_where = -999
                print("nowhere to go")
            ### find waypoint ###

            if self.car_all_cam and to_where < 5:
                target_speed = self.stop_speed
                if not self.car_stop_start:
                    self.car_stop_time = rospy.Time.now().secs
                    self.car_stop_start = True
            if self.car_stop_start and rospy.Time.now().secs - self.car_stop_time > rospy.Time(10).secs:
                target_speed = self.orig_speed
                lane_change = True
                self.car_stop_start = False

                self.lane_change_time = rospy.Time.now().secs
                self.lane_change_start = True
                print('lane change start')
            if self.lane_change_start and rospy.Time.now().secs - self.lane_change_time < rospy.Time(5).secs:
                target_speed = self.orig_speed
                lane_change = True
                print('lane changing')

            ### safety zone ###
            x, y = lidar_xyz[:, 0], lidar_xyz[:, 1]
            theta = np.rad2deg(np.arctan2(x, y)) + 180 # 0 ~ 360 degrees
            theta = np.round(theta).astype(int)
            theta[theta >= 360] -= 360
            dist = np.sqrt(x ** 2 + y ** 2)

            front_view, obstacle, obs_theta, obs_dist = self.set_front(lidar_xyz, x,y, theta, dist)

            safety_zone_msg = ros_numpy.point_cloud2.array_to_pointcloud2(lidar_pc[front_view])
            safety_zone_msg.header.frame_id = self.frame_id
            #safety_zone_msg.header.stamp = rospy.Time.now()
            self.pub_zone.publish(safety_zone_msg)

            ### publish topics ###
            self.pub_target_speed.publish(target_speed)
            #self.pub_dist.publish(
            #self.pub_lane_change.publish(lane_change)
            ### publish topics ###

            print(target_speed)
        else:
            print('set target waypoint')

if __name__ == '__main__':
    rospy.init_node('freespace_vis_v3', anonymous=True)
    publish_rate      = rospy.get_param('~publish_rate' ,     10)
    frame_id          = rospy.get_param('~frame_id', 'base_link')

    car          = '/imcar/points_marker'
    waypoint     = '/map/node'
    target_waypoint = '/map/target_update'
    freespace   = '/os_cloud_node/freespace'
    points_ring = '/os_cloud_node/freespace_vis'
    heading = '/heading2'
    heading_v2  = '/estimated_yaw'
    can_data = '/can_data'

    dist_obj_0 = '/camera0/detected_objects_with_distance'
    dist_obj_1 = '/camera1/detected_objects_with_distance'
    dist_obj_2 = '/camera2/detected_objects_with_distance'

    publisher = Safety(publish_rate, freespace, target_waypoint, waypoint, car, points_ring, heading, heading_v2, can_data,
                       dist_obj_0, dist_obj_1, dist_obj_2, frame_id)

    rospy.spin()
