#!/usr/bin/env python

import rospy
import os
import rospkg
import ros_numpy
import numpy as np
import time
import message_filters as mf
from std_msgs.msg import String, Header, Float64, Bool, Int64MultiArray
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import PointCloud2, Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Vector3, Pose, PoseStamped, TransformStamped
from scipy.spatial import distance
from ackermann_msgs.msg import AckermannDriveStamped
from utils import generate_front_zone, gps_to_utm, to_narray, load_cw, generate_windows_v2
from geodesy.utm import fromLatLong as proj
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose

class Safety:
    def __init__(self, pub_rate, freespace, target_waypoint, car, can_data,
                 dist_obj_0, dist_obj_1, dist_obj_2, traffic_sign, cw_path, turns, frame_id):
        self.frame_id = frame_id
        self.rate = rospy.Rate(pub_rate)

        #self.pub_zone = rospy.Publisher('target_safety_zone', PointCloud2, queue_size=10)
        self.pub_safety = rospy.Publisher('safety', AckermannDriveStamped, queue_size=10)

        self.sub_target_turnpoint = mf.Subscriber(turns, Int64MultiArray)
        self.sub_target_waypoint = mf.Subscriber(target_waypoint, MarkerArray)
        self.sub_car_waypoint = mf.Subscriber(car, Marker)

        self.sub_freespace = mf.Subscriber(freespace, PointCloud2)
        self.sub_detected_obj_0 = mf.Subscriber(dist_obj_0, Detection2DArray)
        self.sub_detected_obj_1 = mf.Subscriber(dist_obj_1, Detection2DArray)
        self.sub_detected_obj_2 = mf.Subscriber(dist_obj_2, Detection2DArray)
        self.sub_traffic_sign = mf.Subscriber(traffic_sign, Detection2DArray)
        self.sub_can_data = mf.Subscriber(can_data, AckermannDriveStamped)

        self.subs = []
        self.callbacks = []

        self.ped_0_cam, self.ped_1_cam, self.ped_2_cam = [False] * 10, [False] * 10, [False] * 10
        self.car_0_cam, self.car_1_cam, self.car_2_cam = [False] * 10, [False] * 10, [False] * 10
        self.traffic_cam = [False] * 10

        self.ped_on_cw_can = [False] * 50
        self.ped_on_cw = False

        self.ped_all_cam = False
        self.car_all_cam = False
        self.traffic_sign = False

        self.stop_speed = 0.0
        self.slow_speed = 10.0
        self.orig_speed = 30.0
        self.voting_speed = [30.0, 30.0]

        self.ped_stop_time, self.ped_stop_to_go_time = 0, 0
        self.ped_stop_start, self.ped_stop_to_go = False, False
        self.ped_disappear = False
        self.car_stop_time, self.lane_change_time = 0, 0
        self.car_stop_start, self.lane_change_start = False, False

        self.traffic_slow_time = 0
        self.traffic_slow_start = False

        self.lane_goal = 999
        self.from_lane_change = -999
        self.lane_change_pos = 0
        self.cw_node = load_cw(cw_path, 'cross_node_v2.txt')
        self.one_node = load_cw(cw_path, 'one_lane.txt')
        subs = [self.sub_target_waypoint, self.sub_car_waypoint, self.sub_freespace, self.sub_can_data,
                self.sub_detected_obj_0, self.sub_detected_obj_1, self.sub_detected_obj_2, self.sub_traffic_sign, self.sub_target_turnpoint]
        callbacks = [self.callback_target_waypoint, self.callback_car_waypoint, self.callback_freespace, self.callback_can_data,
                     self.callback_detected_obj_0, self.callback_detected_obj_1, self.callback_detected_obj_2, self.callback_traffic_sign, self.callback_target_turnpoint]

        for sub, callback in zip(subs, callbacks):
            if sub is not None:
                self.subs.append(sub)
                self.callbacks.append(callback)

        self.ts = mf.ApproximateTimeSynchronizer(self.subs, 10, 0.1, allow_headerless=True)
        self.ts.registerCallback(self.callback)

    def callback_target_turnpoint(self, turn_msg):
        self.target_turnpoint_msg = turn_msg
    def callback_target_waypoint(self, target_waypoint_msg):
        self.target_waypoint_msg = target_waypoint_msg
    def callback_waypoint(self, waypoint_msg):
        self.waypoint_msg = waypoint_msg
    def callback_car_waypoint(self, car_msg):
        self.car_waypoint_msg = car_msg
    def callback_freespace(self, freespace_msg):
        self.freespace_msg = freespace_msg
    def callback_detected_obj_0(self, detected_obj_0_msg):
        self.detected_obj_0_msg = detected_obj_0_msg
    def callback_detected_obj_1(self, detected_obj_1_msg):
        self.detected_obj_1_msg = detected_obj_1_msg
    def callback_detected_obj_2(self, detected_obj_2_msg):
        self.detected_obj_2_msg = detected_obj_2_msg
    def callback_traffic_sign(self, traffic_sign_msg):
        self.traffic_sign_msg = traffic_sign_msg
    def callback_can_data(self, can_data_msg):
        self.can_data_msg = can_data_msg

    def getEquidistantPoints(self, p1, p2, parts):
        return zip(np.linspace(p1[0], p2[0], parts+1),
               np.linspace(p1[1], p2[1], parts+1))

    def to_narray(self, geometry_msg):
        points, points_x, points_y, points_z = [], [], [], []
        for msg in geometry_msg:
            points.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        return np.array(points)

    def callback(self, *args):
        for i, callback in enumerate(self.callbacks):
            callback(args[i])
        #safety_zone_msg = PointCloud2()
        safety_msg = AckermannDriveStamped()
        target_speed = self.orig_speed
        current_speed = self.can_data_msg.drive.speed
        lane_change = False
        self.ped_speed, self.sign_speed, self.car_speed, self.cw_speed, self.safety_speed = 30, 30, 30, 30, 30

        if len(self.traffic_sign_msg.detections) != 0:
            if self.traffic_sign_msg.detections[0].results[0].score > 0.8 and self.traffic_sign_msg.detections[0].bbox.center.x > 0.5:
                self.traffic_cam.pop(0)
                self.traffic_cam.append(True)
            else:
                self.traffic_cam.pop(0)
                self.traffic_cam.append(False)
        elif len(self.traffic_sign_msg.detections) == 0:
            self.traffic_cam.pop(0)
            self.traffic_cam.append(False)
        self.traffic_sign = np.array([self.traffic_cam]).any()

        ### collect detection results ###
        car_position = np.array([[self.car_waypoint_msg.pose.position.x, self.car_waypoint_msg.pose.position.y, self.car_waypoint_msg.pose.position.z]])[:,:2]
        if (distance.cdist(car_position, self.one_node[:,:2]) < 3).nonzero()[0].shape[0] != 0:
            on_one_lane = True
        else:
            on_one_lane = False
        self.ped_0_cam, self.car_0_cam = generate_windows_v2(self.detected_obj_0_msg, self.ped_0_cam, self.car_0_cam, on_one_lane = False)
        self.ped_1_cam, self.car_1_cam = generate_windows_v2(self.detected_obj_1_msg, self.ped_1_cam, self.car_1_cam, on_one_lane = on_one_lane)
        self.ped_2_cam, self.car_2_cam = generate_windows_v2(self.detected_obj_2_msg, self.ped_2_cam, self.car_2_cam, on_one_lane = False)

        self.ped_all_cam = np.array([np.array(self.ped_0_cam).any(), np.array(self.ped_1_cam).any(), np.array(self.ped_2_cam).any()]).any()
        self.car_all_cam = np.array([np.array(self.car_0_cam).any(), np.array(self.car_1_cam).any(), np.array(self.car_2_cam).any()]).any()

        ### transform ###
        trans = TransformStamped()
        trans.transform.translation.x = -self.car_waypoint_msg.pose.position.x
        trans.transform.translation.y = -self.car_waypoint_msg.pose.position.y
        trans.transform.translation.z = -self.car_waypoint_msg.pose.position.z
        trans.transform.rotation.x = 0
        trans.transform.rotation.y = 0
        trans.transform.rotation.z = 0
        trans.transform.rotation.w = 1

        trans1 = TransformStamped()
        trans1.transform.translation.x = 0
        trans1.transform.translation.y = 0
        trans1.transform.translation.z = 0
        trans1.transform.rotation.x = self.car_waypoint_msg.pose.orientation.x
        trans1.transform.rotation.y = self.car_waypoint_msg.pose.orientation.y
        trans1.transform.rotation.z = self.car_waypoint_msg.pose.orientation.z
        trans1.transform.rotation.w = -self.car_waypoint_msg.pose.orientation.w

        transformed_target_waypoints = []
        for marker in self.target_waypoint_msg.markers:
            x = do_transform_pose(PoseStamped(pose=Pose(position=marker.pose.position, orientation=marker.pose.orientation)), trans)
            x = do_transform_pose(x, trans1)
            transformed_target_waypoints.append([x.pose.position.x, x.pose.position.y])
        waypoint = np.array(transformed_target_waypoints)

        transformed_cross_waypoints = []
        for cw in self.cw_node:
            marker1 = Marker()
            marker1.pose.position.x = cw[0]
            marker1.pose.position.y = cw[1]
            marker1.pose.position.z = cw[2]
            marker1.pose.orientation.x = 0
            marker1.pose.orientation.y = 0
            marker1.pose.orientation.z = 0
            marker1.pose.orientation.w = 0
            y = do_transform_pose(PoseStamped(pose=Pose(position=marker1.pose.position, orientation=marker1.pose.orientation)), trans)
            y = do_transform_pose(y, trans1)
            transformed_cross_waypoints.append([y.pose.position.x, y.pose.position.y])
        cw = np.array(transformed_cross_waypoints)

        ### missions ###
        if self.freespace_msg is not None and self.target_waypoint_msg.markers:
        #if True:
            lidar_pc = ros_numpy.point_cloud2.pointcloud2_to_array(self.freespace_msg)
            lidar_xyz = ros_numpy.point_cloud2.get_xyz_points(lidar_pc, remove_nans=True)
            freespace = lidar_xyz[:,:2]

            car_position = np.array([[self.car_waypoint_msg.pose.position.x, self.car_waypoint_msg.pose.position.y, self.car_waypoint_msg.pose.position.z]])[:,:2]
            car_cw_dist  = distance.cdist(car_position, self.cw_node[:,:2]).min()
            car_cw_ind   = (distance.cdist(car_position, self.cw_node[:,:2]) < 40)[0].nonzero()[0]

            ### find waypoint ###
            #if waypoint.shape[0] != 0:
            waypoint_dist = distance.cdist(freespace, waypoint).min(0)
            waypoint_cw_ind = (distance.cdist(waypoint, cw).min(0) < 2).nonzero()[0]
            candidate_cw, candidate_cw_ind = np.unique(np.concatenate((waypoint_cw_ind, car_cw_ind)), return_counts=True)
            near_cw = candidate_cw[candidate_cw_ind > 1]
            near_cws = np.unique((distance.cdist(cw[near_cw], cw) < 10).nonzero()[1])
            #near_cws = np.unique(np.bitwise_and((distance.cdist(cw[near_cw], cw) < 10), (distance.cdist(cw[near_cw], cw) > 2)).nonzero()[1])
            try:
                near_cw_group = near_cws.reshape(near_cws.shape[0]/10, 10)
            except:
                near_cw_group = np.array([])
            ### min_dist ###
            if (waypoint_dist<1).nonzero()[0].shape[0] != 0:
                min_dist = round(distance.cdist(np.array([[0,0]]), np.array([waypoint[(waypoint_dist < 1).nonzero()[0].max()]])), 2)
            else:
                min_dist = -999

            ### cw_speed ###
            if near_cws.shape[0] != 0:
                cws_dist = distance.cdist(cw[near_cws], np.array([[0,0]]))
                #near_cws_dist = cws_dist.min()
                if cws_dist.min() < 5:
                    try:
                        near_cws_dist = cws_dist[(cws_dist > 5)].min()
                    except:
                        near_cws_dist = cws_dist.min()
                else:
                    near_cws_dist = cws_dist.min()
                if 20 < near_cws_dist < 30:
                    self.cw_speed = 15
                elif near_cws_dist <= 20:
                    self.cw_speed = 10
                #self.cw_speed = 15 if near_cws_dist < 30 else 30
                #self.cw_speed = current_speed - (30 - near_cws_dist)
            else:
                self.cw_speed = 30

            ### safety speed ###
            if 2 in self.target_turnpoint_msg.data and 0 < min_dist < 15: ### corner point
                self.safety_speed = 30
            elif 2 not in self.target_turnpoint_msg.data and 20 < min_dist < 40: ### straight point
                self.safety_speed = 15
            elif 2 not in self.target_turnpoint_msg.data and 0 < min_dist <= 20:
                self.safety_speed = 10
            else:
                self.safety_speed = 30

            ### traffic sign ###
            current_time = rospy.Time.now().secs
            if self.traffic_sign:
                target_speed = self.slow_speed
                if not self.traffic_slow_start:
                    self.traffic_slow_time = rospy.Time.now().secs
                    self.traffic_slow_start = True
            if self.traffic_slow_start and rospy.Time.now().secs - self.traffic_slow_time < 5:
                target_speed = self.slow_speed
            elif self.traffic_slow_start and rospy.Time.now().secs - self.traffic_slow_time >= 5:
                target_speed = self.orig_speed
                self.traffic_slow_start, self.traffic_sign = False, False
                self.traffic_slow_time = 0
                self.traffic_cam = [False] * 10
            self.sign_speed = target_speed
            #import IPython; IPython.embed()
            ### pedestrian mission ###
            if near_cws.shape[0] < 10:
                cw_ped = distance.cdist(freespace, cw[near_cws]).min(0)
            elif near_cws.shape[0] >= 10:
                cw_ped = distance.cdist(freespace, cw[near_cws]).min(0)
                #cw_ped = np.sort(distance.cdist(freespace, cw[near_cws]).min(0))[:10]

            if cw_ped.shape[0] != 0:
                if cw_ped.max() > 0.3:
                    self.ped_on_cw_can.pop(0)
                    self.ped_on_cw_can.append(True)
                else:
                    self.ped_on_cw_can.pop(0)
                    self.ped_on_cw_can.append(False)
            else:
                self.ped_on_cw_can.pop(0)
                self.ped_on_cw_can.append(False)
            self.ped_on_cw = np.array(self.ped_on_cw_can).any()

            if near_cws.shape[0] != 0:
                cw_ahead = 0 < near_cws_dist < 30
            else:
                cw_ahead = False
            print(self.ped_on_cw)
            print(self.ped_all_cam)
            print(cw_ahead)
            ped_on_ahead_cw = cw_ahead and self.ped_on_cw and self.ped_all_cam
            if ped_on_ahead_cw:
                target_speed = self.stop_speed
                if not self.ped_stop_start:
                    self.ped_stop_time = rospy.Time.now().secs
                    self.ped_stop_start = True
            elif not np.array(self.ped_on_cw).any() and current_time - self.ped_stop_time > rospy.Time(3).secs:
                target_speed = self.orig_speed
            else:
                target_speed = self.orig_speed
            self.ped_speed = target_speed

            ### car mission ###
            obs_stop = False
            print(self.target_turnpoint_msg.data)
            if (2 in self.target_turnpoint_msg.data[:15]) and 0 < min_dist < 5:
                obs_stop = True
                #print('1')
            elif (2 not in self.target_turnpoint_msg.data[:15]) and 0 < min_dist < 30:
                obs_stop = True
                #print('2')
            else:
                obs_stop = False

            #print(obs_stop)
            #print(self.lane_change_start)
            #print(lane_change)
            if self.to_narray(self.target_waypoint_msg.markers).shape[0] < 20:
                self.lane_change_start = False
            if on_one_lane:
                self.lane_change_start = False
            if obs_stop and not self.lane_change_start:
                target_speed = self.stop_speed
                if self.car_all_cam and not self.car_stop_start:
                    self.car_stop_time = rospy.Time.now().secs
                    self.car_stop_start = True
                elif self.car_stop_start and current_time - self.car_stop_time > 10:
                    target_speed = self.orig_speed
                    self.lane_goal = min_dist + 5 + 5
                    self.lane_change_pos = self.to_narray(self.target_waypoint_msg.markers)[:,: 2][0]
                    self.car_stop_start = False
                    self.lane_change_time = rospy.Time.now().secs
                    self.lane_change_start = True
            elif self.lane_change_start:
                self.from_lane_change = distance.cdist(car_position, np.expand_dims(self.lane_change_pos,0))[0][0]
                lane_change = True
                if self.from_lane_change > self.lane_goal:
                    lane_change = False
                    target_speed = self.orig_speed
                    self.lane_goal = 999
                    self.from_lane_change = -999
                    self.lane_change_pos = 0
                    self.car_stop_time = 0
                    self.lane_change_start, self.car_stop_start, self.car_all_cam = False, False, False
                    self.car_1_cam, self.car_0_cam, self.car_2_cam = [False] * 10, [False] * 10, [False] * 10
            else:
                target_speed = self.orig_speed
                self.car_stop_start = False
            self.car_speed = target_speed
            print(min_dist)
            ### publish topics ###
            publish_speed = min([self.sign_speed, self.ped_speed, self.car_speed, self.cw_speed, self.safety_speed])
            if self.to_narray(self.target_waypoint_msg.markers).shape[0] < 20:
                publish_speed = 30
            self.voting_speed.pop(0)
            self.voting_speed.append(publish_speed)
            if self.voting_speed[-1] - self.voting_speed[0] >= 5:
                publish_speed = self.voting_speed[0] + 5
            self.voting_speed.pop(0)
            self.voting_speed.append(publish_speed)
            safety_msg.drive.speed = publish_speed
            safety_msg.drive.jerk  = float(lane_change)
            safety_msg.header.stamp = rospy.Time.now()
            self.pub_safety.publish(safety_msg)
            print([self.sign_speed, self.ped_speed, self.car_speed, self.cw_speed, self.safety_speed])
        else:
            print('set target waypoint')
            print(self.to_narray(self.target_waypoint_msg.markers).shape[0])
            rospy.logwarn("set target waypoint")
if __name__ == '__main__':
    rospy.init_node('freespace_vis_v3', anonymous=True)
    publish_rate      = rospy.get_param('~publish_rate' ,     10)
    frame_id          = rospy.get_param('~frame_id', 'base_link')

    car          = '/imcar/points_marker'
    target_waypoint = '/map/target_update'
    target_turnpoint = '/map/target_turnpoints'

    freespace   = '/os_cloud_node/freespace'

    can_data = '/can_data'

    dist_obj_0 = '/camera0/detected_objects_with_distance'
    dist_obj_1 = '/camera1/detected_objects_with_distance'
    dist_obj_2 = '/camera2/detected_objects_with_distance'

    traffic_obj = '/camera1/traffic_sign'

    #cw_path = os.path.join(rospkg.RosPack().get_path('safety'))
    cw_path = '/home/imlab/safety/src/safety/src/scripts'
    #cw_path = '/safety/src/safety/src/scripts'
    publisher = Safety(publish_rate, freespace, target_waypoint, car, can_data,
                       dist_obj_0, dist_obj_1, dist_obj_2, traffic_obj, cw_path, target_turnpoint, frame_id)

    rospy.spin()
