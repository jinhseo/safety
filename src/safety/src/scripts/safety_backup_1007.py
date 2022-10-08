#!/home/imlab/.miniconda3/envs/carla/bin/python
import rospy
import os
import rospkg
import ros_numpy
import numpy as np
import time
import message_filters as mf
from std_msgs.msg import String, Header, Float64, Bool, Float64MultiArray
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import PointCloud2, Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Vector3, Pose, PoseStamped, TransformStamped
from scipy.spatial import distance
from novatel_gps_msgs.msg import NovatelHeading2
from ackermann_msgs.msg import AckermannDriveStamped
from utils import generate_front_zone, gps_to_utm, to_narray, load_cw, generate_windows
from geodesy.utm import fromLatLong as proj
#from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose


CLS =  {0: {'class': 'person',     'color': [220, 20, 60]},
        1: {'class': 'bicycle',    'color': [220, 20, 60]},
        2: {'class': 'car',        'color': [  0,  0,142]},
        3: {'class': 'motorcycle', 'color': [220, 20, 60]},
        5: {'class': 'bus',        'color': [  0,  0,142]},
        7: {'class': 'truck',      'color': [  0,  0,142]}
        }

class Safety:
    def __init__(self, pub_rate, freespace, target_waypoint, car, points_ring, heading_v2,
                 dist_obj_0, dist_obj_1, dist_obj_2, traffic_sign, cw_path, frame_id):
        self.frame_id = frame_id
        self.rate = rospy.Rate(pub_rate)

        self.pub_zone = rospy.Publisher('target_safety_zone', PointCloud2, queue_size=10)
        self.pub_safety = rospy.Publisher('safety', AckermannDriveStamped, queue_size=10)

        self.sub_target_waypoint = mf.Subscriber(target_waypoint, MarkerArray)
        self.sub_car_waypoint = mf.Subscriber(car, Marker)

        self.sub_freespace = mf.Subscriber(freespace, PointCloud2)
        self.sub_points_ring = mf.Subscriber(points_ring, MarkerArray)
        self.sub_heading_v2 = mf.Subscriber(heading_v2, Float64) ###TODO: to heading_v2
        #self.sub_can_data = mf.Subscriber(can_data, AckermannDriveStamped)
        self.sub_detected_obj_0 = mf.Subscriber(dist_obj_0, Detection2DArray)
        self.sub_detected_obj_1 = mf.Subscriber(dist_obj_1, Detection2DArray)
        self.sub_detected_obj_2 = mf.Subscriber(dist_obj_2, Detection2DArray)
        self.sub_traffic_sign = mf.Subscriber(traffic_sign, Detection2DArray)

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

        self.ped_all_cam = False
        self.car_all_cam = False

        self.car_1_cam, self.car_0_cam, self.car_2_cam = [False] * 10, [False] * 10, [False] * 10
        self.traffic_cam = [False] * 10
        self.traffic_sign = False

        self.stop_speed = 0.0
        self.slow_speed = 10.0
        self.orig_speed = 30.0

        self.ped_stop_time, self.ped_stop_to_go_time = 0, 0
        self.ped_stop_start, self.ped_stop_to_go = False, False

        self.car_stop_time, self.lane_change_time = 0, 0
        self.car_stop_start, self.lane_change_start = False, False

        self.traffic_slow_time = 0
        self.traffic_slow_start = False

        self.ped_speed, self.sign_speed, self.car_speed, self.safety_speed = 100, 100, 100, 100
        self.anti_clock_nt = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)], [np.sin(np.pi/2), np.cos(np.pi/2)]])

        self.cw_node = load_cw(cw_path, 'cross_node_v2.txt')

        subs = [self.sub_target_waypoint, self.sub_car_waypoint, self.sub_freespace, self.sub_points_ring,
                self.sub_heading_v2, self.sub_detected_obj_0, self.sub_detected_obj_1, self.sub_detected_obj_2, self.sub_traffic_sign]
        callbacks = [self.callback_target_waypoint, self.callback_car_waypoint, self.callback_freespace, self.callback_points_ring,
                     self.callback_heading_v2, self.callback_detected_obj_0, self.callback_detected_obj_1, self.callback_detected_obj_2, self.callback_traffic_sign]

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
    def callback_traffic_sign(self, traffic_sign_msg):
        self.traffic_sign_msg = traffic_sign_msg

    def set_front(self, lidar_xyz, front_x, front_y, theta, dist):
        front_theta = np.logical_and(180 <= theta, theta <= 360)

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
        safety_msg = AckermannDriveStamped()
        target_speed = self.orig_speed
        lane_change = False
        for i, callback in enumerate(self.callbacks):
            callback(args[i])

        if len(self.traffic_sign_msg.detections) != 0:
            self.traffic_cam.pop(0)
            self.traffic_cam.append(True)
        elif len(self.traffic_sign_msg.detections) == 0:
            self.traffic_cam.pop(0)
            self.traffic_cam.append(False)
        self.traffic_sign = np.array([self.traffic_cam]).any()

        if self.traffic_sign:
            target_speed = self.slow_speed
            if not self.traffic_slow_start:
                self.traffic_slow_time = rospy.Time.now().secs
                self.traffic_slow_start = True
            #print('traffic sign is detected, slow down')
            #import IPython; IPython.embed()
        if self.traffic_slow_start and rospy.Time.now().secs - self.traffic_slow_time < rospy.Time(5).secs:
            target_speed = self.slow_speed
        elif self.traffic_slow_start and rospy.Time.now().secs - self.traffic_slow_time >= rospy.Time(5).secs:
            target_speed = self.orig_speed
            self.traffic_slow_start = False
            self.traffic_slow_time = 0
            self.traffic_cam = [False] * 10
            self.traffic_sign = False
        self.sign_speed = target_speed

        ### collect detection results ###
        self.ped_0_cam, self.ped_0_move, self.car_0_cam = generate_windows(self.detected_obj_0_msg, self.ped_0_cam, self.ped_0_move, self.car_0_cam)
        self.ped_1_cam, self.ped_1_move, self.car_1_cam = generate_windows(self.detected_obj_1_msg, self.ped_1_cam, self.ped_1_move, self.car_1_cam)
        self.ped_2_cam, self.ped_2_move, self.car_2_cam = generate_windows(self.detected_obj_2_msg, self.ped_2_cam, self.ped_2_move, self.car_2_cam)

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

        ### missions ###
        transformed_target_waypoints = []
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
            #transformed_waypoint = target_waypoint
            #transformed_cw = np.dot(cw_waypoint, np.array([x_axis, y_axis]))
            transformed_freespace = np.dot(self.anti_clock_nt, freespace.T).T

            car_cw_dist = distance.cdist(car_position, self.cw_node[:,:2]).min()


            #target_near_cw = distance.cdist(to_narray(self.target_waypoint_msg.markers)[:,:2], self.cw_node[:,:2]).min(1).argmin()
            #neighbour_cw = transformed_cw[(distance.cdist(self.cw_node[:,:2], np.array([self.cw_node[:,:2][distance.cdist(car_position, self.cw_node[:,:2]).argmin()]])) < 10).squeeze()]
            #neighbour_cw = transformed_cw[(distance.cdist(self.cw_node[:,:2], np.array([self.cw_node[:,:2][target_near_cw]])) < 5).squeeze()]
            #extend_neighbour_cw = np.linspace(neighbour_cw[0], neighbour_cw[-1], 50)

            #if neighbour_cw.shape[0] != 0:
                #ped_cw_dist = distance.cdist(transformed_freespace, neighbour_cw).min(0)
                #print((distance.cdist(transformed_freespace, neighbour_cw).min(0) > 1.0).any())
            #    print((distance.cdist(transformed_freespace, extend_neighbour_cw).min(0) > 1.0).any())

            ### find waypoint ###
            min_dist = distance.cdist(transformed_freespace, transformed_waypoint).min(0)
            go_waypoint = np.array([0])
            go_waypoint = (min_dist < 1).nonzero()[0]

            #print(distance.cdist(np.array([[0,0]]), np.array([transformed_waypoint[(min_dist < 1).nonzero()[0].max()]])))
            if go_waypoint.shape[0] != 0:
                to_where = go_waypoint.max()
                #to_where = distance.cdist(np.array([[0,0]]), np.array([transformed_waypoint[(min_dist < 1).nonzero()[0].max()]]))
                print("to where:", to_where)
                #print(distance.cdist(np.array([[0,0]]), np.array([transformed_waypoint[(min_dist < 1).nonzero()[0].max()]])))
            else:
                to_where = -999
                print("nowhere to go")

            current_time = rospy.Time.now().secs
            ### pedestrian mission ###
            if car_cw_dist < 15 and self.ped_all_cam and self.ped_diff > 0.05 and not self.ped_stop_to_go:# and self.ped_all_move:
                target_speed = self.stop_speed
                if not self.ped_stop_start:
                    self.ped_stop_time = rospy.Time.now().secs
                    self.ped_stop_start = True
                    #print('start ped stop')
            if self.ped_stop_start and current_time - self.ped_stop_time < rospy.Time(3).secs and not self.ped_stop_to_go:
                target_speed = self.stop_speed
                #print('stop for 3 secs')
            elif self.ped_stop_start and current_time - self.ped_stop_time >= rospy.Time(3).secs and not self.ped_stop_to_go:
                if car_cw_dist < 15 and self.ped_diff < 0.05:
                    #print('stop to go')
                    target_speed = self.orig_speed
                    self.ped_stop_to_go = True
                    self.ped_stop_to_go_time = rospy.Time.now().secs
                else:
                    print('still moving')
                    print(self.ped_diff)
                    target_speed = self.stop_speed
                    self.ped_stop_start = False
            if self.ped_stop_to_go and current_time - self.ped_stop_to_go_time < rospy.Time(5).secs:
                target_speed = self.orig_speed
                #print('go for 3 secs')
            elif self.ped_stop_to_go and current_time - self.ped_stop_to_go_time >= rospy.Time(5).secs:
                print('reset')
                self.ped_stop_start, self.ped_stop_to_go = False, False
                self.ped_stop_time, self.ped_stop_to_go_time = 0, 0
                self.ped_all_cam, self.ped_all_move = False, False
                self.ped_diff = 10
                self.ped_1_cam, self.ped_0_cam, self.ped_2_cam = [False] * 10, [False] * 10, [False] * 10

            self.ped_speed = target_speed

            ### car mission ###
            '''
            p_time = rospy.Time.now().secs
            any_stop = (0 < to_where < 5) and not self.lane_change_start
            #print(any_stop)
            if any_stop:
                target_speed = self.stop_speed
                is_car = self.car_all_cam                                       ###TODO: safet_dist and obj_dist
                if is_car:
                    if rospy.Time.now().secs - p_time > 5:
                        self.lane_change_start = True
                        p_time = rospy.Time.now().secs
                        print('lane change start')
                    else:
                        p_time = rospy.Time.now().secs
            else:
                target_speed = self.orig_speed
                if self.lane_change_start:
                    if rospy.Time.now().secs - p_time > 3:
                        self.lane_change_start = False
                        p_time = rospy.Time.now().secs
                    else:
                        p_time = rospy.Time.now().secs
            '''

            if 0 < to_where <= 5 and not self.lane_change_start:
                target_speed = self.stop_speed
                if self.car_all_cam and not self.car_stop_start:
                    self.car_stop_time = rospy.Time.now().secs
                    self.car_stop_start = True
                    #print('car stop start')
                if self.car_stop_start and current_time - self.car_stop_time > rospy.Time(10).secs:
                    target_speed = self.orig_speed
                    lane_change = True
                    self.car_stop_start = False
                    self.lane_change_time = rospy.Time.now().secs
                    self.lane_change_start = True
                    #print('lane change start')
            else:
                target_speed = self.orig_speed
                self.car_stop_start = False

            if self.lane_change_start and current_time - self.lane_change_time < rospy.Time(5).secs:
                target_speed = self.orig_speed
                lane_change = True
                #print('lane changing')
            elif self.lane_change_start and current_time - self.lane_change_time >= rospy.Time(5).secs:
                self.lane_change_start, self.car_stop_start, self.car_all_cam = False, False, False
                self.car_stop_time, self.lane_change_time = 0, 0
                self.car_1_cam, self.car_0_cam, self.car_2_cam = [False] * 10, [False] * 10, [False] * 10
            self.car_speed = target_speed

            ### safety zone ###
            '''x, y = lidar_xyz[:, 0], lidar_xyz[:, 1]
            theta = np.rad2deg(np.arctan2(x, y)) + 180 # 0 ~ 360 degrees
            theta = np.round(theta).astype(int)
            theta[theta >= 360] -= 360
            dist = np.sqrt(x ** 2 + y ** 2)
            front_view, obstacle, obs_theta, obs_dist = self.set_front(lidar_xyz, x,y, theta, dist)
            if 0 < obs_dist <= 10:
                self.safety_speed = 0
            else:
                self.safety_speed = 30.0
            '''
            #import IPython; IPython.embed()
            #safety_zone_msg = ros_numpy.point_cloud2.array_to_pointcloud2(lidar_pc[front_view])
            #safety_zone_msg.header.frame_id = self.frame_id
            #self.pub_zone.publish(safety_zone_msg)


            ### publish topics ###
            publish_speed = min([self.sign_speed, self.ped_speed, self.car_speed])
            #publish_speed = min([self.sign_speed, self.ped_speed, self.car_speed, self.safety_speed])
            safety_msg.drive.speed = publish_speed
            safety_msg.drive.jerk  = float(lane_change)
            safety_msg.header.stamp = rospy.Time.now()
            self.pub_safety.publish(safety_msg)
            #print(publish_speed)
            #print([self.sign_speed, self.ped_speed, self.car_speed])
        else:
            print('set target waypoint')

if __name__ == '__main__':
    rospy.init_node('freespace_vis_v3', anonymous=True)
    publish_rate      = rospy.get_param('~publish_rate' ,     10)
    frame_id          = rospy.get_param('~frame_id', 'base_link')

    car          = '/imcar/points_marker'
    target_waypoint = '/map/target_update'
    freespace   = '/os_cloud_node/freespace'
    points_ring = '/os_cloud_node/freespace_vis'

    heading_v2  = '/estimated_yaw'
    #can_data = '/can_data'

    dist_obj_0 = '/camera0/detected_objects_with_distance'
    dist_obj_1 = '/camera1/detected_objects_with_distance'
    dist_obj_2 = '/camera2/detected_objects_with_distance'

    traffic_obj = '/camera1/traffic_sign'

    #cw_path = os.path.join(rospkg.RosPack().get_path('safety'))
    cw_path = '/home/imlab/Downloads/autonomous/safety_pkg/src/safety'
    publisher = Safety(publish_rate, freespace, target_waypoint, car, points_ring, heading_v2,
                       dist_obj_0, dist_obj_1, dist_obj_2, traffic_obj, cw_path, frame_id)

    rospy.spin()
