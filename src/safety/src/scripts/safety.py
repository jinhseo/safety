#!/home/imlab/.miniconda3/envs/carla/bin/python
import rospy
import ros_numpy
import numpy as np
import message_filters as mf
from std_msgs.msg import String, Header, Float64, Bool, Float64MultiArray
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Vector3
from scipy.spatial import distance
from novatel_gps_msgs.msg import NovatelHeading2
from ackermann_msgs.msg import AckermannDriveStamped
from utils import generate_front_zone, generate_left_zone, generate_right_zone
from geodesy.utm import fromLatLong as proj
CLS =  {0: {'class': 'person',     'color': [220, 20, 60]},
        1: {'class': 'bicycle',    'color': [220, 20, 60]},
        2: {'class': 'car',        'color': [  0,  0,142]},
        3: {'class': 'motorcycle', 'color': [220, 20, 60]},
        5: {'class': 'bus',        'color': [  0,  0,142]},
        7: {'class': 'truck',      'color': [  0,  0,142]}
        }

class Safety:
    def __init__(self, pub_rate, freespace, target_waypoint, waypoint, car, points_ring, heading, heading_v2, can_data, d_object, frame_id):
        self.frame_id = frame_id
        self.rate = rospy.Rate(pub_rate)

        self.pub_zone = rospy.Publisher('target_safety_zone', PointCloud2, queue_size=10)
        self.pub_front = rospy.Publisher('front_zone', Bool, queue_size=10)
        self.pub_left = rospy.Publisher('left_zone', Bool, queue_size=10)
        self.pub_right = rospy.Publisher('right_zone', Bool, queue_size=10)
        self.pub_dist = rospy.Publisher('safety_dist', Float64, queue_size=10)

        #self.pub_pc_zone = rospy.Publisher('waypoint_zone', PointCloud2, queue_size=10)
        self.pub_array = rospy.Publisher('safety', Float64MultiArray, queue_size=10)

        self.sub_target_waypoint = mf.Subscriber(target_waypoint, MarkerArray)
        #self.sub_waypoint = mf.Subscriber(waypoint, MarkerArray)
        self.sub_car_waypoint = mf.Subscriber(car, Marker)

        self.sub_freespace = mf.Subscriber(freespace, PointCloud2)
        self.sub_points_ring = mf.Subscriber(points_ring, MarkerArray)
        #self.sub_heading = mf.Subscriber(heading, Float64)
        self.sub_heading_v2 = mf.Subscriber(heading_v2, Float64) ###TODO: to heading_v2
        self.sub_can_data = mf.Subscriber(can_data, AckermannDriveStamped)
        self.sub_d_object = mf.Subscriber(d_object, Detection2DArray)

        self.subs = []
        self.callbacks = []

        self.front_h, self.front_w = 10, 3
        self.left_h, self.left_w = 5, 3
        self.right_h, self.right_w = 5, 3

        self.interval = 10
        self.wp_interval = 4
        self.radius = 1**2

        self.front_offset = 2
        self.left_offset  = 1
        self.right_offset = 1

        self.anti_clock_nt = np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)], [np.sin(np.pi/2), np.cos(np.pi/2)]])

        self.start_time = rospy.Time.now().secs
        self.ped_detected_time = 0
        self.ped_disappear_time = rospy.Time.now().secs
        self.collect_dist = []
        self.ped_detected = []
        self.detected_pos = []

        subs = [self.sub_target_waypoint, self.sub_car_waypoint, self.sub_freespace, self.sub_points_ring, self.sub_heading_v2, self.sub_d_object]
        callbacks = [self.callback_target_waypoint, self.callback_car_waypoint, self.callback_freespace, self.callback_points_ring, self.callback_heading_v2, self.callback_d_object]

        ### load cross_walk nodes ###
        cross_walk = open('cross_node.txt', 'r')
        crosswalk_node = cross_walk.readlines()
        self.cw_node = []
        self.center = proj(35.64838, 128.40105, 0)
        self.center = np.array([self.center.easting, self.center.northing, self.center.altitude])
        for cw_n in crosswalk_node:
            lat, long = list(map(float, cw_n.split(',')))
            cw_utm = self.gps_to_utm(lat,long,0)
            self.cw_node.append(cw_utm)
        self.cw_node = np.array(self.cw_node)
        ### load cross_walk nodes ###

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
    def callback_d_object(self, d_object_msg):
        self.d_object_msg = d_object_msg

    def gps_to_utm(self, latitude, longitude, altitude):
        pos = proj(latitude, longitude, altitude)
        pos = np.array([pos.easting, pos.northing, pos.altitude])
        pos[:2] -= self.center[:2]
        return pos

    def to_narray(self, geometry_msg):
        points, points_x, points_y, points_z = [], [], [], []
        for msg in geometry_msg:
            #points.append([msg.x, msg.y, msg.z])
            points.append([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        return np.array(points)

    def set_front(self, lidar_xyz, front_x, front_y, theta, dist, wide_zone=False):
        front_theta = np.logical_and(180 <= theta, theta <= 360)

        if wide_zone:
            front_dist = dist < 15
            front_zone = np.where(front_theta * front_dist)
        else:
            front_zone_x = np.logical_and(self.front_offset < front_x, front_x <= self.front_h+self.front_offset)
            front_zone_y = np.logical_and(-self.front_w/2 <= front_y, front_y <= self.front_w/2)
            front_zone = np.where(front_theta * front_zone_x * front_zone_y)

        zone_x, zone_y = lidar_xyz[front_zone][:,0], lidar_xyz[front_zone][:,1]

        predefined_f_zone = generate_front_zone(self.front_offset, wide_zone=wide_zone)
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
        for i, callback in enumerate(self.callbacks):
            callback(args[i])

        out_msg = PointCloud2()
        filter_array = np.array([], dtype=np.int64)
        safety_msg = Float64MultiArray()

        detected_obj = []

        if len(self.d_object_msg.detections) != 0:
            for d_object in self.d_object_msg.detections:
                detected_obj.append(d_object.results[0].id)
                if d_object.results[0].id == 0:
                    self.detected_pos.append([d_object.bbox.center.x, d_object.bbox.center.y])
        if len(self.detected_pos) > 10:
            self.detected_pos = []

        if self.freespace_msg is not None and self.target_waypoint_msg.markers:
            lidar_pc = ros_numpy.point_cloud2.pointcloud2_to_array(self.freespace_msg)
            lidar_xyz = ros_numpy.point_cloud2.get_xyz_points(lidar_pc, remove_nans=True)
            freespace = lidar_xyz[:,:2]

            ### find waypoint ###
            car_position = np.array([[self.car_waypoint_msg.pose.position.x, self.car_waypoint_msg.pose.position.y, self.car_waypoint_msg.pose.position.z]])[:,:2]
            target_waypoint = self.to_narray(self.target_waypoint_msg.markers)[:,: 2]
            target_waypoint = target_waypoint - car_position

            heading_degree = np.radians(self.heading_v2_msg.data)
            heading_vector = np.dot(np.array([[np.cos(heading_degree), np.sin(heading_degree)], [-np.sin(heading_degree), np.cos(heading_degree)]]), np.array([0,1]))

            y_axis = heading_vector
            x_axis = np.dot(self.anti_clock_nt, heading_vector)
            transformed_waypoint = np.dot(target_waypoint, np.array([x_axis,y_axis]))
            transformed_freespace = np.dot(self.anti_clock_nt, freespace.T).T

            car_cw_dist = distance.cdist(car_position, self.cw_node[:,:2]).min()
            min_dist = distance.cdist(transformed_freespace, transformed_waypoint).min(0)

            go_waypoint = np.array([0])
            go_waypoint = (min_dist < 1).nonzero()[0]
            if go_waypoint.shape[0] != 0:
                to_where = go_waypoint.max()
                #print("to where:", to_where)
            else:
                to_where = -999
                print("nowhere to go")
            ### find waypoint ###

            ### safety zone ###
            car_cw_dist = distance.cdist(car_position, self.cw_node[:,:2]).min()

            if car_cw_dist < 5 and 0 in detected_obj:
                wide_zone, ped_detected = True, True
            else:
                wide_zone, ped_detected = False, False

            self.ped_detected.append(ped_detected)

            if sum(self.ped_detected[:5]) >= 1 and (len(self.ped_detected) - sum(self.ped_detected)) > 5:
                self.ped_disappear_time = rospy.Time.now().secs
                ped_disappear = True
            else:
                ped_disappear = False

            if len(self.ped_detected) > 15:
                last_element = self.ped_detected[-1]
                self.ped_detected = []
                self.ped_detected.append(last_element)

            #print(self.ped_detected)

            x, y = lidar_xyz[:, 0], lidar_xyz[:, 1]
            theta = np.rad2deg(np.arctan2(x, y)) + 180 # 0 ~ 360 degrees
            theta = np.round(theta).astype(int)
            theta[theta >= 360] -= 360
            dist = np.sqrt(x ** 2 + y ** 2)

            front_view, obstacle, obs_theta, obs_dist = self.set_front(lidar_xyz, x,y, theta, dist, wide_zone=wide_zone)

            if obstacle.shape[0] != 0:
                safety_front = False
                safety_dist = round(obs_dist,2)
                print('stop, pedestrian on the crosswalk')
            #elif obstacle.shape[0] == 0:
            #    safety_front = True
            #    safety_dist = self.front_h
            #    print('no obstacles, follow control command')
            elif obstacle.shape[0] == 0 and rospy.Time.now().secs == self.ped_disappear_time + rospy.Time(5).secs:
                print('5 secs after ped is disappeared, follow the control command')
            elif obstacle.shape[0] == 0 and rospy.Time.now().secs < self.ped_disappear_time + rospy.Time(5).secs:
                print('wait for 5 secs')
            elif obstacle.shape[0] == 0 and rospy.Time.now().secs != self.ped_disappear_time + rospy.Time(5).secs:
                print('no obstacles, follow the control command')

            if len(self.detected_pos) > 9:
                avg_x = np.array(self.detected_pos)[:,0].mean()
                avg_y = np.array(self.detected_pos)[:,1].mean()
                last_x = self.detected_pos[-1][0]
                last_y = self.detected_pos[-1][1]
                if last_x - 0.03 < avg_x < last_x + 0.03 and last_y - 0.03 < avg_y < last_y + 0.03:
                    print('object has stopped, follow the control command')
            #if ped_disappear:
            #    print(self.ped_detected)
                #import IPython; IPython.embed()
            out_msg = ros_numpy.point_cloud2.array_to_pointcloud2(lidar_pc[front_view])
            out_msg.header.frame_id = self.frame_id
            self.pub_zone.publish(out_msg)

            #self.collect_dist.append(round(obs_dist, 2))
            ### time sequence ###
            '''current_time = rospy.Time.now().secs
            if current_time == self.start_time + rospy.Time(1).secs:
                if len(self.collect_dist) == 1:
                    estim_movement = 'none'
                else:
                    dist_tau = sum(self.collect_dist) / len(self.collect_dist)
                    for dist in self.collect_dist:
                        if (dist > dist_tau + 5) or (dist < dist_tau - 5):
                            self.collect_dist.remove(dist)
                    self.start_time = rospy.Time.now().secs
                    self.collect_dist = []
            print(self.collect_dist)'''
            ### time sequence ###

            #safety_msg.data = np.array([safety_front, safety_left, safety_right, safety_dist, to_where])
            #self.pub_array.publish(safety_msg)


            ### speed ###
            #current_speed = self.can_data_msg.drive.speed
            ### speed ###

            ### recognition ###
            if len(self.d_object_msg.detections) != 0:
                cls_id = self.d_object_msg.detections[0].results[0].id
            else:
                cls_id = 'no detection'
            #print(cls_id)
            #import IPython; IPython.embed()
            ### recognition ###
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

    detected_objects = '/camera1/detected_objects'

    publisher = Safety(publish_rate, freespace, target_waypoint, waypoint, car, points_ring, heading, heading_v2, can_data, detected_objects, frame_id)

    rospy.spin()
