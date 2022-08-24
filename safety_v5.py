#!/root/anaconda3/envs/carla_py2/bin/python2
import rospy
import ros_numpy
import numpy as np
import message_filters as mf
from std_msgs.msg import String, Header, Float64, Bool
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Vector3
from scipy.spatial import distance
from novatel_gps_msgs.msg import NovatelHeading2

class Safety:
    def __init__(self, pub_rate, freespace, target_waypoint, waypoint, car, points_ring, heading_v2, frame_id):
        self.frame_id = frame_id
        self.rate = rospy.Rate(pub_rate)

        self.pub_zone = rospy.Publisher('target_safety_zone', PointCloud2, queue_size=10)
        self.pub_front = rospy.Publisher('front_zone', Bool, queue_size=10)
        self.pub_left = rospy.Publisher('left_zone', Bool, queue_size=10)
        self.pub_right = rospy.Publisher('right_zone', Bool, queue_size=10)
        self.pub_dist = rospy.Publisher('safety_dist', Float64, queue_size=10)

        self.sub_freespace = mf.Subscriber(freespace, PointCloud2)
        self.sub_target_waypoint = mf.Subscriber(target_waypoint, Marker)
        self.sub_waypoint = mf.Subscriber(waypoint, Marker)
        self.sub_car_waypoint = mf.Subscriber(car, Marker)
        self.sub_points_ring = mf.Subscriber(points_ring, MarkerArray)
        self.sub_heading_v2 = mf.Subscriber(heading_v2, Float64)
        self.subs = []
        self.callbacks = []

        self.front_h, self.front_w = 15, 3
        self.left_h, self.left_w = 5, 3
        self.right_h, self.right_w = 5, 3

        self.interval = 5
        self.wp_interval = 5*5
        self.radius = 1**2

        self.front_x = np.linspace(0, self.front_h, self.front_h*self.interval)
        self.front_y = np.linspace(-self.front_w/2, self.front_w/2, self.front_w*self.interval)
        self.front_X, self.front_Y = np.meshgrid(self.front_x, self.front_y)
        self.front_zone = np.concatenate((self.front_X.flatten(), self.front_Y.flatten()), axis=0).reshape(2, self.front_h*self.interval*self.front_w*self.interval)

        self.left_x = np.linspace(-self.left_h, 0, self.left_h*self.interval)
        self.left_y = np.linspace(0, self.left_w, self.left_w*self.interval)
        self.left_X, self.left_Y = np.meshgrid(self.left_x, self.left_y)
        self.left_zone = np.concatenate((self.left_X.flatten(), self.left_Y.flatten()), axis=0).reshape(2, self.left_h*self.interval*self.left_w*self.interval)

        self.right_x = np.linspace(-self.right_h, 0, self.right_h *self.interval)
        self.right_y = np.linspace(-self.right_w, 0, self.right_w *self.interval)
        self.right_X, self.right_Y = np.meshgrid(self.right_x, self.right_y)
        self.right_zone = np.concatenate((self.right_X.flatten(), self.right_Y.flatten()), axis=0).reshape(2, self.right_h*self.interval*self.right_w*self.interval)

        subs = [self.sub_target_waypoint, self.sub_waypoint, self.sub_car_waypoint, self.sub_freespace, self.sub_points_ring, self.sub_heading_v2]
        callbacks = [self.callback_target_waypoint, self.callback_waypoint, self.callback_car_waypoint, self.callback_freespace, self.callback_points_ring, self.callback_heading_v2]

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

    def to_narray(self, geometry_msg):
        points, points_x, points_y, points_z = [], [], [], []
        for msg in geometry_msg:
            points.append([msg.x, msg.y, msg.z])
        return np.array(points)

    def set_front(self, lidar_xyz, front_x, front_y, theta):
        dist = np.sqrt(front_x ** 2 + front_y ** 2)
        theta = np.rad2deg(np.arctan2(front_x, front_y)) + 180 # 0 ~ 360 degrees
        theta = np.round(theta).astype(int)
        theta[theta >= 360] -= 360

        front_theta = np.logical_and(180 <= theta, theta <= 360)
        front_zone_x = np.logical_and(0 < front_x, front_x <= self.front_h)
        front_zone_y = np.logical_and(-self.front_w/2 <= front_y, front_y <= self.front_w/2)

        #front_dist = dist < 5
        front_zone = np.where(front_theta * front_zone_x * front_zone_y)
        #front_zone = np.where(front_theta)
        #front_zone = np.where(front_theta * front_dist)

        zone_x, zone_y = lidar_xyz[front_zone][:,0], lidar_xyz[front_zone][:,1]

        empty_zone = np.round(self.front_zone.T[(distance.cdist(np.array([zone_x,zone_y]).T, self.front_zone.T).min(0) >= 1)])
        empty_point = np.unique(empty_zone, axis=0)
        obstacle = np.round(np.dot(np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)], [np.sin(np.pi/2), np.cos(np.pi/2)]]), empty_point.T).T)

        if obstacle.shape[0] != 0:
            obs_dist = distance.cdist(np.array([[0, 0]]), obstacle).min() - 2
            obs_ind  = distance.cdist(np.array([[0, 0]]), obstacle).argmin()
            obs_theta = np.rad2deg(np.arctan2(np.array([0,1]), obstacle[distance.cdist(np.array([[0,1]]), obstacle).argmin()]))
            obs_theta[obs_theta >= 180] -= 180
            obs_theta = obs_theta.max()
            #size_candidates = obstacle[:,1][np.logical_and(obstacle[:,1] - 1 <= obstacle[:,1], obstacle[:,1] <= obstacle[:,1][obs_ind] + 1)]
            #approx_size = size_candidates.max() - size_candidates.min()
        else:
            obs_theta = -999
            obs_dist  = -999
            #approx_size = -999
        return front_zone, obstacle, obs_theta, obs_dist

    def set_left(self, lidar_xyz, left_x, left_y, theta):
        left_theta = np.logical_and(90 <= theta, theta <= 270)
        left_zone_x = np.logical_and(-self.left_h <= left_x, left_x < 0)
        left_zone_y = np.logical_and(0 < left_y, left_y <= self.left_w)
        left_zone = np.where(left_theta * left_zone_x * left_zone_y)[0]

        zone_x, zone_y = lidar_xyz[left_zone][:,0], lidar_xyz[left_zone][:,1]

        empty_zone = np.round(self.left_zone.T[(distance.cdist(np.array([zone_x,zone_y]).T, self.left_zone.T).min(0) >= 1)])
        empty_point = np.unique(empty_zone, axis=0)
        obstacle = np.round(np.dot(np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)], [np.sin(np.pi/2), np.cos(np.pi/2)]]), empty_point.T).T)

        if obstacle.shape[0] != 0:
            obs_dist = distance.cdist(np.array([[0, 0]]), obstacle).min() - 2
            obs_ind  = distance.cdist(np.array([[0, 0]]), obstacle).argmin()
            obs_theta = np.rad2deg(np.arctan2(np.array([0,1]), obstacle[distance.cdist(np.array([[0,1]]), obstacle).argmin()]))
            obs_theta[obs_theta >= 180] -= 180
            obs_theta = obs_theta.max()
        else:
            obs_theta = -999
            obs_dist  = -999
        return left_zone, obstacle, obs_theta, obs_dist

    def set_right(self, lidar_xyz, right_x, right_y, theta):
        right_theta = np.logical_or(np.logical_and(0 <= theta, theta <=90), theta >= 270)
        right_zone_x = np.logical_and(-self.right_h <= right_x, right_x < 0)
        right_zone_y = np.logical_and(0 > right_y, right_y >= -self.right_w)
        right_zone = np.where(right_theta * right_zone_x * right_zone_y)[0]

        zone_x, zone_y = lidar_xyz[right_zone][:,0], lidar_xyz[right_zone][:,1]

        empty_zone = np.round(self.right_zone.T[(distance.cdist(np.array([zone_x,zone_y]).T, self.right_zone.T).min(0) >= 1)])
        empty_point = np.unique(empty_zone, axis=0)
        obstacle = np.round(np.dot(np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)], [np.sin(np.pi/2), np.cos(np.pi/2)]]), empty_point.T).T)

        if obstacle.shape[0] != 0:
            obs_dist = distance.cdist(np.array([[0, 0]]), obstacle).min() - 2
            obs_ind  = distance.cdist(np.array([[0, 0]]), obstacle).argmin()
            obs_theta = np.rad2deg(np.arctan2(np.array([0,1]), obstacle[distance.cdist(np.array([[0,1]]), obstacle).argmin()]))
            obs_theta[obs_theta >= 180] -= 180
            obs_theta = obs_theta.max()
        else:
            obs_theta = -999
            obs_dist  = -999
        return right_zone, obstacle, obs_theta, obs_dist

    def callback(self, *args):
        out_msg = PointCloud2()
        filter_array = np.array([], dtype=np.int64)
        for i, callback in enumerate(self.callbacks):
            callback(args[i])
        if self.freespace_msg is not None:
            lidar_pc = ros_numpy.point_cloud2.pointcloud2_to_array(self.freespace_msg)
            lidar_xyz = ros_numpy.point_cloud2.get_xyz_points(lidar_pc, remove_nans=True)
            freespace = lidar_xyz[:,:2]

            ### find waypoint ###
            car_position = np.array([[self.car_waypoint_msg.pose.position.x, self.car_waypoint_msg.pose.position.y, self.car_waypoint_msg.pose.position.z]])
            car_position = car_position[:, :2]

            waypoint = self.to_narray(self.waypoint_msg.points)[:,:2]

            target_waypoint = self.to_narray(self.target_waypoint_msg.points)
            target_waypoint = target_waypoint[:, :2]

            nearest_waypoint = waypoint[np.argpartition(distance.cdist(car_position, waypoint).squeeze(), 2)[:2]] - car_position
            closest_index = distance.cdist(car_position, target_waypoint).argmin()

            target_waypoint = target_waypoint - car_position

            heading_vector = target_waypoint[closest_index+1] - target_waypoint[closest_index]

            heading_degree = np.radians(self.heading_v2_msg.data)
            heading_vector = np.dot(np.array([[np.cos(heading_degree), np.sin(heading_degree)], [-np.sin(heading_degree), np.cos(heading_degree)]]), np.array([0,1]))

            y_axis = heading_vector
            x_axis = np.dot(np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)], [np.sin(np.pi/2), np.cos(np.pi/2)]]), heading_vector)
            transformed_waypoint = np.dot(target_waypoint, np.array([x_axis,y_axis]))
            transformed_freespace = np.dot(np.array([[0,1],[-1,0]]), freespace.T).T

            #'''
            waypoint_zone = []
            for waypoint_x, waypoint_y in transformed_waypoint:
                theta = np.linspace(0, 2*np.pi, self.wp_interval)
                r = np.sqrt(self.radius)
                x1 = r*np.cos(theta)
                x2 = r*np.sin(theta)
                circle_zone = np.reshape(np.concatenate((x1 + waypoint_x, x2 + waypoint_y), axis=0), (self.wp_interval, 2), order='F')
                waypoint_zone.append(circle_zone)

            waypoint_zone = np.array(waypoint_zone).reshape(target_waypoint.shape[0]*self.wp_interval,2)
            zone_dist = distance.cdist(transformed_freespace, waypoint_zone)

            min_dist = []
            for n in range(target_waypoint.shape[0]):
                min_dist.append(zone_dist[:,self.wp_interval*n:self.wp_interval*(n+1)].min())
            min_dist = np.array(min_dist)
            #'''

            #min_dist = distance.cdist(transformed_freespace, transformed_waypoint).min(0)
            go_waypoint = (min_dist < 1).nonzero()[0]
            block_waypoint = (min_dist > 1).nonzero()[0]

            #print(go_waypoint)
            print("to where:", go_waypoint.max())
            #print("num:", len(go_waypoint))
            ### find waypoint ###

            ### safety zone ###
            #lidar_xyz = transformed_freespace[:,:2]
            x, y = lidar_xyz[:, 0], lidar_xyz[:, 1]
            theta = np.rad2deg(np.arctan2(x, y)) + 180 # 0 ~ 360 degrees
            theta = np.round(theta).astype(int)
            theta[theta >= 360] -= 360

            front_view, obstacle, obs_theta, obs_dist = self.set_front(lidar_xyz, x,y, theta)
            filter_array = np.append(filter_array, front_view)

            left_view, left_obs, left_theta, left_dist = self.set_left(lidar_xyz, x,y, theta)
            filter_array = np.append(filter_array, left_view)

            right_view, right_obs, right_theta, right_dist = self.set_right(lidar_xyz, x,y, theta)
            filter_array = np.append(filter_array, right_view)

            if obstacle.shape[0] != 0:
                safety_front = False
                safety_dist = round(obs_dist,2)
                print('dist:', round(obs_dist,2))
            else:
                safety_front = True
                safety_dist = self.front_h
            #self.pub_front.publish(safety_front)

            if left_obs.shape[0] != 0:
                safety_left = False
                print('left_dist:', left_dist)
            else:
                safety_left = True
            #self.pub_left.publish(safety_left)

            if right_obs.shape[0] != 0:
                safety_right = False
                print('right_dist:', right_dist)
            else:
                safety_right = True

            self.pub_front.publish(safety_front)
            self.pub_left.publish(safety_left)
            self.pub_right.publish(safety_right)
            self.pub_dist.publish(safety_dist)

            #print('no obstacle')
            out_msg = ros_numpy.point_cloud2.array_to_pointcloud2(lidar_pc[filter_array])
            out_msg.header.frame_id = self.frame_id

            self.pub_zone.publish(out_msg)
            #import IPython; IPython.embed()

if __name__ == '__main__':
    rospy.init_node('freespace_vis_v3', anonymous=True)
    publish_rate      = rospy.get_param('~publish_rate' ,     10)
    frame_id          = rospy.get_param('~frame_id', 'base_link')

    car          = '/imcar/points_marker'
    waypoint     = '/map/waypoint_marker'
    target_waypoint = '/map/target_waypoint_marker'
    freespace   = '/os_cloud_node/freespace'
    points_ring = '/os_cloud_node/freespace_vis'
    heading_v2  = '/estimated_yaw'

    publisher = Safety(publish_rate, freespace, target_waypoint, waypoint, car, points_ring, heading_v2, frame_id)

    rospy.spin()
