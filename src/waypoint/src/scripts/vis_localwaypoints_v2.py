import numpy as np
import math
# from pyproj import Proj, transform
import csv
import rospy
import message_filters
from sensor_msgs.msg import PointCloud2, NavSatFix
from std_msgs.msg import ColorRGBA, Float64
from geometry_msgs.msg import Transform, Pose, Point, Vector3, PoseStamped, PoseWithCovarianceStamped
from geodesy.utm import fromLatLong as proj
from novatel_gps_msgs.msg import NovatelHeading2, NovatelPosition
from visualization_msgs.msg import Marker, MarkerArray
from tf import TransformBroadcaster as TB
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import Quaternion
from path_generator import path_generator
from ackermann_msgs.msg import AckermannDriveStamped
from math import sin, cos, pi, sqrt

center = proj(36.01300, 129.3198799, 0)
center = np.array([center.easting, center.northing, center.altitude])

g_heading = 999
g_heading_update_flag = False
g_start_waypoint_index = 1
g_prev_yaw = 999

rviz_points = Marker()
rviz_target_points = Marker()
PG = path_generator()

rospy.init_node("waypoint_marker")

marker_pub = rospy.Publisher("/map/waypoint_marker", Marker, queue_size=10)
target_marker_pub = rospy.Publisher("/map/target_waypoint_marker", Marker, queue_size=10)
gps_point_publisher = rospy.Publisher('/imcar/points_marker', Marker, queue_size=10)
yaw_publisher = rospy.Publisher('estimated_yaw', Float64, queue_size=10)

tf_pub = TB()

''' UTM '''
# def gps_to_utm(latitude, longitude, altitude):
#     pos = proj(latitude, longitude, altitude)
#     pos = np.array([pos.easting, pos.northing, pos.altitude])
#     pos[:2] -= center[:2]
#     return pos

''' transforms '''
def location_to_ros_point(imcar_location):
    ros_point = Point()
    ros_point.x = imcar_location.x
    # ros_point.y = -imcar_location.y
    ros_point.y = imcar_location.y
    ros_point.z = imcar_location.z

    return ros_point

def location_to_pose(imcar_location, g_speed, g_steer):
    global g_heading_update_flag
    global g_prev_yaw

    ros_pose = PoseStamped()
    ros_pose.header.stamp = rospy.Time.now()
    ros_pose.header.frame_id = 'map'

    ros_pose.pose.position = location_to_ros_point(imcar_location)

    if g_heading_update_flag:
        yaw = -math.radians(g_heading)
        q = quaternion_from_euler(0, 0, yaw + 1.5707)
        ros_pose.pose.orientation = Quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        _, _, g_euler_yaw = euler_from_quaternion(
            [ros_pose.pose.orientation.x, ros_pose.pose.orientation.y, ros_pose.pose.orientation.z,
             ros_pose.pose.orientation.w])
        g_prev_yaw = g_euler_yaw
        g_heading_update_flag = False
        return ros_pose, g_euler_yaw, 'gps_yaw'

    else:
        predicted_yaw = get_predicted_yaw(g_prev_yaw, g_speed / 3.6, np.radians(g_steer / 13.14), L=3, time_stamp=0.058)
        g_euler_yaw = predicted_yaw
        q = quaternion_from_euler(0, 0, g_euler_yaw)
        ros_pose.pose.orientation = Quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        g_prev_yaw = predicted_yaw

        return ros_pose, g_euler_yaw, 'pred_yaw'

def get_predicted_yaw(cur_yaw, v, steering_angle, L=3, time_stamp=0.25):
    return cur_yaw + v * np.tan(steering_angle) * time_stamp / L

# def location_to_pose(imcar_location, heading):
#     ros_pose = Pose()
#     ros_pose.position = location_to_ros_point(imcar_location)
#     yaw = -math.radians(heading)
#     rospy.loginfo('yaw: {} '.format(yaw))
#     q = quaternion_from_euler(0, 0, yaw + 1.5707)
#     ros_pose.orientation = Quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
#     # ros_pose.orientation.w = 1
#
#     return ros_pose

def pose_to_marker_msg(pose):
    marker_msg = Marker()
    marker_msg.type = 0
    marker_msg.header.frame_id = "map"
    marker_msg.pose = pose
    marker_msg.scale.x = 2.0
    marker_msg.scale.y = 1.0
    marker_msg.scale.z = 1.0
    marker_msg.color.r = 255.0
    marker_msg.color.a = 1.0
    return marker_msg

''' Callback '''
def callback_heading(message):
    global g_heading
    global g_heading_update_flag
    g_heading = message.heading
    g_heading_update_flag = True

def callback(gps_sub, can_sub):
    global g_start_waypoint_index
    global rviz_target_points

    lat = gps_sub.latitude
    lon = gps_sub.longitude
    alt = gps_sub.altitude
    g_steer = can_sub.drive.steering_angle
    g_speed = can_sub.drive.speed

    ''' current position publisher '''
    point = PG.gps_to_utm(lat, lon, 0)
    ros_translation = Vector3()
    ros_translation.x = point[0]
    ros_translation.y = point[1]
    ros_translation.z = 0
    # rospy.loginfo('points: {} ({})'.format(point, [lon, lat]))
    ros_pose, euler_yaw, yaw_str = location_to_pose(ros_translation, g_speed, g_steer)

    marker_msg = pose_to_marker_msg(ros_pose.pose)
    marker_msg.header.stamp = rospy.Time.now()
    gps_point_publisher.publish(marker_msg)

    ''' tf publisher '''
    tf_pub.sendTransform((ros_pose.pose.position.x, ros_pose.pose.position.y, ros_pose.pose.position.z),
 (ros_pose.pose.orientation.x, ros_pose.pose.orientation.y, ros_pose.pose.orientation.z, ros_pose.pose.orientation.w), rospy.Time.now(), 'base_link', 'map')

    ''' start_point_update '''
    g_start_waypoint_index = PG.get_closest_index([point[0], point[1], 0])
    goal_waypoint_index = int((g_start_waypoint_index + 50) % PG.get_total_waypoints())
    # rospy.loginfo('goal_waypoint_index: {} '.format(goal_waypoint_index))
    route_list = PG.generate_path(g_start_waypoint_index, goal_waypoint_index)

    target_points = []
    for i in range(len(route_list)):
        target_points.append(PG.gps_to_utm(route_list[i][1], route_list[i][0], 0))
    target_points = np.array(target_points)
    rviz_target_points = utm_to_rviz_points(target_points, 1, 0, 0, 0.5)

    ''' waypoint publisher '''
    rviz_points.header.stamp = rospy.Time.now()
    marker_pub.publish(rviz_points)
    rviz_target_points.header.stamp = rospy.Time.now()
    target_marker_pub.publish(rviz_target_points)

    ''' Yaw publisher '''
    euler_yaw = np.degrees(-(euler_yaw - 1.5707))
    if euler_yaw < 0:
        euler_yaw += 360.0

    yaw_publisher.publish(euler_yaw)

#     pub_transform()
#
# ''' def functions '''
# def pub_transform(ros_pose):

def callback_initpose(message):
    global g_start_waypoint_index
    g_start_waypoint_index = PG.get_closest_index([message.pose.pose.position.x, message.pose.pose.position.y, 0])
    # rospy.loginfo('g_start_waypoint_index: {} '.format(g_start_waypoint_index))

def callback_goalpose(message):
    global rviz_target_points
    goal_waypoint_index = PG.get_closest_index([message.pose.position.x, message.pose.position.y, 0])
    route_list = PG.generate_path(g_start_waypoint_index, goal_waypoint_index)

    target_points = []
    for i in range(len(route_list)):
        target_points.append(PG.gps_to_utm(route_list[i][1], route_list[i][0], 0))

    target_points = np.array(target_points)
    # rospy.loginfo('target_points: {} '.format(target_points))

    rviz_target_points = utm_to_rviz_points(target_points, 1, 0, 0)

def get_ros_fnc():

    rospy.Subscriber('/heading2', NovatelHeading2, callback_heading)
    rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, callback_initpose)
    rospy.Subscriber('/move_base_simple/goal', PoseStamped, callback_goalpose)

    gps_sub = message_filters.Subscriber('/fix', NavSatFix)
    can_sub = message_filters.Subscriber('/can_data', AckermannDriveStamped)

    ts = message_filters.ApproximateTimeSynchronizer([gps_sub, can_sub], 10, 0.1, allow_headerless=True)
    ts.registerCallback(callback)

    rospy.spin()

def utm_to_rviz_points(utm_points, R = 1, G = 1, B = 0, marker_size=0.2):
    rviz_points = Marker()

    rviz_points.header.frame_id = "map"
    rviz_points.ns = "points"
    rviz_points.id = 1

    rviz_points.type = Marker.POINTS
    rviz_points.action = Marker.ADD

    rviz_points.color = ColorRGBA(R, G, B, 1)
    rviz_points.scale = Vector3(marker_size, marker_size, 0)

    for i in range(len(utm_points)):
        rviz_points.points.append(Point(utm_points[i][0], utm_points[i][1], 0))

    return rviz_points


if __name__ == '__main__':
    rviz_points = utm_to_rviz_points(PG.get_points()) # lat, lon, alt
    # rospy.loginfo('PG.get_points(): {} '.format(PG.get_points()))
    # rospy.loginfo('rviz_points: {} '.format(rviz_points))
    get_ros_fnc()


