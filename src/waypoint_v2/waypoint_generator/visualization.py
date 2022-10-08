import numpy as np
import math
import csv
import rospy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import ColorRGBA, Int32MultiArray
from geometry_msgs.msg import Point, Vector3, PoseStamped, PoseWithCovarianceStamped
from novatel_gps_msgs.msg import NovatelHeading2
from visualization_msgs.msg import Marker, MarkerArray
from tf import TransformBroadcaster as TB
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import Quaternion
from path_generator import path_generator
import message_filters
from ackermann_msgs.msg import AckermannDriveStamped
from argparse import ArgumentParser
from intersection import stop_to_intersectionID_and_signalGroup
g_heading = 999
g_heading_update_flag = False
g_prev_yaw = 999

source_node_index = None
destination_node_index = None
nodes = None
edges = None
target_node_ids = []
target_nodes = None
node_id_to_marker_index = {}
changes = None
stops = None
dists = None
dist_stop = None
dist_target = None
eventState = -1
minEndTime = -1

rospy.init_node("waypoint_marker")
node_pub = rospy.Publisher("/map/node", MarkerArray, queue_size=10, latch=True)
target_pub = rospy.Publisher("/map/target", MarkerArray, queue_size=10, latch=True)
edge_pub = rospy.Publisher("/map/edge", MarkerArray, queue_size=10, latch=True)
stop_pub = rospy.Publisher("/map/stop", MarkerArray, queue_size=10)
gps_pub = rospy.Publisher('/map/gps', Marker, queue_size=10)
v2x_map_pub = rospy.Publisher('/v2x/map', Int32MultiArray, queue_size=10)
gps_point_publisher = rospy.Publisher('/imcar/points_marker', Marker, queue_size=10)
tf_pub = TB()

gps_marker = None

def location_to_ros_point(imcar_location):
    ros_point = Point()
    ros_point.x = imcar_location.x
    ros_point.y = imcar_location.y
    ros_point.z = imcar_location.z
    return ros_point

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

    else:
        predicted_yaw = get_predicted_yaw(g_prev_yaw, g_speed / 3.6, np.radians(g_steer / 13.14), L=3, time_stamp=0.058)
        g_euler_yaw = predicted_yaw
        q = quaternion_from_euler(0, 0, g_euler_yaw)
        ros_pose.pose.orientation = Quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3]))
        g_prev_yaw = predicted_yaw

    marker_msg = pose_to_marker_msg(ros_pose.pose)
    gps_point_publisher.publish(marker_msg)
    return ros_pose, g_euler_yaw

def get_predicted_yaw(cur_yaw, v, steering_angle, L=3, time_stamp=0.25):
    return cur_yaw + v * np.tan(steering_angle) * time_stamp / L

def gps_to_gps_marker(point):
    marker = Marker()
    marker.id = 0
    marker.type = Marker.POINTS
    marker.action = Marker.ADD
    marker.header.frame_id = "map"
    marker.ns = "gps"
    marker.scale = Vector3(1, 1, 0)
    marker.color = ColorRGBA(0, 0, 1, 1)
    marker.points.append(Point(point[0], point[1],point[2]))
    return marker

# def stop_line_to_stop_markerarray():
#     nodes = MarkerArray()
#     cnt = 0

#     for node in PG.stop_lines:
#         marker = Marker()
#         marker.id = cnt
#         marker.type = Marker.POINTS
#         marker.action = Marker.ADD
#         marker.header.frame_id = "/map"
#         marker.ns = "stops"
#         marker.scale = Vector3(1, 1, 0)
#         marker.color = ColorRGBA(0, 0, 1, 1)
#         lat, lon =node[0], node[1]
#         x, y, z = PG.gps_to_utm(lat, lon, 0)
#         marker.points.append(Point(x, y, z))
#         nodes.markers.append(marker)
#         cnt += 1

#     return nodes

def graph_to_node_markerarray():
    global node_id_to_marker_index

    nodes = MarkerArray()
    node_id_list = list(PG.G.nodes())
    for i, n_id in enumerate(node_id_list):
        marker = Marker()
        marker.id = n_id
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.header.frame_id = "map"
        marker.ns = "nodes"
        marker.scale = Vector3(1, 1, 0)
        # marker.color = ColorRGBA(1, 1, 0, 1)
        if PG.G.nodes[n_id]['stop']=='STOP':
            marker.color = ColorRGBA(1, 0, 0, 1)
        elif PG.G.nodes[n_id]['stop']=='TEST':
            marker.color = ColorRGBA(0, 0, 1, 1)
        elif PG.G.nodes[n_id]['stop']=='CROSS':
            marker.color = ColorRGBA(1, 1, 1, 1)
        else:
            marker.color = ColorRGBA(1, 1, 0, 1)
        lat, lon = PG.G.nodes[n_id]['vertex']
        x, y, z = PG.gps_to_utm(lat, lon, 0)
        marker.points.append(Point(x, y, z))
        nodes.markers.append(marker)
        node_id_to_marker_index[n_id]=i

    return nodes

def graph_to_edge_markerarray():
    edges = MarkerArray()
    for i, (s, d) in enumerate(PG.G.edges):
        start_lat, start_lon = PG.G.nodes[s]['vertex']
        start = PG.gps_to_utm(start_lat, start_lon, 0)
        start_point = Point(start[0], start[1], start[2])

        end_lat, end_lon = PG.G.nodes[d]['vertex']
        end = PG.gps_to_utm(end_lat, end_lon, 0)
        end_point = Point(end[0], end[1], end[2])

        marker = Marker()
        marker.id = i
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.header.frame_id = "map"
        marker.ns = "edges"
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.points.append(start_point)
        marker.points.append(end_point)

        edges.markers.append(marker)

    rospy.loginfo('# of nodes: {}\t# of edges: {}'.format(len(PG.G.nodes), len(PG.G.edges)))
    return edges


def target_node_to_node_markerarray(target_nodes, changes):
    nodes = MarkerArray()
    for i, node in enumerate(target_nodes):
        marker = Marker()
        marker.id = i
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.header.frame_id = "map"
        marker.ns = "targets"
        marker.scale = Vector3(1, 1, 0)
        marker.color = ColorRGBA(1, 0, 0, 1)
        marker.text = changes[i]
        lat, lon = node
        x, y, z = PG.gps_to_utm(lat, lon, 0)
        marker.points.append(Point(x, y, z))
        nodes.markers.append(marker)
    return nodes



def callback_heading(message):
    global g_heading
    global g_heading_update_flag
    g_heading = message.heading
    g_heading_update_flag = True

def callback(gps_sub, can_sub):
    global source_node_index, destination_node_index
    global target_nodes
    global dist_target, dist_stop
    global gps_marker
    global f

    lat = gps_sub.latitude
    lon = gps_sub.longitude
    alt = gps_sub.altitude
    steer = can_sub.drive.steering_angle
    speed = can_sub.drive.speed

    if g_heading != 999:
        ''' current position publisher '''
        point = PG.gps_to_utm(lat, lon, 0)

        if gps_marker is None:
            gps_marker = gps_to_gps_marker(point)
        else:
            gps_marker.points[0] = Point(point[0], point[1],point[2])

        gps_marker.text='-1.0,-1.0,-1.0'

        if source_node_index is not None:
            if PG.G.nodes[source_node_index]['stop']=='STOP':
                color = ColorRGBA(1, 0, 0, 1)
            elif PG.G.nodes[source_node_index]['stop']=='TEST':
                color = ColorRGBA(1, 0, 1, 1)
            elif PG.G.nodes[source_node_index]['stop']=='CROSS':
                color = ColorRGBA(1, 1, 1, 1)
            else:
                color = ColorRGBA(1, 1, 0, 1)
            nodes.markers[node_id_to_marker_index[source_node_index]].color = color
        source_node_index = PG.get_closest_index(point)
        nodes.markers[node_id_to_marker_index[source_node_index]].color = ColorRGBA(0, 0, 1, 1)

        v2x_map_msg = Int32MultiArray(data=[-1, -1, -1])
        if destination_node_index is not None and target_nodes is not None:
            idx_current = PG.get_closest_index_among(np.array([lat, lon]), target_nodes)
            idx_stop = (idx_current+stops[idx_current:].index('STOP')) if 'STOP' in stops[idx_current:] else -1
            MIN_NUM_NODES = 10

            if len(target_nodes) < 1+MIN_NUM_NODES:
                dist_target = PG.latlon_dist(np.array([lat, lon]),np.array(target_nodes[-1]))
                dist_stop = PG.latlon_dist(np.array([lat, lon]),np.array(target_nodes[idx_stop])) if idx_stop!=-1 else -1
            elif idx_current < len(target_nodes)-MIN_NUM_NODES:
                dist_target = PG.latlon_dist(np.array([lat, lon]),np.array(target_nodes[idx_current+1])) + sum(dists[idx_current+1:])
                if idx_stop == len(target_nodes)-MIN_NUM_NODES:
                    dist_stop = PG.latlon_dist(np.array([lat, lon]),np.array(target_nodes[idx_stop])) if idx_stop!=-1 else -1
                else:
                    dist_stop = PG.latlon_dist(np.array([lat, lon]),np.array(target_nodes[idx_current+1])) + sum(dists[idx_current+1:(idx_stop+1)]) if idx_stop!=-1 else -1
            else:
                dist_target = PG.latlon_dist(np.array([lat, lon]),np.array(target_nodes[-1]))
                dist_stop = PG.latlon_dist(np.array([lat, lon]),np.array(target_nodes[idx_stop])) if idx_stop!=-1 else -1

            if idx_stop!=-1 and tuple(target_nodes[idx_stop]) in stop_to_intersectionID_and_signalGroup.keys():
                v2x_map_msg = Int32MultiArray(data=stop_to_intersectionID_and_signalGroup[tuple(target_nodes[idx_stop])])
            gps_marker.text = str(dist_stop)+','+str(eventState)+','+str(minEndTime)+','+str(dist_target)

        print('dist_stop:{}\teventState:{}\tminEndTime:{}\tdist_target:{}'.format(dist_stop, eventState, minEndTime, dist_target))
        gps_pub.publish(gps_marker)
        v2x_map_pub.publish(v2x_map_msg)

        ''' pub orientation '''
        ros_translation = Vector3()
        ros_translation.x = point[0]
        ros_translation.y = point[1]
        ros_translation.z = 0
        ros_pose, euler_yaw = location_to_pose(ros_translation, speed, steer)
        tf_pub.sendTransform((ros_pose.pose.position.x, ros_pose.pose.position.y, ros_pose.pose.position.z), (ros_pose.pose.orientation.x, ros_pose.pose.orientation.y, ros_pose.pose.orientation.z, ros_pose.pose.orientation.w),
                            rospy.Time.now(), 'base_link', 'map')

def callback_initpose(message):
    global source_node_index
    global nodes
    # global stops

    if source_node_index is not None:
        if PG.G.nodes[source_node_index]['stop']=='STOP':
            color = ColorRGBA(1, 0, 0, 1)
        elif PG.G.nodes[source_node_index]['stop']=='TEST':
            color = ColorRGBA(1, 0, 1, 1)
        elif PG.G.nodes[source_node_index]['stop']=='CROSS':
            color = ColorRGBA(1, 1, 1, 1)
        else:
            color = ColorRGBA(1, 1, 0, 1)
        nodes.markers[node_id_to_marker_index[source_node_index]].color = color
    source_node_index = PG.get_closest_index([message.pose.pose.position.x, message.pose.pose.position.y, 0])
    nodes.markers[node_id_to_marker_index[source_node_index]].color = ColorRGBA(0, 0, 1, 1)

    node_pub.publish(nodes)
    # stop_pub.publish(stops)

# def callback_initpose(message):
#     global nodes
#     # global stops

#     source_node_index = PG.get_closest_index([message.pose.pose.position.x, message.pose.pose.position.y, 0])
#     nodes.markers[node_id_to_marker_index[source_node_index]].color = ColorRGBA(1, 0, 0, 1)

#     node_pub.publish(nodes)
#     # stop_pub.publish(stops)


def callback_goalpose(message):
    global source_node_index, destination_node_index
    global target_nodes, target_node_ids, changes, stops, dists
    global nodes # edges
    global node_id_to_marker_index

    destination_node_index = PG.get_closest_index([message.pose.position.x, message.pose.position.y, 0])
    for n_id in target_node_ids:
        if PG.G.nodes[n_id]['stop']=='STOP':
            color = ColorRGBA(1, 0, 0, 1)
        elif PG.G.nodes[n_id]['stop']=='TEST':
            color = ColorRGBA(1, 0, 1, 1)
        elif PG.G.nodes[n_id]['stop']=='CROSS':
            color = ColorRGBA(1, 1, 1, 1)
        else:
            color = ColorRGBA(1, 1, 0, 1)
        nodes.markers[node_id_to_marker_index[n_id]].color = color
    target_nodes, target_node_ids, changes, stops, dists= PG.generate_path(source_node_index, destination_node_index)
    for n_id in target_node_ids:
        nodes.markers[node_id_to_marker_index[n_id]].color = ColorRGBA(0, 0, 1, 1)

    targets = target_node_to_node_markerarray(target_nodes, changes)

    node_pub.publish(nodes)
    target_pub.publish(targets)
    rospy.loginfo('Source:{} / Destination:{}'.format(source_node_index, destination_node_index))

def callback_spat(message):
    global eventState
    global minEndTime
    eventState, minEndTime = message.data

def get_ros_fnc():
    rospy.Subscriber('/heading2', NovatelHeading2, callback_heading)
    rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, callback_initpose)
    rospy.Subscriber('/move_base_simple/goal', PoseStamped, callback_goalpose)
    rospy.Subscriber('/v2x/spat', Int32MultiArray, callback_spat)
    gps_sub = message_filters.Subscriber('/fix', NavSatFix)
    can_sub = message_filters.Subscriber('/can_data', AckermannDriveStamped)
    ts = message_filters.ApproximateTimeSynchronizer([gps_sub, can_sub], 10, 0.1, allow_headerless=True)
    ts.registerCallback(callback)

    rospy.spin()
    rospy.spin()


if __name__ == '__main__':
    parser = ArgumentParser(description='Waypoint Generator')
    parser.add_argument('--type', type=str, default='A_4', choices=['A_4','B1_7','B2_7'])
    parser.add_argument('--crossroad', action='store_true')
    parser.add_argument('--intersection', type=str, default='')

    args = parser.parse_args()
    PG = path_generator(type = args.type, crossroad = args.crossroad, intersection = args.intersection)
    nodes = graph_to_node_markerarray()
    node_pub.publish(nodes)
    edges = graph_to_edge_markerarray()
    edge_pub.publish(edges)
    # stops = stop_line_to_stop_markerarray()
    # stop_pub.publish(stops)

    get_ros_fnc()
