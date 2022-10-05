#!/home/imlab/.miniconda3/envs/carla/bin/python
from geodesy.utm import fromLatLong as proj
import numpy as np
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import NavSatFix
from path_generator import path_generator
from geometry_msgs.msg import Point, Quaternion, Vector3
from std_msgs.msg import ColorRGBA, Float64MultiArray

PG = path_generator()
waypoints = []
center = proj(35.64838, 128.40105, 0)

target_update_pub = rospy.Publisher("/map/target_update", MarkerArray, queue_size=10)

def get_target_waypoints(waypoints, current_x, current_y, n):
    closest_len = float('Inf')
    closest_index = 0

    for i in range(len(waypoints)):
        distance = np.sqrt((waypoints[i][0] - current_x) ** 2 + (waypoints[i][1] - current_y) ** 2)
        if distance <= closest_len:
            closest_len = distance
            closest_index = i

    end_index = np.min([closest_index + n, len(waypoints)])

    return waypoints[closest_index:end_index]

def gps_to_utm(latitude, longitude, altitude):
    pos = proj(latitude, longitude, altitude)
    pos = np.array([pos.easting, pos.northing, pos.altitude])
    pos[:2] -= center[:2]
    return pos

def callback_map_target(message):
    global waypoints
    waypoints = []
    for m in message.markers:
        waypoints.append([m.points[0].x, m.points[0].y, 8.333])

def callback(gps_sub):
    out_msg = MarkerArray()
    #node_msg = Float64MultiArray()
    point = PG.gps_to_utm(gps_sub.latitude, gps_sub.longitude, 0)
    target_waypoints = get_target_waypoints(waypoints, point[0], point[1], 20)

    #if target_waypoints is not None:
        #marker = Marker()

    for target_w in target_waypoints:
        marker = Marker()
        marker.pose.position = Point(x=target_w[0], y=target_w[1], z=target_w[-1])
        marker.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
        marker.type = Marker.SPHERE
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.scale = Vector3(x=1.5, y=1.5, z=1.5)
        #marker.id = i
        marker.color = ColorRGBA(0, 1, 1, 1)
        out_msg.markers.append(marker)
    #node_msg.data = np.array(target_waypoints[0])

    target_update_pub.publish(out_msg)
    #print(target_waypoints)
def getRosFnc():
    rospy.init_node('getRosFnc_main', anonymous=True)
    rospy.Subscriber('/map/target', MarkerArray, callback_map_target)
    rospy.Subscriber('/fix', NavSatFix, callback)
    rospy.spin()

if __name__ == '__main__':
    getRosFnc()
