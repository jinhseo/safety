#!/root/anaconda3/envs/carla_py2/bin/python2
import rospy
import ros_numpy
import numpy as np
import message_filters as mf
from std_msgs.msg import String, Header, Float64, Bool, Float64MultiArray
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion, Vector3
from scipy.spatial import distance
from novatel_gps_msgs.msg import NovatelHeading2
from ackermann_msgs.msg import AckermannDriveStamped

class Safety:
    def __init__(self, pub_rate, heading_v2, can_data, safety, frame_id):
        self.frame_id = frame_id
        self.rate = rospy.Rate(pub_rate)

        #self.pub_pc_zone = rospy.Publisher('waypoint_zone', PointCloud2, queue_size=10)
        self.pub_array = rospy.Publisher('safety', Float64MultiArray, queue_size=10)

        self.sub_safety = mf.Subscriber(safety, Float64MultiArray)
        self.sub_heading_v2 = mf.Subscriber(heading_v2, Float64)
        self.sub_can_data = mf.Subscriber(can_data, AckermannDriveStamped)

        self.subs = []
        self.callbacks = []

        self.collect_dist = []
        self.collect_yaw = []
        self.start_time = rospy.Time.now().secs

        subs = [self.sub_safety, self.sub_heading_v2, self.sub_can_data]
        callbacks = [self.callback_safety, self.callback_heading_v2, self.callback_can_data]

        for sub, callback in zip(subs, callbacks):
            if sub is not None:
                self.subs.append(sub)
                self.callbacks.append(callback)
        self.ts = mf.ApproximateTimeSynchronizer(self.subs, 10, 0.1, allow_headerless=True)
        self.ts.registerCallback(self.callback)

    def callback_safety(self, safety_msg):
        self.safety_msg = safety_msg
    def callback_heading_v2(self, heading_v2_msg):
        self.heading_v2_msg = heading_v2_msg
    def callback_can_data(self, can_data_msg):
        self.can_data_msg = can_data_msg

    def callback(self, *args):
        for i, callback in enumerate(self.callbacks):
            callback(args[i])

        self.collect_dist.append(self.safety_msg.data[-1])
        self.collect_yaw.append(round(self.heading_v2_msg.data, 2))

        current_time = rospy.Time.now().secs
        if current_time == self.start_time + rospy.Time(1).secs:
            ### speed ###
            if len(self.collect_dist) == 1:
                estim_speed = 'none'
            else:
                dist_tau = sum(self.collect_dist) / len(self.collect_dist)
                for dist in self.collect_dist:
                    if (dist > dist_tau + 5) or (dist < dist_tau - 5):
                        self.collect_dist.remove(dist)
                estim_speed = self.collect_dist[-1] - self.collect_dist[0]
                self.start_time = rospy.Time.now().secs
                self.collect_dist = []
            ### speed ###

            ### curve ###
            change = self.collect_yaw[-1] - self.collect_yaw[0]
            if change < 10:
                print('straight')
            else:
                print('curve')
            self.collect_yaw = []
            ### curve ###
            print(estim_speed)
        #print(self.collect_dist)
        #import IPython; IPython.embed()

if __name__ == '__main__':
    rospy.init_node('freespace_vis_v3', anonymous=True)
    publish_rate      = rospy.get_param('~publish_rate' ,     10)
    frame_id          = rospy.get_param('~frame_id', 'base_link')

    heading_v2  = '/estimated_yaw'
    can_data = '/can_data'
    safety = '/safety'

    publisher = Safety(publish_rate, heading_v2, can_data, safety, frame_id)

    rospy.spin()
