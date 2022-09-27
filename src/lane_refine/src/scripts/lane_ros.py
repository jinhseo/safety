#!/root/anaconda3/envs/carla_py2/bin/python2
import rospy
import message_filters as mf
import numpy as np
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import String, Header
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32MultiArray
from rospy.numpy_msg import numpy_msg
import lane_refine

class Lane_Topic:
    def __init__(self, size, pub_rate, lane_topic, frame_id):
        self.subs = []
        self.size = size
        self.image_h = size[0]
        self.image_w = size[1]
        self.frame_id = frame_id
        self.rate = rospy.Rate(pub_rate)

        self.lane_msg = ImageMsg()
        self.pub = rospy.Publisher('/autonomous/lane_refine', ImageMsg, queue_size=10)
        self.pub_rst = rospy.Publisher('/autonomous/lane_refine_rst', numpy_msg(Float32MultiArray), queue_size=10)
        self.sub_lane = mf.Subscriber(lane_topic, ImageMsg)

        subs = [self.sub_lane]
        callbacks = [self.callback_lane]

        self.subs = []
        self.callbacks = []

        self.save_idx = 0

        for sub, callback in zip(subs, callbacks):
            if sub is not None:
                self.subs.append(sub)
                self.callbacks.append(callback)

        self.ts = mf.ApproximateTimeSynchronizer(subs, 10, 0.1, allow_headerless=True)
        self.ts.registerCallback(self.callback)

        self.bridge = CvBridge()

        self.valid_cnt = 5
        self.valid_left_idx = 0
        self.valid_right_idx = 0
        self.left_valid_arr = np.zeros(self.valid_cnt)
        self.right_valid_arr = np.zeros(self.valid_cnt)


    def to_numpy(self, msg, type='gray'):
        img = msg.data
        img = np.frombuffer(img, dtype=np.uint8)
        img = img.reshape((msg.height, msg.width, -1))

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[-1] == 4:
            img = img[:, :, :-1]
        elif img.shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def callback_lane(self, lane_msg):
        self.lane_msg = lane_msg

    def callback(self, lanes_sub):
        now = rospy.Time.now()
        lane_signal = []
        img = np.zeros((self.image_h, self.image_w, 3)).astype(np.uint8)
        if self.sub_lane is not None:
            msg = self.to_numpy(lanes_sub)
            if msg is None:
                msg = np.zeros((self.image_h, self.image_w, 3)).astype(np.uint8)
            img[msg > 128] = 255
            img = img[0:360, :, :]

        lane_obj = lane_refine.Lane(orig_frame=img)
        lane_obj.plot_roi(plot=False, save_idx=self.save_idx)

        warped_frame = lane_obj.perspective_transform(frame=img, plot=False)
        histogram = lane_obj.calculate_histogram(plot=False)
        left_fit, right_fit = lane_obj.get_lane_line_indices_sliding_windows(plot=False)
        lane_obj.get_lane_line_previous_window(left_fit, right_fit, plot=False)
        frame_with_lane_lines = lane_obj.overlay_lane_lines(plot=False)
        left_curve, right_curve, left_signal, right_signal = lane_obj.calculate_curvature(print_to_terminal=False)
        left_offset, right_offset, center_offset = lane_obj.calculate_car_position(print_to_terminal=False)

        # print('Curvature: ', (left_curve + right_curve) / 2, ' Left: ', left_signal, ': ', left_offset, ' Right: ', right_signal, ': ', right_offset, ' Center_offset ', center_offset)
        # print(right_signal, left_signal)
        # print(left_offset, right_offset)
        # print(center_offset)

        img = frame_with_lane_lines

        msg = self.bridge.cv2_to_imgmsg(img)
        msg.header.frame_id = self.frame_id
        msg.header.stamp = now
        self.pub.publish(msg)

        self.left_valid_arr[self.valid_left_idx] = left_offset
        self.right_valid_arr[self.valid_right_idx] = right_offset

        curvature = (left_curve + right_curve) / 2
        if curvature == float('inf') or curvature == float("-inf") or abs(curvature) > 1000:
            self.valid_left_idx = 0
            self.valid_right_idx = 0

        if abs(left_offset) > 90 or abs(left_offset) < 10:
            self.valid_left_idx = 0

        if abs(right_offset) > 90 or abs(right_offset) < 10:
            self.valid_right_idx = 0

        if left_signal == False:
            self.valid_left_idx = 0

        if right_signal == False:
            self.valid_right_idx = 0

        if self.valid_left_idx + 1 == self.valid_cnt and self.valid_right_idx + 1 == self.valid_cnt:
            result_data = [1, np.mean(self.left_valid_arr), 1, np.mean(self.right_valid_arr), center_offset]
            result_data = np.array(result_data).astype(np.float32)
            self.pub_rst.publish(Float32MultiArray(data=result_data))
            self.valid_left_idx = 0
            self.valid_right_idx = 0
        elif self.valid_left_idx + 1 == self.valid_cnt:
            result_data = [1, np.mean(self.left_valid_arr), 0, 0, center_offset]
            result_data = np.array(result_data).astype(np.float32)
            self.pub_rst.publish(Float32MultiArray(data=result_data))
            self.valid_left_idx = 0
            self.valid_right_idx = 0
        elif self.valid_right_idx + 1 == self.valid_cnt:
            result_data = [0, 0, 1, np.mean(self.right_valid_arr), center_offset]
            result_data = np.array(result_data).astype(np.float32)
            self.pub_rst.publish(Float32MultiArray(data=result_data))
            self.valid_left_idx = 0
            self.valid_right_idx = 0

        self.valid_left_idx += 1
        self.valid_right_idx += 1

        # put_text = 'Curvature: {} \n Left: {}:{} \n Right: {}:{} \n Center: {}'.format((left_curve + right_curve) / 2, left_signal, left_offset,right_signal, right_offset, center_offset)
        # org = (40, 40)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img,put_text,org,font, 0.5,
        #             (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.imwrite('./outputs/{:0>5d}.jpg'.format(self.save_idx), img)
        # self.save_idx += 1

if __name__ == '__main__':
    rospy.init_node('lane_refine', anonymous=True)

    image_height      = rospy.get_param('~image_height',      480) #480
    image_width       = rospy.get_param('~image_width' ,      640) #640
    publish_rate      = rospy.get_param('~publish_rate' ,     10)

    lanes = '/camera1/detected_lanes'
    frame_id          = rospy.get_param('~frame_id', 'camera0')
    publisher = Lane_Topic((image_height, image_width), publish_rate, lanes, frame_id)

    rospy.spin()

