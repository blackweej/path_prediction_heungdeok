import rospy
from sensor_msgs.msg import LaserScan, Image, Imu
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import socketio
import json
import numpy as np
import cv2
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ROSDataCollector:
    def __init__(self):
        # ROS 노드 초기화
        rospy.init_node('ros_data_collector')
        
        # 설정값
        self.IMAGE_QUALITY = 80
        self.IMAGE_MAX_DIMENSION = 800
        self.DATA_RATE = 0.1  # 10Hz
        
        # CV Bridge
        self.cv_bridge = CvBridge()
        
        # Socket.IO 클라이언트 설정
        self.sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=5,
            reconnection_delay=2
        )
        self.setup_socketio()
        
        # Subscribers 설정
        self.setup_subscribers()
        
        # 마지막 데이터 전송 시간 관리
        self.last_send_time = {
            'lidar': 0,
            'camera': 0,
            'imu': 0,
            'odom': 0
        }

    def setup_socketio(self):
        @self.sio.event
        def connect():
            logger.info('Connected to server')
            
        @self.sio.event
        def disconnect():
            logger.info('Disconnected from server')
            
        try:
            self.sio.connect('http://localhost:3000')
        except Exception as e:
            logger.error(f'Connection failed: {e}')

    def setup_subscribers(self):
        rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        rospy.Subscriber('image_topic', Image, self.camera_callback)
        rospy.Subscriber('/imu', Imu, self.imu_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
    def should_send_data(self, data_type):
        current_time = rospy.Time.now().to_sec()
        if current_time - self.last_send_time[data_type] >= self.DATA_RATE:
            self.last_send_time[data_type] = current_time
            return True
        return False

    def compress_image(self, cv_image):
        # 이미지 크기 조정
        height, width = cv_image.shape[:2]
        if max(height, width) > self.IMAGE_MAX_DIMENSION:
            scale = self.IMAGE_MAX_DIMENSION / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            cv_image = cv2.resize(cv_image, (new_width, new_height))
            
        # JPEG 압축
        _, jpeg_data = cv2.imencode('.jpg', cv_image, 
                                  [cv2.IMWRITE_JPEG_QUALITY, self.IMAGE_QUALITY])
        return jpeg_data.tobytes()

    def lidar_callback(self, msg):
        if not self.should_send_data('lidar'):
            return
            
        try:
            data = {
                'timestamp': rospy.Time.now().to_sec(),
                'ranges': list(msg.ranges),
                'angle_min': msg.angle_min,
                'angle_max': msg.angle_max,
                'angle_increment': msg.angle_increment
            }
            self.sio.emit('lidar_data', data)
            
        except Exception as e:
            logger.error(f'Error in lidar callback: {e}')

    def camera_callback(self, msg):
        if not self.should_send_data('camera'):
            return
            
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            compressed_img = self.compress_image(cv_image)
            
            data = {
                'timestamp': rospy.Time.now().to_sec(),
                'image': compressed_img,
                'width': cv_image.shape[1],
                'height': cv_image.shape[0]
            }
            self.sio.emit('camera_data', data)
            
        except Exception as e:
            logger.error(f'Error in camera callback: {e}')

    def imu_callback(self, msg):
        if not self.should_send_data('imu'):
            return
            
        try:
            data = {
                'timestamp': rospy.Time.now().to_sec(),
                'orientation': {
                    'x': msg.orientation.x,
                    'y': msg.orientation.y,
                    'z': msg.orientation.z,
                    'w': msg.orientation.w
                },
                'angular_velocity': {
                    'x': msg.angular_velocity.x,
                    'y': msg.angular_velocity.y,
                    'z': msg.angular_velocity.z
                },
                'linear_acceleration': {
                    'x': msg.linear_acceleration.x,
                    'y': msg.linear_acceleration.y,
                    'z': msg.linear_acceleration.z
                }
            }
            self.sio.emit('imu_data', data)
            
        except Exception as e:
            logger.error(f'Error in imu callback: {e}')

    def odom_callback(self, msg):
        if not self.should_send_data('odom'):
            return
            
        try:
            data = {
                'timestamp': rospy.Time.now().to_sec(),
                'position': {
                    'x': msg.pose.pose.position.x,
                    'y': msg.pose.pose.position.y,
                    'z': msg.pose.pose.position.z
                },
                'orientation': {
                    'x': msg.pose.pose.orientation.x,
                    'y': msg.pose.pose.orientation.y,
                    'z': msg.pose.pose.orientation.z,
                    'w': msg.pose.pose.orientation.w
                }
            }
            self.sio.emit('odom_data', data)
            
        except Exception as e:
            logger.error(f'Error in odom callback: {e}')

    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            logger.info("Shutting down")
        finally:
            if self.sio.connected:
                self.sio.disconnect()

if __name__ == '__main__':
    try:
        collector = ROSDataCollector()
        collector.run()
    except rospy.ROSInterruptException:
        pass