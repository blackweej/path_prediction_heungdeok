from aiohttp import web
import socketio
import logging
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from threading import Lock

from core.slam_system import SLAMSystem
from core.navigation_system import NavigationSystem
from prediction.pedestrian_system import PredictionSystem
from visualization.visualizer import Visualizer
from support import RecoveryManager, SystemState
from time_sync_manager import TimeSyncManager

class IntegratedServer:
    def __init__(self):
        # ROS 노드 초기화
        rospy.init_node('integrated_system', anonymous=True)
        
        # 시스템 컴포넌트 초기화
        self.slam_system = SLAMSystem()
        self.navigation_system = NavigationSystem()
        self.prediction_system = PredictionSystem()
        self.visualizer = Visualizer()
        
        # 복구 관리자 초기화
        self.recovery_manager = RecoveryManager()
        
        # 시간 동기화 관리자 초기화
        self.time_sync = TimeSyncManager(sync_window=0.1)  # 100ms 동기화 윈도우
        
        # SocketIO 서버 설정
        self.sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins=['*'])
        self.app = web.Application()
        self.sio.attach(self.app)
        
        # 데이터 관리
        self.data_lock = Lock()
        self.latest_data = {
            'lidar': None,
            'camera': None,
            'imu': None,
            'odom': None
        }
        
        # 시스템 상태
        self.system_status = {
            'initialized': False,
            'last_processing_time': 0,
            'processing_interval': 0.1  # 10Hz
        }
        
        # ROS Publishers
        self.setup_ros_publishers()
        
        # 이벤트 핸들러 설정
        self.setup_event_handlers()
        
        logging.info("Integrated server initialized")

    def setup_ros_publishers(self):
        """ROS 퍼블리셔 설정"""
        self.map_pub = rospy.Publisher('integrated/map', OccupancyGrid, queue_size=1)
        self.path_pub = rospy.Publisher('integrated/path', Path, queue_size=1)
        self.prediction_pub = rospy.Publisher('integrated/predictions', MarkerArray, queue_size=1)
        self.pose_pub = rospy.Publisher('integrated/robot_pose', PoseStamped, queue_size=1)

    async def handle_lidar_data(self, data):
        """LiDAR 데이터 처리"""
        try:
            timestamp = data['timestamp']
            self.time_sync.add_lidar_data(timestamp, data)
            await self.process_synced_data()
            
        except Exception as e:
            logging.error(f'Error processing lidar data: {e}')
            self.recovery_manager.handle_slam_failure('sensor_error', str(e))

    async def handle_camera_data(self, data):
        """카메라 데이터 처리"""
        try:
            timestamp = data['timestamp']
            self.time_sync.add_camera_data(timestamp, data)
            await self.process_synced_data()
            
        except Exception as e:
            logging.error(f'Error processing camera data: {e}')
            self.recovery_manager.handle_prediction_failure('sensor_error', str(e))

    async def handle_imu_data(self, data):
        """IMU 데이터 처리"""
        try:
            timestamp = data['timestamp']
            self.time_sync.add_imu_data(timestamp, data)
            await self.process_synced_data()
            
        except Exception as e:
            logging.error(f'Error processing IMU data: {e}')
            self.recovery_manager.handle_slam_failure('sensor_error', str(e))

    async def process_synced_data(self):
        """동기화된 데이터 처리"""
        try:
            synced_data = self.time_sync.get_synced_data()
            if not synced_data:
                return

            # 시간 차이 모니터링
            if synced_data['time_diff'] > 0.05:  # 50ms 이상 차이
                rospy.logwarn(f"High time difference between sensors: {synced_data['time_diff']*1000:.2f}ms")

            # SLAM 업데이트
            if not await self.update_slam(synced_data):
                if not self.recovery_manager.handle_slam_failure('processing_error', 'SLAM update failed'):
                    return

            # 예측 시스템 업데이트
            if not await self.update_prediction(synced_data):
                if not self.recovery_manager.handle_prediction_failure('processing_error', 'Prediction update failed'):
                    return

            # 내비게이션 업데이트
            if not await self.update_navigation():
                if not self.recovery_manager.handle_navigation_failure('processing_error', 'Navigation update failed'):
                    return

            # 성공적인 업데이트 후 실패 카운트 리셋
            self.recovery_manager.reset_failure_count('slam')
            self.recovery_manager.reset_failure_count('navigation')
            self.recovery_manager.reset_failure_count('prediction')

            # 시각화 업데이트
            await self.update_visualization()

            # 상태 브로드캐스트
            await self.broadcast_status()

        except Exception as e:
            logging.error(f'Error in data processing: {e}')
            self.recovery_manager.trigger_emergency_stop(str(e))

    async def update_slam(self, synced_data):
        """SLAM 시스템 업데이트"""
        try:
            pose = self.slam_system.process_scan(synced_data['lidar'])
            if pose is not None:
                # 맵 발행
                map_msg = self.slam_system.get_map()
                self.map_pub.publish(map_msg)
                
                # 포즈 발행
                self.pose_pub.publish(self.create_pose_message(pose))
                return True
            return False
            
        except Exception as e:
            logging.error(f"Error in SLAM update: {e}")
            return False

    async def update_prediction(self, synced_data):
        """예측 시스템 업데이트"""
        try:
            predictions = self.prediction_system.predict(synced_data)
            if predictions:
                prediction_markers = self.create_prediction_markers(predictions)
                self.prediction_pub.publish(prediction_markers)
                return True
            return False
            
        except Exception as e:
            logging.error(f"Error in prediction update: {e}")
            return False

    async def update_navigation(self):
        """내비게이션 시스템 업데이트"""
        try:
            if self.navigation_system.update():
                path = self.navigation_system.get_current_path()
                if path is not None:
                    self.path_pub.publish(path)
                return True
            return False
            
        except Exception as e:
            logging.error(f"Error in navigation update: {e}")
            return False

    async def update_visualization(self):
        """시각화 업데이트"""
        try:
            markers = self.visualizer.update(
                pose=self.slam_system.get_current_pose(),
                predictions=self.prediction_system.get_predictions(),
                path=self.navigation_system.get_current_path()
            )
            return True
            
        except Exception as e:
            logging.error(f"Error in visualization update: {e}")
            return False

    async def broadcast_status(self):
        """시스템 상태 브로드캐스트"""
        try:
            recovery_status = self.recovery_manager.get_system_status()
            
            status = {
                'system_state': recovery_status['state'],
                'slam': {
                    'status': 'active' if recovery_status['state'] == SystemState.NORMAL.value else 'recovery',
                    'failure_count': recovery_status['failure_counts']['slam']
                },
                'navigation': {
                    'status': self.navigation_system.navigation_state,
                    'failure_count': recovery_status['failure_counts']['navigation']
                },
                'prediction': {
                    'status': 'active' if recovery_status['state'] == SystemState.NORMAL.value else 'recovery',
                    'failure_count': recovery_status['failure_counts']['prediction']
                }
            }
            
            await self.sio.emit('system_status', status)
            
        except Exception as e:
            logging.error(f'Error broadcasting status: {e}')

    async def start(self):
        """서버 시작"""
        try:
            runner = web.AppRunner(self.app)
            await runner.setup()
            site = web.TCPSite(runner, 'localhost', 3000)
            await site.start()
            logging.info("Server started on http://localhost:3000")
            
            # 시스템 초기화 완료
            self.system_status['initialized'] = True
            
        except Exception as e:
            logging.error(f"Failed to start server: {e}")
            raise

async def main():
    server = IntegratedServer()
    await server.start()
    
    try:
        while not rospy.is_shutdown():
            await asyncio.sleep(0.1)
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
