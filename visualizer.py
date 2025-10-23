import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import Path
import numpy as np
import tf
from typing import Dict, List, Tuple
import logging
from threading import Lock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self):
        # Publishers
        self.prediction_pub = rospy.Publisher(
            'visualization/predictions', MarkerArray, queue_size=1)
        self.path_pub = rospy.Publisher(
            'visualization/path', MarkerArray, queue_size=1)
        self.obstacle_pub = rospy.Publisher(
            'visualization/obstacles', MarkerArray, queue_size=1)
        self.safety_pub = rospy.Publisher(
            'visualization/safety_zones', MarkerArray, queue_size=1)
        
        # 시각화 파라미터
        self.prediction_lifetime = rospy.Duration(0.5)  # 예측 마커 수명
        self.safety_margin = 0.5  # 안전 영역 반경
        
        # Color definitions
        self.colors = {
            'prediction': ColorRGBA(0.0, 1.0, 0.0, 0.6),  # 초록색
            'path': ColorRGBA(0.0, 0.0, 1.0, 0.8),       # 파란색
            'obstacle': ColorRGBA(1.0, 0.0, 0.0, 0.8),   # 빨간색
            'safety': ColorRGBA(1.0, 1.0, 0.0, 0.3)      # 노란색
        }
        
        self.lock = Lock()

    def update(self, pose: np.ndarray = None, predictions: Dict = None, 
               path: Path = None, obstacles: List = None):
        """전체 시각화 업데이트"""
        with self.lock:
            try:
                if predictions:
                    self.visualize_predictions(predictions)
                if path:
                    self.visualize_path(path)
                if obstacles:
                    self.visualize_obstacles(obstacles)
                if pose is not None and predictions:
                    self.visualize_safety_zones(pose, predictions)
            except Exception as e:
                logger.error(f"Error in visualization update: {e}")

    def visualize_predictions(self, predictions: Dict):
        """보행자 예측 궤적 시각화"""
        try:
            marker_array = MarkerArray()
            
            for ped_id, pred_data in predictions.items():
                # 궤적 라인
                trajectory_marker = Marker()
                trajectory_marker.header.frame_id = "map"
                trajectory_marker.header.stamp = rospy.Time.now()
                trajectory_marker.ns = "pedestrian_trajectories"
                trajectory_marker.id = ped_id
                trajectory_marker.type = Marker.LINE_STRIP
                trajectory_marker.action = Marker.ADD
                trajectory_marker.scale = Vector3(0.05, 0.0, 0.0)
                trajectory_marker.color = self.colors['prediction']
                trajectory_marker.lifetime = self.prediction_lifetime
                
                # 궤적 포인트 추가
                for point in pred_data['trajectory']:
                    p = Point()
                    p.x = point[0]
                    p.y = point[1]
                    p.z = 0.1
                    trajectory_marker.points.append(p)
                    
                marker_array.markers.append(trajectory_marker)
                
                # 불확실성 영역
                uncertainty_marker = self.create_uncertainty_marker(
                    pred_data['trajectory'],
                    pred_data['uncertainty'],
                    ped_id
                )
                marker_array.markers.append(uncertainty_marker)
                
            self.prediction_pub.publish(marker_array)
            
        except Exception as e:
            logger.error(f"Error visualizing predictions: {e}")

    def create_uncertainty_marker(self, trajectory: np.ndarray, 
                                uncertainty: np.ndarray, id_num: int) -> Marker:
        """불확실성 영역 마커 생성"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "prediction_uncertainty"
        marker.id = id_num
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale = Vector3(0.02, 0.0, 0.0)
        marker.color.a = 0.3
        marker.lifetime = self.prediction_lifetime
        
        # 불확실성 경계선 생성
        for i in range(len(trajectory)):
            std_dev = uncertainty[i]
            
            # 위쪽 경계
            p_upper = Point()
            p_upper.x = trajectory[i][0] + std_dev[0]
            p_upper.y = trajectory[i][1] + std_dev[1]
            p_upper.z = 0.1
            marker.points.append(p_upper)
            
        # 아래쪽 경계 (역순)
        for i in range(len(trajectory)-1, -1, -1):
            std_dev = uncertainty[i]
            
            p_lower = Point()
            p_lower.x = trajectory[i][0] - std_dev[0]
            p_lower.y = trajectory[i][1] - std_dev[1]
            p_lower.z = 0.1
            marker.points.append(p_lower)
            
        return marker

    def visualize_path(self, path: Path):
        """계획된 경로 시각화"""
        try:
            marker_array = MarkerArray()
            
            # 경로 라인
            path_marker = Marker()
            path_marker.header.frame_id = "map"
            path_marker.header.stamp = rospy.Time.now()
            path_marker.ns = "planned_path"
            path_marker.id = 0
            path_marker.type = Marker.LINE_STRIP
            path_marker.action = Marker.ADD
            path_marker.scale = Vector3(0.1, 0.0, 0.0)
            path_marker.color = self.colors['path']
            
            # 경로 포인트 추가
            for pose in path.poses:
                p = Point()
                p.x = pose.pose.position.x
                p.y = pose.pose.position.y
                p.z = 0.1
                path_marker.points.append(p)
                
            marker_array.markers.append(path_marker)
            
            # 방향 화살표
            arrows_marker = self.create_path_arrows(path)
            marker_array.markers.append(arrows_marker)
            
            self.path_pub.publish(marker_array)
            
        except Exception as e:
            logger.error(f"Error visualizing path: {e}")

    def create_path_arrows(self, path: Path) -> Marker:
        """경로 방향 화살표 마커 생성"""
        arrows_marker = Marker()
        arrows_marker.header.frame_id = "map"
        arrows_marker.header.stamp = rospy.Time.now()
        arrows_marker.ns = "path_directions"
        arrows_marker.id = 1
        arrows_marker.type = Marker.ARROW
        arrows_marker.action = Marker.ADD
        arrows_marker.scale = Vector3(0.3, 0.1, 0.1)
        arrows_marker.color = self.colors['path']
        
        # 일정 간격으로 화살표 추가
        step = max(1, len(path.poses) // 10)  # 최대 10개의 화살표
        for i in range(0, len(path.poses) - 1, step):
            start = path.poses[i].pose.position
            end = path.poses[i + 1].pose.position
            
            p_start = Point()
            p_start.x = start.x
            p_start.y = start.y
            p_start.z = 0.1
            
            p_end = Point()
            p_end.x = end.x
            p_end.y = end.y
            p_end.z = 0.1
            
            arrows_marker.points.append(p_start)
            arrows_marker.points.append(p_end)
            
        return arrows_marker

    def visualize_obstacles(self, obstacles: List[Tuple[float, float]]):
        """장애물 시각화"""
        try:
            marker_array = MarkerArray()
            
            for i, obstacle in enumerate(obstacles):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "obstacles"
                marker.id = i
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD
                marker.scale = Vector3(0.3, 0.3, 0.5)  # 장애물 크기
                marker.color = self.colors['obstacle']
                
                marker.pose.position.x = obstacle[0]
                marker.pose.position.y = obstacle[1]
                marker.pose.position.z = 0.25
                
                marker_array.markers.append(marker)
                
            self.obstacle_pub.publish(marker_array)
            
        except Exception as e:
            logger.error(f"Error visualizing obstacles: {e}")

    def visualize_safety_zones(self, robot_pose: np.ndarray, predictions: Dict):
        """안전 영역 시각화"""
        try:
            marker_array = MarkerArray()
            
            # 로봇 안전 영역
            robot_safety = Marker()
            robot_safety.header.frame_id = "map"
            robot_safety.header.stamp = rospy.Time.now()
            robot_safety.ns = "safety_zones"
            robot_safety.id = 0
            robot_safety.type = Marker.CYLINDER
            robot_safety.action = Marker.ADD
            robot_safety.scale = Vector3(
                self.safety_margin * 2,
                self.safety_margin * 2,
                0.1
            )
            robot_safety.color = self.colors['safety']
            
            robot_safety.pose.position.x = robot_pose[0, 3]
            robot_safety.pose.position.y = robot_pose[1, 3]
            robot_safety.pose.position.z = 0.05
            
            marker_array.markers.append(robot_safety)
            
            # 보행자 안전 영역
            for i, (ped_id, pred_data) in enumerate(predictions.items()):
                current_pos = pred_data['trajectory'][0]
                
                ped_safety = Marker()
                ped_safety.header.frame_id = "map"
                ped_safety.header.stamp = rospy.Time.now()
                ped_safety.ns = "safety_zones"
                ped_safety.id = i + 1
                ped_safety.type = Marker.CYLINDER
                ped_safety.action = Marker.ADD
                ped_safety.scale = Vector3(
                    self.safety_margin * 2,
                    self.safety_margin * 2,
                    0.1
                )
                ped_safety.color = self.colors['safety']
                
                ped_safety.pose.position.x = current_pos[0]
                ped_safety.pose.position.y = current_pos[1]
                ped_safety.pose.position.z = 0.05
                
                marker_array.markers.append(ped_safety)
                
            self.safety_pub.publish(marker_array)
            
        except Exception as e:
            logger.error(f"Error visualizing safety zones: {e}")