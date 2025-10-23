import numpy as np
from typing import List, Tuple, Optional, Dict
import rospy
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from scipy.optimize import linear_sum_assignment
import tf2_ros
from scipy.spatial import KDTree
import logging
from threading import Lock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NavigationSystem:
    def __init__(self, config: Optional[dict] = None):
        """내비게이션 시스템 초기화"""
        # ROS 토픽 설정
        self._setup_ros_interface()
        
        # 시스템 상태 및 설정
        self.config = config or self.load_default_config()
        self.current_path = None
        self.goal_pose = None
        self.system_lock = Lock()
        
        # 지도 관련 데이터
        self.costmap = None
        self.pedestrian_map = None
        self.planning_resolution = 0.2
        
        # 보행자 예측 데이터
        self.pedestrian_predictions = {}
        self.prediction_horizon = 30  # 3초 (10Hz 기준)
        
        # 경로 계획 파라미터
        self.safety_margin = 0.5
        self.replanning_threshold = 0.3
        self.path_smoothing_factor = 0.1
        
        # 시스템 상태
        self.navigation_state = "IDLE"  # IDLE, PLANNING, ACTIVE, REPLANNING
        self.last_update_time = rospy.Time.now()

    def _setup_ros_interface(self):
        """ROS 인터페이스 설정"""
        # Publishers
        self.path_pub = rospy.Publisher('planned_path', Path, queue_size=1)
        self.local_path_pub = rospy.Publisher('local_path', Path, queue_size=1)
        self.markers_pub = rospy.Publisher('navigation_markers', MarkerArray, queue_size=1)
        
        # Subscribers
        rospy.Subscriber('costmap', OccupancyGrid, self.costmap_callback)
        rospy.Subscriber('pedestrian_predictions', MarkerArray, self.prediction_callback)
        
        # TF 설정
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def load_default_config(self) -> dict:
        """기본 설정값 로드"""
        return {
            'safety_margin': 0.5,
            'path_resolution': 0.2,
            'goal_tolerance': {
                'xy': 0.1,
                'yaw': 0.1
            },
            'planning': {
                'max_iterations': 1000,
                'timeout': 1.0,
                'smoothing_factor': 0.1
            },
            'replanning': {
                'min_distance': 0.3,
                'interval': 0.5
            }
        }

    def set_goal(self, goal: PoseStamped) -> bool:
        """목표점 설정 및 초기 경로 계획"""
        with self.system_lock:
            try:
                self.goal_pose = goal
                self.navigation_state = "PLANNING"
                success = self.plan_global_path()
                
                if success:
                    self.navigation_state = "ACTIVE"
                    self.publish_navigation_markers()
                    logger.info("New goal set and path planned successfully")
                else:
                    self.navigation_state = "IDLE"
                    logger.warning("Failed to plan path to new goal")
                
                return success
                
            except Exception as e:
                logger.error(f"Error in set_goal: {e}")
                self.navigation_state = "IDLE"
                return False

    def update(self):
        """내비게이션 시스템 주기적 업데이트"""
        if self.navigation_state not in ["ACTIVE", "REPLANNING"]:
            return

        try:
            current_time = rospy.Time.now()
            dt = (current_time - self.last_update_time).to_sec()
            
            if dt < self.config['replanning']['interval']:
                return
                
            self.last_update_time = current_time
            
            # 경로 안전성 검사
            if not self.check_path_safety():
                self.navigation_state = "REPLANNING"
                if not self.replan_path():
                    logger.warning("Failed to replan path")
                    return
                    
            # 로컬 경로 최적화
            self.optimize_local_path()
            
            # 시각화 업데이트
            self.publish_navigation_markers()
            
        except Exception as e:
            logger.error(f"Error in update: {e}")

    def plan_global_path(self) -> bool:
        """전역 경로 계획"""
        try:
            current_pose = self.get_current_pose()
            if current_pose is None or self.goal_pose is None:
                return False
                
            # 시작점과 목표점
            start = (current_pose.pose.position.x, current_pose.pose.position.y)
            goal = (self.goal_pose.pose.position.x, self.goal_pose.pose.position.y)
            
            # A* 경로 계획
            path_points = self.a_star_planning(start, goal)
            if not path_points:
                return False
                
            # 경로 후처리
            smoothed_path = self.post_process_path(path_points)
            
            # Path 메시지 생성 및 저장
            self.current_path = self.create_path_message(smoothed_path)
            self.path_pub.publish(self.current_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in global path planning: {e}")
            return False

    def a_star_planning(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """A* 알고리즘을 사용한 경로 계획"""
        try:
            # 그리드 맵 가져오기
            if self.costmap is None:
                logger.error("No costmap available")
                return []
                
            # 시작점과 목표점을 그리드 좌표로 변환
            start_grid = self.world_to_grid(start[0], start[1])
            goal_grid = self.world_to_grid(goal[0], goal[1])
            
            # A* 알고리즘 구현
            open_set = {start_grid}
            closed_set = set()
            came_from = {}
            
            g_score = {start_grid: 0}
            f_score = {start_grid: self.heuristic(start_grid, goal_grid)}
            
            while open_set:
                current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
                
                if current == goal_grid:
                    path = self.reconstruct_path(came_from, current)
                    return [self.grid_to_world(x, y) for x, y in path]
                    
                open_set.remove(current)
                closed_set.add(current)
                
                for neighbor in self.get_neighbors(current):
                    if neighbor in closed_set:
                        continue
                        
                    tentative_g_score = g_score[current] + \
                                    self.get_movement_cost(current, neighbor)
                    
                    if neighbor not in open_set:
                        open_set.add(neighbor)
                    elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                        continue
                        
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + \
                                    self.heuristic(neighbor, goal_grid)
            
            return []
            
        except Exception as e:
            logger.error(f"Error in A* planning: {e}")
            return []

    def get_neighbors(self, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """현재 셀의 이웃 셀 반환"""
        neighbors = []
        movements = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        for dx, dy in movements:
            x, y = current[0] + dx, current[1] + dy
            
            if self.is_valid_cell(x, y):
                neighbors.append((x, y))
                
        return neighbors

    def is_valid_cell(self, x: int, y: int) -> bool:
        """셀의 유효성 검사"""
        if self.costmap is None:
            return False
            
        if not (0 <= x < self.costmap.shape[0] and 0 <= y < self.costmap.shape[1]):
            return False
            
        return self.costmap[x, y] < 50  # 장애물이 아닌 경우

    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """휴리스틱 함수 (유클리드 거리)"""
        return np.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)

    def reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], 
                        current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """경로 재구성"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def create_path_message(self, path_points: List[Tuple[float, float]]) -> Path:
        """Path 메시지 생성"""
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()
        
        for point in path_points:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0.0
            
            # 방향 설정 (간단한 버전)
            pose.pose.orientation.w = 1.0
            
            path_msg.poses.append(pose)
        
        return path_msg


    def replan_path(self) -> bool:
        """경로 재계획"""
        logger.info("Initiating path replanning")
        
        try:
            if self.plan_global_path():
                self.navigation_state = "ACTIVE"
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error in path replanning: {e}")
            return False

    def optimize_local_path(self):
        """로컬 경로 최적화"""
        if self.current_path is None:
            return
            
        try:
            current_pose = self.get_current_pose()
            if current_pose is None:
                return
                
            # 현재 위치에서 가까운 경로 세그먼트 찾기
            closest_idx = self.find_closest_path_point(current_pose)
            
            # 로컬 경로 추출
            local_path_points = self.extract_local_path(closest_idx)
            
            # 동적 장애물 회피 최적화
            optimized_points = self.optimize_for_dynamic_obstacles(local_path_points)
            
            # 로컬 경로 발행
            local_path_msg = self.create_path_message(optimized_points)
            self.local_path_pub.publish(local_path_msg)
            
        except Exception as e:
            logger.error(f"Error in local path optimization: {e}")

    def check_path_safety(self) -> bool:
        """경로 안전성 검사"""
        if self.current_path is None:
            return True
            
        try:
            for pose in self.current_path.poses:
                position = np.array([pose.pose.position.x, pose.pose.position.y])
                
                # 정적 장애물 검사
                if not self.check_static_safety(position):
                    return False
                    
                # 동적 장애물 (보행자) 검사
                if not self.check_dynamic_safety(position):
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error in path safety check: {e}")
            return False

    def check_static_safety(self, position: np.ndarray) -> bool:
        """정적 장애물 안전성 검사"""
        if self.costmap is None:
            return True
            
        try:
            grid_pos = self.world_to_grid(position[0], position[1])
            if not (0 <= grid_pos[0] < self.costmap.shape[0] and 
                   0 <= grid_pos[1] < self.costmap.shape[1]):
                return False
                
            return self.costmap[grid_pos[0], grid_pos[1]] < 50
            
        except Exception as e:
            logger.error(f"Error in static safety check: {e}")
            return False

    def check_dynamic_safety(self, position: np.ndarray) -> bool:
        """동적 장애물 안전성 검사"""
        try:
            for pred_id, pred_data in self.pedestrian_predictions.items():
                for pred_pos in pred_data['trajectory']:
                    distance = np.linalg.norm(position - pred_pos[:2])
                    if distance < self.safety_margin:
                        return False
            return True
            
        except Exception as e:
            logger.error(f"Error in dynamic safety check: {e}")
            return False

    def publish_navigation_markers(self):
        """내비게이션 관련 마커 발행"""
        try:
            marker_array = MarkerArray()
            
            # 전역 경로 마커
            if self.current_path is not None:
                path_marker = self.create_path_marker(self.current_path, 'global_path', 
                                                    [0, 0, 1, 0.8])  # 파란색
                marker_array.markers.append(path_marker)
            
            # 보행자 예측 마커
            for ped_id, pred_data in self.pedestrian_predictions.items():
                pred_marker = self.create_prediction_marker(pred_data, ped_id)
                marker_array.markers.append(pred_marker)
            
            # 안전 영역 마커
            safety_markers = self.create_safety_markers()
            marker_array.markers.extend(safety_markers)
            
            self.markers_pub.publish(marker_array)
            
        except Exception as e:
            logger.error(f"Error publishing markers: {e}")

    def prediction_callback(self, msg: MarkerArray):
        """보행자 예측 데이터 콜백"""
        try:
            predictions = {}
            for marker in msg.markers:
                if marker.ns == "pedestrian_predictions":
                    ped_id = marker.id
                    trajectory = []
                    for point in marker.points:
                        trajectory.append(np.array([point.x, point.y]))
                    predictions[ped_id] = {
                        'trajectory': np.array(trajectory),
                        'timestamp': rospy.Time.now().to_sec()
                    }
            
            with self.system_lock:
                self.pedestrian_predictions = predictions
                
        except Exception as e:
            logger.error(f"Error in prediction callback: {e}")

    def costmap_callback(self, msg: OccupancyGrid):
        """코스트맵 업데이트 콜백"""
        try:
            costmap = np.array(msg.data).reshape(msg.info.height, msg.info.width)
            with self.system_lock:
                self.costmap = costmap
                self.map_resolution = msg.info.resolution
                self.map_origin = np.array([msg.info.origin.position.x, 
                                          msg.info.origin.position.y])
                
        except Exception as e:
            logger.error(f"Error in costmap callback: {e}")

    def get_current_pose(self) -> Optional[PoseStamped]:
        """현재 로봇의 위치 가져오기"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rospy.Time(0), rospy.Duration(1.0)
            )
            
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = rospy.Time.now()
            
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            
            pose.pose.orientation = transform.transform.rotation
            
            return pose
            
        except Exception as e:
            logger.error(f"Error getting current pose: {e}")
            return None

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """월드 좌표를 그리드 좌표로 변환"""
        grid_x = int((x - self.map_origin[0]) / self.map_resolution)
        grid_y = int((y - self.map_origin[1]) / self.map_resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x: int, grid_y: int) -> np.ndarray:
        """그리드 좌표를 월드 좌표로 변환"""
        world_x = grid_x * self.map_resolution + self.map_origin[0]
        world_y = grid_y * self.map_resolution + self.map_origin[1]
        return np.array([world_x, world_y])

    def find_closest_path_point(self, current_pose: PoseStamped) -> int:
        """현재 위치에서 가장 가까운 경로 포인트 찾기"""
        if self.current_path is None:
            return 0
            
        current_pos = np.array([current_pose.pose.position.x, 
                            current_pose.pose.position.y])
        
        min_dist = float('inf')
        closest_idx = 0
        
        for i, pose in enumerate(self.current_path.poses):
            path_pos = np.array([pose.pose.position.x, pose.pose.position.y])
            dist = np.linalg.norm(current_pos - path_pos)
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        return closest_idx

    def extract_local_path(self, center_idx: int, window_size: int = 20) -> List[np.ndarray]:
        """주어진 인덱스 주변의 로컬 경로 추출"""
        if self.current_path is None:
            return []
            
        start_idx = max(0, center_idx - window_size // 2)
        end_idx = min(len(self.current_path.poses), center_idx + window_size // 2)
        
        local_points = []
        for i in range(start_idx, end_idx):
            pose = self.current_path.poses[i]
            point = np.array([pose.pose.position.x, pose.pose.position.y])
            local_points.append(point)
            
        return local_points

    def post_process_path(self, path_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """경로 후처리 (평활화 및 간격 조정)"""
        if len(path_points) <= 2:
            return path_points
            
        # 경로 평활화
        smoothed = self.smooth_path(path_points)
        
        # 경로 포인트 리샘플링
        resampled = self.resample_path(smoothed)
        
        return resampled

    def smooth_path(self, path_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """경로 평활화"""
        if len(path_points) <= 2:
            return path_points
            
        alpha = self.config['planning']['smoothing_factor']
        smoothed = np.array(path_points)
        prev_smoothed = np.array(path_points)
        
        for _ in range(100):  # 최대 반복 횟수
            for i in range(1, len(smoothed) - 1):
                smoothed[i] = smoothed[i] + alpha * (
                    prev_smoothed[i-1] + prev_smoothed[i+1] - 2 * smoothed[i]
                )
                
            if np.allclose(smoothed, prev_smoothed, atol=0.001):
                break
                
            prev_smoothed = smoothed.copy()
            
        return smoothed.tolist()

    def resample_path(self, path_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """경로 리샘플링 (일정한 간격으로)"""
        if len(path_points) <= 2:
            return path_points
            
        target_distance = self.config['path_resolution']
        resampled = [path_points[0]]
        
        for i in range(1, len(path_points)):
            last_point = np.array(resampled[-1])
            current_point = np.array(path_points[i])
            
            distance = np.linalg.norm(current_point - last_point)
            if distance >= target_distance:
                direction = (current_point - last_point) / distance
                num_points = int(distance / target_distance)
                
                for j in range(1, num_points + 1):
                    new_point = last_point + direction * j * target_distance
                    resampled.append(new_point.tolist())
                    
        return resampled

    def create_path_marker(self, path: Path, ns: str, color: List[float]) -> Marker:
        """경로 시각화를 위한 마커 생성"""
        marker = Marker()
        marker.header = path.header
        marker.ns = ns
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        marker.scale.x = 0.1  # 선 굵기
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        
        marker.points = [pose.pose.position for pose in path.poses]
        
        return marker

    def create_safety_markers(self) -> List[Marker]:
        """안전 영역 시각화를 위한 마커 생성"""
        markers = []
        
        # 현재 위치 안전 영역
        current_pose = self.get_current_pose()
        if current_pose is not None:
            safety_marker = Marker()
            safety_marker.header.frame_id = "map"
            safety_marker.header.stamp = rospy.Time.now()
            safety_marker.ns = "safety_zones"
            safety_marker.id = 0
            safety_marker.type = Marker.CYLINDER
            safety_marker.action = Marker.ADD
            
            safety_marker.scale.x = self.safety_margin * 2
            safety_marker.scale.y = self.safety_margin * 2
            safety_marker.scale.z = 0.1
            
            safety_marker.color.r = 1.0
            safety_marker.color.g = 1.0
            safety_marker.color.a = 0.3
            
            safety_marker.pose = current_pose.pose
            markers.append(safety_marker)
        
        return markers

    def create_prediction_marker(self, pred_data: dict, ped_id: int) -> Marker:
        """보행자 예측 궤적을 위한 마커 생성"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "pedestrian_predictions"
        marker.id = ped_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        marker.scale.x = 0.05
        marker.color.g = 1.0
        marker.color.a = 0.6
        
        for point in pred_data['trajectory']:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 0.1
            marker.points.append(p)
        
        return marker

    def optimize_for_dynamic_obstacles(self, path_points: List[np.ndarray]) -> List[np.ndarray]:
        """동적 장애물을 고려한 경로 최적화"""
        if not self.pedestrian_predictions:
            return path_points
            
        optimized_points = []
        for point in path_points:
            # 각 경로 포인트에 대해 최적의 위치 찾기
            best_point = point
            min_cost = float('inf')
            
            # 주변 영역 탐색
            for dx in np.linspace(-0.3, 0.3, 5):
                for dy in np.linspace(-0.3, 0.3, 5):
                    test_point = point + np.array([dx, dy])
                    
                    # 비용 계산
                    cost = self.calculate_point_cost(test_point)
                    
                    if cost < min_cost:
                        min_cost = cost
                        best_point = test_point
                        
            optimized_points.append(best_point)
        
        return optimized_points

    def calculate_point_cost(self, point: np.ndarray) -> float:
        """경로 포인트의 비용 계산"""
        # 장애물까지의 거리 비용
        obstacle_cost = self.calculate_obstacle_cost(point)
        
        # 보행자 예측 경로까지의 거리 비용
        prediction_cost = self.calculate_prediction_cost(point)
        
        # 경로 부드러움 비용
        smoothness_cost = self.calculate_smoothness_cost(point)
        
        return obstacle_cost + prediction_cost + smoothness_cost

    def calculate_obstacle_cost(self, point: np.ndarray) -> float:
        """장애물까지의 거리 기반 비용"""
        if self.costmap is None:
            return 0.0
            
        grid_pos = self.world_to_grid(point[0], point[1])
        if not (0 <= grid_pos[0] < self.costmap.shape[0] and 
                0 <= grid_pos[1] < self.costmap.shape[1]):
            return float('inf')
            
        cost = self.costmap[grid_pos[0], grid_pos[1]] / 100.0
        return cost

    def calculate_prediction_cost(self, point: np.ndarray) -> float:
        """보행자 예측 경로까지의 거리 기반 비용"""
        min_cost = 0.0
        
        for pred_data in self.pedestrian_predictions.values():
            for pred_point in pred_data['trajectory']:
                dist = np.linalg.norm(point - pred_point[:2])
                if dist < self.safety_margin:
                    cost = 1.0 - (dist / self.safety_margin)
                    min_cost = max(min_cost, cost)
                    
        return min_cost

    def calculate_smoothness_cost(self, point: np.ndarray) -> float:
        """경로 부드러움 기반 비용"""
        if len(self.current_path.poses) < 3:
            return 0.0
            
        # 현재 포인트와 인접한 포인트들과의 각도 계산
        prev_point = np.array([self.current_path.poses[-2].pose.position.x,
                            self.current_path.poses[-2].pose.position.y])
        curr_point = np.array([self.current_path.poses[-1].pose.position.x,
                            self.current_path.poses[-1].pose.position.y])
        
        v1 = curr_point - prev_point
        v2 = point - curr_point
        
        angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
        
        return abs(angle) / np.pi