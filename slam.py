import rospy
import numpy as np
from typing import Dict, List, Optional, Tuple
import tf2_ros
import tf.transformations
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import LaserScan
from scipy.spatial import KDTree
import logging
from threading import Lock
import math
from copy import deepcopy
import cv2
from scipy.spatial.transform import Rotation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MapManager:
    def __init__(self, resolution: float = 0.05):
        self.resolution = resolution
        self.origin = np.zeros(3)
        self.width = 1000
        self.height = 1000
        self.grid = np.ones((self.width, self.height), dtype=np.int8) * -1
        self.lock = Lock()

        # 맵핑 파라미터
        self.max_range = 30.0
        self.min_range = 0.3
        self.occupied_thresh = 0.65
        self.free_thresh = 0.196
        self.log_odds_thresh = 100
        self.map_update_counter = 0
        self.map_update_interval = 5

        # 맵 최적화를 위한 데이터
        self.keyframes = []
        self.submap_size = 50
        self.submaps = {}

    def update_after_loop_closure(self):
        """루프 클로저 후 맵 업데이트"""
        with self.lock:
            try:
                # 모든 키프레임의 포즈가 업데이트되었으므로 전체 맵 재구성
                temp_grid = np.ones((self.width, self.height), dtype=np.int8) * -1

                # 모든 키프레임의 스캔 데이터를 새로운 포즈로 투영
                for keyframe in self.keyframes:
                    points = keyframe['points']
                    pose = keyframe['pose']
                    
                    # 포인트를 월드 좌표계로 변환
                    world_points = (pose[:2, :2] @ points.T).T + pose[:2, 3]

                    # 각 포인트에 대해 레이 캐스팅 수행
                    for point in world_points:
                        # 시작점 (로봇 위치)
                        start = self.world_to_map(pose[:2, 3])
                        # 끝점 (관측된 포인트)
                        end = self.world_to_map(point)

                        # 레이 캐스팅
                        ray_cells = self.trace_ray(start, end)

                        # 맵 업데이트
                        for i, (x, y) in enumerate(ray_cells):
                            if not (0 <= x < self.width and 0 <= y < self.height):
                                continue

                            if i == len(ray_cells) - 1:
                                # 끝점은 점유 공간
                                temp_grid[x, y] = 100
                            else:
                                # 중간점은 자유 공간
                                if temp_grid[x, y] != 100:  # 점유 공간 우선
                                    temp_grid[x, y] = 0

                # 기존 맵과 병합
                valid_mask = (temp_grid != -1)
                self.grid[valid_mask] = temp_grid[valid_mask]

                logger.info("Map updated after loop closure")

            except Exception as e:
                logger.error(f"Error updating map after loop closure: {e}")


    def update_from_scan(self, points: np.ndarray, pose: np.ndarray):
        """LiDAR 스캔으로 맵 업데이트"""
        with self.lock:
            try:
                # 프레임 스킵 체크
                self.map_update_counter += 1
                if self.map_update_counter % self.map_update_interval != 0:
                    return

                # 레이 캐스팅 기반 맵 업데이트
                self.update_with_raycasting(points, pose)

                # 서브맵 업데이트
                self.update_submaps(points, pose)

                # 키프레임 추가 여부 확인
                if self.should_add_keyframe(pose):
                    self.add_keyframe(points, pose)
                    self.optimize_map()

            except Exception as e:
                logger.error(f"Error in map update: {e}")

    def update_with_raycasting(self, points: np.ndarray, pose: np.ndarray):
        """레이 캐스팅을 통한 맵 업데이트"""
        position = pose[:2, 3]
        rotation = pose[:2, :2]

        for point in points:
            # 포인트를 월드 좌표계로 변환
            world_point = rotation @ point + position

            # 시작점과 끝점의 맵 좌표 계산
            start = self.world_to_map(position)
            end = self.world_to_map(world_point)

            # Bresenham 레이 캐스팅
            ray_cells = self.trace_ray(start, end)

            # 맵 업데이트
            for i, (x, y) in enumerate(ray_cells):
                if not (0 <= x < self.width and 0 <= y < self.height):
                    continue

                if i == len(ray_cells) - 1:
                    # 끝점은 점유 공간
                    self.update_cell_log_odds(x, y, True)
                else:
                    # 중간점은 자유 공간
                    self.update_cell_log_odds(x, y, False)

    def update_cell_log_odds(self, x: int, y: int, is_occupied: bool):
        """로그 오즈를 사용한 셀 업데이트"""
        if not (0 <= x < self.width and 0 <= y < self.height):
            return

        current = self.grid[x, y]
        if current == -1:  # 미탐험 영역
            self.grid[x, y] = 0

        # 로그 오즈 업데이트
        update = 1 if is_occupied else -1
        new_value = np.clip(current + update, -self.log_odds_thresh, self.log_odds_thresh)
        self.grid[x, y] = int(new_value)

    def trace_ray(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Bresenham 알고리즘을 사용한 레이 트레이싱"""
        x0, y0 = start
        x1, y1 = end
        cells = []

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x1 > x0 else -1
        sy = 1 if y1 > y0 else -1

        if dx > dy:
            err = dx / 2.0
            while x != x1:
                cells.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                cells.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

        cells.append((x, y))
        return cells

    def update_submaps(self, points: np.ndarray, pose: np.ndarray):
        """서브맵 업데이트"""
        position = pose[:2, 3]
        submap_x = int(position[0] / (self.resolution * self.submap_size))
        submap_y = int(position[1] / (self.resolution * self.submap_size))
        submap_key = (submap_x, submap_y)

        if submap_key not in self.submaps:
            self.submaps[submap_key] = np.ones((self.submap_size, self.submap_size), dtype=np.int8) * -1

        for point in points:
            world_point = pose[:2, :2] @ point + pose[:2, 3]
            local_x = int((world_point[0] % (self.resolution * self.submap_size)) / self.resolution)
            local_y = int((world_point[1] % (self.resolution * self.submap_size)) / self.resolution)

            if 0 <= local_x < self.submap_size and 0 <= local_y < self.submap_size:
                self.submaps[submap_key][local_x, local_y] = 100

    def should_add_keyframe(self, pose: np.ndarray) -> bool:
        """키프레임 추가 여부 결정"""
        if not self.keyframes:
            return True

        last_keyframe = self.keyframes[-1]
        distance = np.linalg.norm(pose[:2, 3] - last_keyframe['pose'][:2, 3])
        rotation_diff = self.compute_rotation_diff(pose[:2, :2], last_keyframe['pose'][:2, :2])

        return distance > 0.5 or rotation_diff > 0.3

    def add_keyframe(self, points: np.ndarray, pose: np.ndarray):
        """키프레임 추가"""
        self.keyframes.append({
            'points': points.copy(),
            'pose': pose.copy(),
            'timestamp': rospy.Time.now().to_sec()
        })

        if len(self.keyframes) > 100:
            self.keyframes.pop(0)

    def optimize_map(self):
        """맵 최적화"""
        if len(self.keyframes) < 2:
            return

        try:
            for i in range(len(self.keyframes) - 1):
                source_points = self.keyframes[i]['points']
                target_points = self.keyframes[i + 1]['points']

                transform = self.icp_matching(source_points, target_points)
                self.keyframes[i + 1]['pose'] = transform @ self.keyframes[i + 1]['pose']

            self.update_global_map()

        except Exception as e:
            logger.error(f"Error in map optimization: {e}")

    def icp_matching(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """ICP 매칭 수행"""
        max_iterations = 30
        tolerance = 0.001
        transform = np.eye(4)
        prev_error = float('inf')

        for _ in range(max_iterations):
            # 대응점 찾기
            source_transformed = (transform[:2, :2] @ source.T).T + transform[:2, 3]
            tree = KDTree(target)
            distances, indices = tree.query(source_transformed)
            
            if len(indices) < 3:
                break

            # 유효한 대응점만 선택
            valid_mask = distances < np.mean(distances) * 3
            source_valid = source[valid_mask]
            target_valid = target[indices[valid_mask]]

            if len(source_valid) < 3:
                break

            # SVD로 변환 계산
            source_mean = np.mean(source_valid, axis=0)
            target_mean = np.mean(target_valid, axis=0)
            
            H = (source_valid - source_mean).T @ (target_valid - target_mean)
            U, S, Vt = np.linalg.svd(H)
            
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
                
            t = target_mean - R @ source_mean

            # 변환 행렬 업데이트
            current_transform = np.eye(4)
            current_transform[:2, :2] = R
            current_transform[:2, 3] = t
            transform = current_transform @ transform

            # 수렴 체크
            mean_error = np.mean(distances[valid_mask])
            if abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error

        return transform

    def update_global_map(self):
        """전역 맵 업데이트"""
        with self.lock:
            temp_grid = np.ones((self.width, self.height), dtype=np.int8) * -1

            for keyframe in self.keyframes:
                points = keyframe['points']
                pose = keyframe['pose']

                world_points = (pose[:2, :2] @ points.T).T + pose[:2, 3]
                
                for point in world_points:
                    x, y = self.world_to_map(point)
                    if 0 <= x < self.width and 0 <= y < self.height:
                        temp_grid[x, y] = 100

            self.grid = temp_grid

    def compute_rotation_diff(self, R1: np.ndarray, R2: np.ndarray) -> float:
        """두 회전 행렬 간의 각도 차이 계산"""
        diff = R1 @ R2.T
        trace = np.trace(diff)
        angle = np.arccos((trace - 1) / 2)
        return angle

    def world_to_map(self, point: np.ndarray) -> Tuple[int, int]:
        """월드 좌표를 맵 좌표로 변환"""
        x = int((point[0] - self.origin[0]) / self.resolution)
        y = int((point[1] - self.origin[1]) / self.resolution)
        return x, y

    def map_to_world(self, x: int, y: int) -> np.ndarray:
        """맵 좌표를 월드 좌표로 변환"""
        world_x = x * self.resolution + self.origin[0]
        world_y = y * self.resolution + self.origin[1]
        return np.array([world_x, world_y])

    def get_map_message(self) -> OccupancyGrid:
        """ROS OccupancyGrid 메시지 생성"""
        with self.lock:
            grid_msg = OccupancyGrid()
            grid_msg.header.frame_id = "map"
            grid_msg.header.stamp = rospy.Time.now()
            
            grid_msg.info.resolution = self.resolution
            grid_msg.info.width = self.width
            grid_msg.info.height = self.height
            grid_msg.info.origin.position.x = self.origin[0]
            grid_msg.info.origin.position.y = self.origin[1]
            
            grid_msg.data = self.grid.flatten().tolist()
            
            return grid_msg

    def save_map(self, filepath: str):
        """맵 저장"""
        try:
            map_data = {
                'grid': self.grid,
                'resolution': self.resolution,
                'origin': self.origin,
                'width': self.width,
                'height': self.height,
                'keyframes': self.keyframes
            }
            np.save(filepath, map_data)
            logger.info(f"Map saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving map: {e}")
            return False

    def load_map(self, filepath: str):
        """맵 로드"""
        try:
            map_data = np.load(filepath, allow_pickle=True).item()
            self.grid = map_data['grid']
            self.resolution = map_data['resolution']
            self.origin = map_data['origin']
            self.width = map_data['width']
            self.height = map_data['height']
            if 'keyframes' in map_data:
                self.keyframes = map_data['keyframes']
            logger.info(f"Map loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading map: {e}")
            return False

class LoopClosureDetector:
    def __init__(self, distance_threshold: float = 1.0, 
                 similarity_threshold: float = 0.7):
        self.distance_threshold = distance_threshold
        self.similarity_threshold = similarity_threshold
        self.keyframes: List[dict] = []
        self.lock = Lock()

    def add_keyframe(self, scan: np.ndarray, pose: np.ndarray):
        """키프레임 추가"""
        with self.lock:
            self.keyframes.append({
                'scan': scan,
                'pose': pose.copy(),
                'timestamp': rospy.Time.now().to_sec()
            })

    def detect(self, current_scan: np.ndarray, 
                current_pose: np.ndarray) -> Optional[dict]:
        """루프 클로저 감지"""
        with self.lock:
            if len(self.keyframes) < 10:  # 최소 키프레임 수
                return None

            # 현재 위치 근처의 키프레임 검색
            candidates = self.find_candidates(current_pose)
            
            for candidate in candidates:
                similarity = self.compute_similarity(
                    current_scan, 
                    candidate['scan']
                )
                
                if similarity > self.similarity_threshold:
                    return {
                        'keyframe': candidate,
                        'similarity': similarity
                    }
            
            return None

    def find_candidates(self, current_pose: np.ndarray) -> List[dict]:
        """후보 키프레임 찾기"""
        candidates = []
        current_position = current_pose[:3, 3]
        
        for keyframe in self.keyframes[:-10]:  # 최근 키프레임 제외
            keyframe_position = keyframe['pose'][:3, 3]
            distance = np.linalg.norm(current_position - keyframe_position)
            
            if distance < self.distance_threshold:
                candidates.append(keyframe)
        
        return candidates

    def compute_similarity(self, scan1: np.ndarray, 
                            scan2: np.ndarray) -> float:
        """스캔 유사도 계산"""
        # ICP 매칭 수행
        matches = self.find_correspondences(scan1, scan2)
        
        if len(matches) < 10:
            return 0.0
            
        # 매칭 점 비율로 유사도 계산
        similarity = len(matches) / min(len(scan1), len(scan2))
        
        return similarity

    def find_correspondences(self, scan1: np.ndarray, 
                            scan2: np.ndarray) -> np.ndarray:
        """대응점 찾기"""
        tree = KDTree(scan2)
        distances, indices = tree.query(scan1, k=1)
        
        # 거리 임계값 이내의 매칭만 사용
        valid_matches = distances < 0.5
        matches = np.column_stack((
            np.where(valid_matches)[0],
            indices[valid_matches]
        ))
        
        return matches

class FastLioHandler:
    def __init__(self, extT: List[float], extR: List[float]):
        """외부 파라미터와 상태 초기화"""
        # 외부 파라미터 설정
        self.ext_trans = np.array(extT)
        self.ext_rot = np.array(extR)
        
        # 상태 변수 초기화
        self.last_pose = np.eye(4)
        self.pose_covariance = np.eye(6)
        
        # ICP 파라미터
        self.icp_max_iterations = 30
        self.icp_tolerance = 0.001
        
        # 키프레임 관리
        self.keyframes = []
        self.keyframe_dist_thresh = 0.5
        self.keyframe_angle_thresh = 0.5

    def process_scan(self, points: np.ndarray) -> np.ndarray:
        """LiDAR 스캔 처리 및 포즈 추정"""
        try:
            # 포인트 클라우드 전처리
            filtered_points = self.preprocess_points(points)
            
            if not self.keyframes:
                # 첫 번째 스캔이면 키프레임으로 저장
                self.add_keyframe(filtered_points, self.last_pose)
                return self.last_pose
            
            # ICP 매칭
            estimated_pose = self.icp_matching(filtered_points)
            
            # 키프레임 추가 여부 확인
            if self.should_add_keyframe(estimated_pose):
                self.add_keyframe(filtered_points, estimated_pose)
            
            # 포즈 업데이트
            self.last_pose = estimated_pose
            
            return estimated_pose
            
        except Exception as e:
            logger.error(f"Error in FastLIO processing: {e}")
            return self.last_pose

    def preprocess_points(self, points: np.ndarray) -> np.ndarray:
        """포인트 클라우드 전처리"""
        # 외부 파라미터 적용
        R = self.euler_to_rotation_matrix(self.ext_rot)
        T = self.ext_trans
        
        # 포인트 변환
        transformed = (R @ points.T).T + T
        
        # 거리 기반 필터링
        distances = np.linalg.norm(transformed, axis=1)
        valid_mask = (distances > 0.5) & (distances < 30.0)
        
        return transformed[valid_mask]

    def icp_matching(self, current_points: np.ndarray) -> np.ndarray:
        """ICP 기반 스캔 매칭"""
        # 가장 최근 키프레임 가져오기
        prev_points = self.keyframes[-1]['points']
        initial_pose = self.last_pose
        
        current_pose = initial_pose
        prev_error = float('inf')
        
        for _ in range(self.icp_max_iterations):
            # 대응점 찾기
            current_transformed = (current_pose[:2, :2] @ current_points.T).T + current_pose[:2, 3]
            tree = KDTree(prev_points)
            distances, indices = tree.query(current_transformed)
            
            # 유효한 대응점만 선택
            valid_mask = distances < np.mean(distances) * 3
            source_valid = current_points[valid_mask]
            target_valid = prev_points[indices[valid_mask]]

            if len(source_valid) < 3:
                break

            # SVD로 변환 계산
            source_mean = np.mean(source_valid, axis=0)
            target_mean = np.mean(target_valid, axis=0)
            
            H = (source_valid - source_mean).T @ (target_valid - target_mean)
            U, S, Vt = np.linalg.svd(H)
            
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
                
            t = target_mean - R @ source_mean

            # 변환 행렬 업데이트
            current_transform = np.eye(4)
            current_transform[:2, :2] = R
            current_transform[:2, 3] = t
            current_pose = current_transform @ current_pose
                
            # 수렴 확인
            current_error = np.mean(distances[valid_mask])
            if abs(prev_error - current_error) < self.icp_tolerance:
                break
                
            prev_error = current_error
            
        return current_pose

    def add_keyframe(self, points: np.ndarray, pose: np.ndarray):
        """키프레임 추가"""
        self.keyframes.append({
            'points': points.copy(),
            'pose': pose.copy(),
            'timestamp': rospy.Time.now().to_sec()
        })
        
        # 키프레임 개수 제한
        if len(self.keyframes) > 100:
            self.keyframes.pop(0)

    def should_add_keyframe(self, current_pose: np.ndarray) -> bool:
        """키프레임 추가 여부 결정"""
        if not self.keyframes:
            return True
            
        last_keyframe = self.keyframes[-1]
        
        # 이동 거리 계산
        trans_diff = np.linalg.norm(current_pose[:3, 3] - last_keyframe['pose'][:3, 3])
        
        # 회전 각도 계산
        R_diff = current_pose[:3, :3] @ last_keyframe['pose'][:3, :3].T
        angle_diff = np.arccos((np.trace(R_diff) - 1) / 2)
        
        return (trans_diff > self.keyframe_dist_thresh or 
                angle_diff > self.keyframe_angle_thresh)

    @staticmethod
    def euler_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
        """오일러 각을 회전 행렬로 변환"""
        Rx = np.array([[1, 0, 0],
                        [0, np.cos(euler[0]), -np.sin(euler[0])],
                        [0, np.sin(euler[0]), np.cos(euler[0])]])
                        
        Ry = np.array([[np.cos(euler[1]), 0, np.sin(euler[1])],
                        [0, 1, 0],
                        [-np.sin(euler[1]), 0, np.cos(euler[1])]])
                        
        Rz = np.array([[np.cos(euler[2]), -np.sin(euler[2]), 0],
                        [np.sin(euler[2]), np.cos(euler[2]), 0],
                        [0, 0, 1]])
                        
        return Rz @ Ry @ Rx

class SLAMSystem:
    def __init__(self, config: Optional[dict] = None):
        """SLAM 시스템 초기화"""
        self.config = config or self.load_default_config()
        
        # 맵 관리
        self.map_manager = MapManager(self.config['map_resolution'])
        
        # FastLIO 초기화
        self.fastlio = FastLioHandler(
            extT=[0, 0, 0],
            extR=[0, 0, 0]
        )
        
        # 루프 클로저 감지
        self.loop_detector = LoopClosureDetector()
        
        # 위치 추정
        self.current_pose = np.eye(4)
        self.pose_covariance = np.eye(6)
        
        # Transform broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Publishers
        self.map_pub = rospy.Publisher('map', OccupancyGrid, queue_size=1)
        self.pose_pub = rospy.Publisher('slam_pose', PoseStamped, queue_size=1)
        
        # 시스템 상태
        self.is_first_scan = True
        self.system_lock = Lock()

    def load_default_config(self) -> dict:
        """기본 설정값 로드"""
        return {
            'map_resolution': 0.05,
            'max_laser_range': 30.0,
            'min_laser_range': 0.1,
            'map_update_interval': 0.5
        }

    def process_scan(self, scan_data: dict) -> bool:
        """LiDAR 스캔 처리"""
        with self.system_lock:
            try:
                # 스캔 데이터 전처리
                points = self.preprocess_scan(scan_data)
                if points is None:
                    return False

                # FastLIO를 통한 포즈 추정
                estimated_pose = self.fastlio.process_scan(points)
                
                # 루프 클로저 감지
                loop_detection = self.loop_detector.detect(points, estimated_pose)
                if loop_detection:
                    # 루프 클로저 처리
                    self.handle_loop_closure(loop_detection)
                
                # 포즈 업데이트
                self.update_pose(estimated_pose)
                
                # 맵 업데이트
                self.map_manager.update_from_scan(points, self.current_pose)
                
                # Transform 브로드캐스트
                self.broadcast_transform()
                
                # 발행
                self.publish_updates()
                
                return True
                
            except Exception as e:
                logger.error(f"Error in scan processing: {e}")
                return False

    def preprocess_scan(self, scan_data: dict) -> Optional[np.ndarray]:
        """LiDAR 스캔 전처리"""
        try:
            ranges = np.array(scan_data['ranges'])
            angles = np.arange(
                scan_data['angle_min'],
                scan_data['angle_max'],
                scan_data['angle_increment']
            )
            
            # 유효한 범위 필터링
            valid_mask = (ranges > self.config['min_laser_range']) & \
                        (ranges < self.config['max_laser_range'])
            
            if not np.any(valid_mask):
                return None
                
            points = np.zeros((np.sum(valid_mask), 2))
            points[:, 0] = np.cos(angles[valid_mask]) * ranges[valid_mask]
            points[:, 1] = np.sin(angles[valid_mask]) * ranges[valid_mask]
            
            return points
            
        except Exception as e:
            logger.error(f"Error in scan preprocessing: {e}")
            return None

    def handle_loop_closure(self, detection: dict):
        """루프 클로저 처리"""
        try:
            keyframe = detection['keyframe']
            
            # 포즈 그래프 최적화 (간단한 버전)
            correction = self.compute_loop_closure_correction(
                self.current_pose,
                keyframe['pose']
            )
            
            # 포즈 보정
            self.current_pose = correction @ self.current_pose
            
            # 맵 업데이트
            self.map_manager.update_after_loop_closure()
            
        except Exception as e:
            logger.error(f"Error in loop closure handling: {e}")

    def broadcast_transform(self):
        """TF 브로드캐스트"""
        try:
            transform = TransformStamped()
            transform.header.frame_id = "map"
            transform.header.stamp = rospy.Time.now()
            transform.child_frame_id = "base_link"
            
            # 위치 설정
            transform.transform.translation.x = self.current_pose[0, 3]
            transform.transform.translation.y = self.current_pose[1, 3]
            transform.transform.translation.z = self.current_pose[2, 3]
            
            # 회전 설정
            q = tf.transformations.quaternion_from_matrix(self.current_pose)
            transform.transform.rotation.x = q[0]
            transform.transform.rotation.y = q[1]
            transform.transform.rotation.z = q[2]
            transform.transform.rotation.w = q[3]
            
            self.tf_broadcaster.sendTransform(transform)
            
        except Exception as e:
            logger.error(f"Error in transform broadcast: {e}")

    def publish_updates(self):
        """업데이트된 데이터 발행"""
        try:
            # 맵 발행
            map_msg = self.map_manager.get_map_message()
            self.map_pub.publish(map_msg)
            
            # 포즈 발행
            pose_msg = PoseStamped()
            pose_msg.header.frame_id = "map"
            pose_msg.header.stamp = rospy.Time.now()
            
            pose_msg.pose.position.x = self.current_pose[0, 3]
            pose_msg.pose.position.y = self.current_pose[1, 3]
            pose_msg.pose.position.z = self.current_pose[2, 3]
            
            q = tf.transformations.quaternion_from_matrix(self.current_pose)
            pose_msg.pose.orientation.x = q[0]
            pose_msg.pose.orientation.y = q[1]
            pose_msg.pose.orientation.z = q[2]
            pose_msg.pose.orientation.w = q[3]
            
            self.pose_pub.publish(pose_msg)
            
        except Exception as e:
            logger.error(f"Error publishing updates: {e}")

    def update_pose(self, new_pose: np.ndarray):
        """포즈 업데이트"""
        self.current_pose = new_pose
        
        # 불확실성 업데이트 (간단한 모델)
        noise = np.eye(6) * 0.01  # 시스템 노이즈
        self.pose_covariance += noise

    def compute_loop_closure_correction(self, current_pose: np.ndarray, 
                                        keyframe_pose: np.ndarray) -> np.ndarray:
        """루프 클로저 보정 행렬 계산"""
        try:
            # 상대적 변환 계산
            relative_transform = np.linalg.inv(keyframe_pose) @ current_pose
            
            # 보정 행렬 계산 (부드러운 보정을 위해 보간)
            correction = np.eye(4)
            alpha = 0.7  # 보정 강도 (0~1)
            
            # 회전 보간
            R_diff = relative_transform[:3, :3]
            angle = np.arccos((np.trace(R_diff) - 1) / 2)
            if angle > np.pi:
                angle -= 2 * np.pi
            angle *= alpha
            
            axis = np.array([R_diff[2, 1] - R_diff[1, 2],
                            R_diff[0, 2] - R_diff[2, 0],
                            R_diff[1, 0] - R_diff[0, 1]])
            axis = axis / (2 * np.sin(angle))
            
            correction[:3, :3] = self.axis_angle_to_rotation_matrix(axis, angle)
            
            # 이동 보간
            correction[:3, 3] = relative_transform[:3, 3] * alpha
            
            return correction
            
        except Exception as e:
            logger.error(f"Error computing loop closure correction: {e}")
            return np.eye(4)

    @staticmethod
    def axis_angle_to_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
        """축-각도 표현을 회전 행렬로 변환"""
        c = np.cos(angle)
        s = np.sin(angle)
        v = 1 - c
        kx, ky, kz = axis
        
        R = np.array([
            [kx*kx*v + c,    kx*ky*v - kz*s, kx*kz*v + ky*s],
            [kx*ky*v + kz*s, ky*ky*v + c,    ky*kz*v - kx*s],
            [kx*kz*v - ky*s, ky*kz*v + kx*s, kz*kz*v + c   ]
        ])
        
        return R

    def save_state(self, filepath: str):
        """SLAM 시스템 상태 저장"""
        try:
            state = {
                'current_pose': self.current_pose,
                'pose_covariance': self.pose_covariance,
                'is_first_scan': self.is_first_scan,
                'config': self.config
            }
            np.save(filepath, state)
            
            # 맵 저장
            self.map_manager.save_map(filepath + '_map.npy')
            
            logger.info(f"SLAM state saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving SLAM state: {e}")
            return False

    def load_state(self, filepath: str):
        """SLAM 시스템 상태 로드"""
        try:
            state = np.load(filepath, allow_pickle=True).item()
            self.current_pose = state['current_pose']
            self.pose_covariance = state['pose_covariance']
            self.is_first_scan = state['is_first_scan']
            self.config = state['config']
            
            # 맵 로드
            self.map_manager.load_map(filepath + '_map.npy')
            
            logger.info(f"SLAM state loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading SLAM state: {e}")
            return False

    def get_current_pose(self) -> np.ndarray:
        """현재 포즈 반환"""
        return self.current_pose

    def get_map(self) -> OccupancyGrid:
        """현재 맵 반환"""
        return self.map_manager.get_map_message()

if __name__ == '__main__':
    try:
        rospy.init_node('slam_system')
        slam = SLAMSystem()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass