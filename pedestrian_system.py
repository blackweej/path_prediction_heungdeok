import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict, Optional
import torch.nn.functional as F
import cv2
import logging
from threading import Lock
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PedestrianLSTM(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, 
                 output_dim: int = 4, num_layers: int = 2):
        super(PedestrianLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 특징 추출을 위한 CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_dim + 64,  # CNN 특징 연결
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.3
        )
        
        # 어텐션 메커니즘
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
        # 출력 레이어
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # 불확실성 예측
        self.uncertainty = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus()  # 항상 양수 불확실성
        )

    def forward(self, sensor_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # CNN 특징 추출
        image_features = self.cnn(sensor_data['image'])
        image_features = image_features.view(sensor_data['image'].size(0), -1)
        
        # 센서 데이터와 이미지 특징 결합
        combined_input = torch.cat([
            sensor_data['trajectory'],
            image_features.repeat(1, sensor_data['trajectory'].size(1), 1)
        ], dim=-1)
        
        # LSTM 처리
        lstm_out, _ = self.lstm(combined_input)
        
        # 어텐션 적용
        attn_out, _ = self.attention(
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1)
        )
        attn_out = attn_out.transpose(0, 1)
        
        # 예측 생성
        predictions = self.fc(attn_out[:, -1, :])
        uncertainties = self.uncertainty(attn_out[:, -1, :])
        
        return predictions, uncertainties

class PredictionSystem:
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PedestrianLSTM().to(self.device)
        
        if model_path:
            self.load_model(model_path)
            
        self.tracker = MultiPedestrianTracker()
        self.lock = Lock()
        
        # 예측 파라미터
        self.prediction_horizon = 30  # 3초 (10Hz 기준)
        self.min_trajectory_length = 10
        self.max_trajectory_length = 50
        
        # 데이터 버퍼
        self.latest_data = {
            'image': None,
            'lidar': None,
            'tracked_pedestrians': {}
        }

    def load_model(self, model_path: str):
        """모델 가중치 로드"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def update_sensor_data(self, data_type: str, data: dict):
        """센서 데이터 업데이트"""
        with self.lock:
            try:
                if data_type == 'image':
                    # 이미지 디코딩 및 전처리
                    nparr = np.frombuffer(data['image'], np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    self.latest_data['image'] = self.preprocess_image(image)
                    
                elif data_type == 'lidar':
                    self.latest_data['lidar'] = data
                    
                # 보행자 추적 업데이트
                if self.latest_data['image'] is not None and self.latest_data['lidar'] is not None:
                    self.update_tracking()
                    
            except Exception as e:
                logger.error(f"Error updating sensor data: {e}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """이미지 전처리"""
        try:
            # 크기 조정
            image = cv2.resize(image, (224, 224))
            
            # 정규화
            image = image.astype(np.float32) / 255.0
            image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None

    def update_tracking(self):
        """보행자 추적 업데이트"""
        with self.lock:
            try:
                # 이미지에서 보행자 검출
                detections = self.detect_pedestrians(self.latest_data['image'])
                
                # LiDAR 데이터로 거리 정보 추가
                detections_3d = self.add_depth_information(detections, self.latest_data['lidar'])
                
                # 추적 업데이트
                self.tracker.update(detections_3d, time.time())
                
                # 추적 결과 저장
                self.latest_data['tracked_pedestrians'] = self.tracker.get_tracked_objects()
                
            except Exception as e:
                logger.error(f"Error updating tracking: {e}")

    def predict(self) -> Dict[int, dict]:
        """보행자 경로 예측"""
        with self.lock:
            try:
                predictions = {}
                
                for ped_id, tracked_data in self.latest_data['tracked_pedestrians'].items():
                    trajectory = tracked_data['trajectory']
                    
                    if len(trajectory) < self.min_trajectory_length:
                        continue
                        
                    # 입력 데이터 준비
                    sensor_data = self.prepare_input_data(tracked_data)
                    
                    # 예측 수행
                    with torch.no_grad():
                        pred, uncert = self.model(sensor_data)
                        
                    # 예측 결과 후처리
                    future_trajectory = self.postprocess_prediction(
                        pred.cpu().numpy(),
                        trajectory[-1]
                    )
                    
                    predictions[ped_id] = {
                        'trajectory': future_trajectory,
                        'uncertainty': uncert.cpu().numpy(),
                        'current_position': trajectory[-1],
                        'timestamp': time.time()
                    }
                
                return predictions
                
            except Exception as e:
                logger.error(f"Error in prediction: {e}")
                return {}

    def prepare_input_data(self, tracked_data: dict) -> Dict[str, torch.Tensor]:
        """모델 입력 데이터 준비"""
        # 궤적 데이터 준비
        trajectory = np.array(tracked_data['trajectory'][-self.max_trajectory_length:])
        trajectory = torch.FloatTensor(trajectory).unsqueeze(0)
        
        # 이미지 데이터 준비
        image = torch.FloatTensor(self.latest_data['image']).unsqueeze(0)
        image = image.permute(0, 3, 1, 2)  # BHWC -> BCHW
        
        return {
            'trajectory': trajectory.to(self.device),
            'image': image.to(self.device)
        }

    def postprocess_prediction(self, prediction: np.ndarray, 
                             current_position: np.ndarray) -> np.ndarray:
        """예측 결과 후처리"""
        future_trajectory = np.zeros((self.prediction_horizon, 2))
        current_pos = current_position[:2]
        velocity = prediction[2:]
        
        for i in range(self.prediction_horizon):
            future_trajectory[i] = current_pos + velocity * (i * 0.1)
            
        return future_trajectory

    def detect_pedestrians(self, image: np.ndarray) -> np.ndarray:
        """이미지에서 보행자 검출"""
        # HOG 디텍터 사용
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        boxes, weights = hog.detectMultiScale(
            image, 
            winStride=(8, 8),
            padding=(32, 32),
            scale=1.05
        )
        
        return np.array([box.center for box in boxes])

    def add_depth_information(self, detections: np.ndarray, 
                            lidar_data: dict) -> np.ndarray:
        """LiDAR 데이터를 사용하여 깊이 정보 추가"""
        detections_3d = []
        
        for detection in detections:
            # 이미지 좌표를 LiDAR 좌표로 변환
            lidar_point = self.image_to_lidar(detection, lidar_data)
            if lidar_point is not None:
                detections_3d.append(lidar_point)
                
        return np.array(detections_3d)

    def image_to_lidar(self, image_point: np.ndarray, 
                      lidar_data: dict) -> Optional[np.ndarray]:
        """이미지 좌표를 LiDAR 좌표로 변환"""
        try:
            # 카메라 캘리브레이션 매트릭스 (하드코딩된 값 사용)
            K = np.array([
                [1000, 0, 640],
                [0, 1000, 480],
                [0, 0, 1]
            ])
            
            # LiDAR 데이터에서 가장 가까운 포인트 찾기
            ranges = np.array(lidar_data['ranges'])
            angles = np.arange(
                lidar_data['angle_min'],
                lidar_data['angle_max'],
                lidar_data['angle_increment']
            )
            
            # 2D -> 3D 변환
            depth = np.min(ranges)  # 단순화를 위해 최소 거리 사용
            x = (image_point[0] - K[0, 2]) * depth / K[0, 0]
            y = (image_point[1] - K[1, 2]) * depth / K[1, 1]
            
            return np.array([x, y, depth])
            
        except Exception as e:
            logger.error(f"Error in coordinate transformation: {e}")
            return None
        
class MultiPedestrianTracker:
    def __init__(self, max_age: int = 10, min_hits: int = 3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = {}
        self.next_id = 0
        
        # 데이터 연결 파라미터
        self.max_distance = 2.0  # 최대 매칭 거리 (미터)
        self.max_missing = 5     # 최대 놓친 프레임 수
        
    def update(self, detections: np.ndarray, timestamp: float) -> None:
        """새로운 검출로 트래커 업데이트"""
        try:
            # 기존 트래커 예측
            predicted_states = {}
            for track_id, tracker in self.trackers.items():
                if tracker['active']:
                    predicted_states[track_id] = self.predict_state(tracker, timestamp)

            # 데이터 연결 수행
            matches, unmatched_detections = self.associate_detections(
                detections, 
                predicted_states
            )

            # 매칭된 트래커 업데이트
            for track_id, detection_idx in matches:
                self.update_tracker(
                    track_id, 
                    detections[detection_idx], 
                    timestamp
                )

            # 새로운 트래커 생성
            for idx in unmatched_detections:
                self.create_tracker(detections[idx], timestamp)

            # 매칭되지 않은 트래커 처리
            self.handle_unmatched_trackers(matches, timestamp)

        except Exception as e:
            logger.error(f"Error in tracker update: {e}")

    def predict_state(self, tracker: dict, timestamp: float) -> np.ndarray:
        """트래커 상태 예측"""
        dt = timestamp - tracker['last_timestamp']
        current_state = tracker['state']
        
        # 등속 모델 사용
        predicted_state = np.copy(current_state)
        predicted_state[:2] += current_state[2:] * dt
        
        return predicted_state

    def associate_detections(self, detections: np.ndarray, 
                           predicted_states: dict) -> Tuple[List[Tuple[int, int]], List[int]]:
        """헝가리안 알고리즘을 사용한 데이터 연결"""
        if not predicted_states or len(detections) == 0:
            return [], list(range(len(detections)))

        # 비용 행렬 계산
        cost_matrix = np.zeros((len(predicted_states), len(detections)))
        for i, (track_id, pred_state) in enumerate(predicted_states.items()):
            for j, detection in enumerate(detections):
                cost_matrix[i, j] = np.linalg.norm(pred_state[:2] - detection[:2])

        # 헝가리안 알고리즘
        rows, cols = linear_sum_assignment(cost_matrix)

        # 매칭 결과 처리
        matches = []
        unmatched_detections = list(range(len(detections)))
        track_ids = list(predicted_states.keys())

        for row, col in zip(rows, cols):
            if cost_matrix[row, col] < self.max_distance:
                track_id = track_ids[row]
                matches.append((track_id, col))
                unmatched_detections.remove(col)

        return matches, unmatched_detections

    def create_tracker(self, detection: np.ndarray, timestamp: float) -> None:
        """새로운 트래커 생성"""
        # 초기 상태: [x, y, vx, vy]
        initial_state = np.zeros(4)
        initial_state[:2] = detection[:2]
        
        self.trackers[self.next_id] = {
            'state': initial_state,
            'trajectory': [detection[:2]],
            'last_timestamp': timestamp,
            'hits': 1,
            'age': 0,
            'time_since_update': 0,
            'active': False  # min_hits에 도달할 때까지 비활성
        }
        self.next_id += 1

    def update_tracker(self, track_id: int, detection: np.ndarray, 
                      timestamp: float) -> None:
        """기존 트래커 업데이트"""
        tracker = self.trackers[track_id]
        dt = timestamp - tracker['last_timestamp']
        
        # 칼만 게인 (단순화된 버전)
        kalman_gain = 0.5
        
        # 상태 업데이트
        predicted_state = self.predict_state(tracker, timestamp)
        measurement = np.zeros_like(predicted_state)
        measurement[:2] = detection[:2]
        
        if len(tracker['trajectory']) >= 2:
            measurement[2:] = (detection[:2] - tracker['trajectory'][-1]) / dt
        
        new_state = predicted_state + kalman_gain * (measurement - predicted_state)
        
        # 트래커 정보 업데이트
        tracker['state'] = new_state
        tracker['trajectory'].append(detection[:2])
        tracker['last_timestamp'] = timestamp
        tracker['hits'] += 1
        tracker['time_since_update'] = 0
        
        # 충분한 hits가 쌓이면 활성화
        if tracker['hits'] >= self.min_hits:
            tracker['active'] = True

    def handle_unmatched_trackers(self, matches: List[Tuple[int, int]], 
                                timestamp: float) -> None:
        """매칭되지 않은 트래커 처리"""
        matched_track_ids = [match[0] for match in matches]
        
        for track_id, tracker in self.trackers.items():
            if track_id not in matched_track_ids:
                tracker['time_since_update'] += 1
                
                # 예측 상태로 업데이트
                if tracker['active']:
                    predicted_state = self.predict_state(tracker, timestamp)
                    tracker['state'] = predicted_state
                    tracker['trajectory'].append(predicted_state[:2])
                    tracker['last_timestamp'] = timestamp
                
                # 오래된 트래커 비활성화
                if tracker['time_since_update'] > self.max_missing:
                    tracker['active'] = False

    def get_tracked_objects(self) -> Dict[int, dict]:
        """현재 추적 중인 객체들의 정보 반환"""
        tracked_objects = {}
        
        for track_id, tracker in self.trackers.items():
            if tracker['active']:
                tracked_objects[track_id] = {
                    'state': tracker['state'],
                    'trajectory': tracker['trajectory'],
                    'age': tracker['age'],
                    'last_update': tracker['last_timestamp']
                }
                
        return tracked_objects