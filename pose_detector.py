"""
YOLO11-Pose wrapper for player detection + 17-keypoint skeleton estimation.
"""
import cv2
import numpy as np
from ultralytics import YOLO


class PoseDetector:
    def __init__(self, model_name='yolo11n-pose.pt', device=None, conf=0.3):
        if device is None:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_name)
        self.device = device
        self.conf = conf

    def detect(self, frame, conf=None):
        """
        Detect people in frame.

        Returns:
            list of dicts with:
                'bbox': np.array [x1, y1, x2, y2]
                'keypoints': np.array (17, 3) — x, y, confidence
                'score': float
        """
        if conf is None:
            conf = self.conf
        results = self.model(frame, device=self.device, conf=conf, verbose=False)
        detections = []
        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            kps = r.keypoints.data.cpu().numpy()  # (N, 17, 3)
            for i in range(len(boxes)):
                detections.append({
                    'bbox': boxes[i],
                    'keypoints': kps[i],
                    'score': float(scores[i]),
                })
        detections.sort(key=lambda d: d['score'], reverse=True)
        return detections


# Skeleton connections for drawing
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6),                                     # shoulders
    (5, 7), (7, 9),                             # left arm
    (6, 8), (8, 10),                            # right arm
    (5, 11), (6, 12),                           # torso
    (11, 12),                                   # hips
    (11, 13), (13, 15),                         # left leg
    (12, 14), (14, 16),                         # right leg
]


if __name__ == '__main__':
    import sys
    video_path = sys.argv[1] if len(sys.argv) > 1 else 'Tennis_original.mp4'

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f'Failed to read {video_path}')
        sys.exit(1)

    print(f'Frame shape: {frame.shape}')
    detector = PoseDetector()
    detections = detector.detect(frame)
    print(f'Detected {len(detections)} people')

    for i, det in enumerate(detections):
        bbox = det['bbox']
        kps = det['keypoints']
        print(f'  Person {i}: bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}] '
              f'score={det["score"]:.2f} '
              f'keypoints_visible={np.sum(kps[:, 2] > 0.3)}/17')

    # Draw result
    out = frame.copy()
    colors = [(0, 255, 0), (0, 200, 200), (255, 0, 0), (255, 0, 255)]
    for i, det in enumerate(detections):
        color = colors[i % len(colors)]
        bbox = det['bbox'].astype(int)
        cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        kps = det['keypoints']
        for j in range(17):
            if kps[j, 2] > 0.3:
                cv2.circle(out, (int(kps[j, 0]), int(kps[j, 1])), 3, color, -1)
        for a, b in SKELETON_CONNECTIONS:
            if kps[a, 2] > 0.3 and kps[b, 2] > 0.3:
                cv2.line(out, (int(kps[a, 0]), int(kps[a, 1])),
                         (int(kps[b, 0]), int(kps[b, 1])), color, 2)

    cv2.imwrite('test_pose_detector.png', out)
    print('Saved test_pose_detector.png')
