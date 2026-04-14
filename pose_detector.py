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

    def _run_model(self, frame, conf):
        """Run YOLO on a single image, return raw detections."""
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
        return detections

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
        detections = self._run_model(frame, conf)
        detections.sort(key=lambda d: d['score'], reverse=True)
        return detections

    def detect_dual_pass(self, frame, conf=None, crop_scale=4):
        """
        Two-pass detection: full frame for near player, cropped+upscaled top
        region for the far player who is too small for direct detection.

        Pass 1: full frame → near player (bottom half of court)
        Pass 2: tight crop of far court area, upscale 4x → far player

        The crop must be small enough that after upscaling, the player still
        fills a meaningful portion of the YOLO input (640px). A tight crop
        of ~170x400px upscaled 4x → 680x1600 works well.

        Returns:
            list of dicts (same format as detect()), deduplicated
        """
        if conf is None:
            conf = self.conf
        h, w = frame.shape[:2]

        # Pass 1: full frame
        full_dets = self._run_model(frame, conf)

        # Pass 2: tight crop of far court area and upscale
        # Far player is typically in y=[8%..32%], x=[27%..58%] of frame
        y_start = int(h * 0.08)
        y_end = int(h * 0.32)
        x_start = int(w * 0.27)
        x_end = int(w * 0.58)
        crop = frame[y_start:y_end, x_start:x_end]
        ch, cw = crop.shape[:2]
        crop_up = cv2.resize(crop, (cw * crop_scale, ch * crop_scale),
                             interpolation=cv2.INTER_CUBIC)
        crop_dets = self._run_model(crop_up, conf)

        # Map crop detections back to original frame coordinates
        for det in crop_dets:
            det['bbox'][0] = det['bbox'][0] / crop_scale + x_start
            det['bbox'][1] = det['bbox'][1] / crop_scale + y_start
            det['bbox'][2] = det['bbox'][2] / crop_scale + x_start
            det['bbox'][3] = det['bbox'][3] / crop_scale + y_start
            det['keypoints'][:, 0] = det['keypoints'][:, 0] / crop_scale + x_start
            det['keypoints'][:, 1] = det['keypoints'][:, 1] / crop_scale + y_start

        # Merge: keep all full-frame dets, add crop dets that don't overlap
        all_dets = list(full_dets)
        for cd in crop_dets:
            cx = (cd['bbox'][0] + cd['bbox'][2]) / 2
            cy = (cd['bbox'][1] + cd['bbox'][3]) / 2
            # Only add if not overlapping with an existing detection
            overlaps = False
            for fd in all_dets:
                fx = (fd['bbox'][0] + fd['bbox'][2]) / 2
                fy = (fd['bbox'][1] + fd['bbox'][3]) / 2
                if abs(cx - fx) < 50 and abs(cy - fy) < 50:
                    # If crop detection has higher score, replace
                    if cd['score'] > fd['score']:
                        fd['bbox'] = cd['bbox']
                        fd['keypoints'] = cd['keypoints']
                        fd['score'] = cd['score']
                    overlaps = True
                    break
            if not overlaps:
                all_dets.append(cd)

        all_dets.sort(key=lambda d: d['score'], reverse=True)
        return all_dets


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
