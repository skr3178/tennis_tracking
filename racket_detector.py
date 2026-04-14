"""
Racket detection + segmentation using YOLO11n-seg.
Filters for COCO class 38 (tennis racket).
"""
import cv2
import numpy as np
from ultralytics import YOLO

COCO_TENNIS_RACKET = 38


class RacketDetector:
    def __init__(self, model_name='yolo11n-seg.pt', device=None, conf=0.25):
        if device is None:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_name)
        self.device = device
        self.conf = conf

    def detect(self, frame, conf=None):
        """
        Detect tennis rackets in frame.

        Returns:
            list of dicts with:
                'bbox': np.array [x1, y1, x2, y2]
                'mask': binary mask (H, W) or None
                'score': float
                'angle': float — orientation angle of racket bbox in degrees
        """
        if conf is None:
            conf = self.conf
        results = self.model(frame, device=self.device, conf=conf, verbose=False)
        detections = []
        h, w = frame.shape[:2]

        for r in results:
            if r.boxes is None:
                continue
            classes = r.boxes.cls.cpu().numpy().astype(int)
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            masks = r.masks
            for i in range(len(boxes)):
                if classes[i] != COCO_TENNIS_RACKET:
                    continue
                mask = None
                if masks is not None and i < len(masks.data):
                    mask = masks.data[i].cpu().numpy()
                    mask = cv2.resize(mask, (w, h))
                    mask = (mask > 0.5).astype(np.uint8)

                # Compute angle from bbox
                x1, y1, x2, y2 = boxes[i]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

                detections.append({
                    'bbox': boxes[i],
                    'mask': mask,
                    'score': float(scores[i]),
                    'angle': angle,
                })
        detections.sort(key=lambda d: d['score'], reverse=True)
        return detections


def assign_rackets_to_players(rackets, players):
    """
    Assign each racket to the nearest player based on wrist keypoint proximity.

    Args:
        rackets: list of racket dicts from RacketDetector.detect()
        players: list of player dicts from PoseDetector.detect()

    Returns:
        list of (racket_idx, player_idx) pairs
    """
    if not rackets or not players:
        return []

    assignments = []
    used_players = set()

    for ri, racket in enumerate(rackets):
        rx = (racket['bbox'][0] + racket['bbox'][2]) / 2
        ry = (racket['bbox'][1] + racket['bbox'][3]) / 2

        best_dist = float('inf')
        best_pi = -1
        for pi, player in enumerate(players):
            if pi in used_players:
                continue
            kps = player['keypoints']
            # Check both wrists (indices 9=left_wrist, 10=right_wrist)
            for wrist_idx in [9, 10]:
                if kps[wrist_idx, 2] > 0.3:
                    dist = np.hypot(kps[wrist_idx, 0] - rx, kps[wrist_idx, 1] - ry)
                    if dist < best_dist:
                        best_dist = dist
                        best_pi = pi

        if best_pi >= 0 and best_dist < 200:
            assignments.append((ri, best_pi))
            used_players.add(best_pi)

    return assignments


if __name__ == '__main__':
    import sys
    video_path = sys.argv[1] if len(sys.argv) > 1 else 'Tennis_original.mp4'

    cap = cv2.VideoCapture(video_path)
    # Try several frames to find one with visible rackets
    best_frame = None
    best_count = 0
    detector = RacketDetector(conf=0.15)

    for frame_num in [50, 100, 150, 200, 250, 300, 400, 500]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
        rackets = detector.detect(frame, conf=0.15)
        print(f'Frame {frame_num}: {len(rackets)} rackets detected')
        for i, r in enumerate(rackets):
            bbox = r['bbox']
            print(f'  Racket {i}: bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}] '
                  f'score={r["score"]:.2f} angle={r["angle"]:.1f}')
        if len(rackets) > best_count:
            best_count = len(rackets)
            best_frame = frame.copy()
            best_rackets = rackets
            best_frame_num = frame_num
    cap.release()

    if best_frame is not None and best_count > 0:
        out = best_frame.copy()
        for r in best_rackets:
            bbox = r['bbox'].astype(int)
            cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            if r['mask'] is not None:
                overlay = out.copy()
                overlay[r['mask'] > 0] = [0, 0, 255]
                cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)
        cv2.imwrite('test_racket_detector.png', out)
        print(f'\nSaved test_racket_detector.png (frame {best_frame_num}, {best_count} rackets)')
    else:
        print('No rackets detected in any frame')
