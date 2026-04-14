"""
YOLO11-Pose wrapper for player detection + 17-keypoint skeleton estimation.
Includes dual-pass detection (full frame + crop for far player) and
court-area filtering to reject spectators/ball boys.
"""
import cv2
import numpy as np
from ultralytics import YOLO


class PoseDetector:
    def __init__(self, model_name='yolo26x-pose.pt', device=None, conf=0.3,
                 crop_model_name='yolo11n-pose.pt'):
        if device is None:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_name)
        # Use lighter model for crop passes (3 crops per frame) to save GPU memory
        self.crop_model = YOLO(crop_model_name)
        self.device = device
        self.conf = conf

    def _run_crop_model(self, frame, conf):
        """Run the lighter crop model."""
        results = self.crop_model(frame, device=self.device, conf=conf, verbose=False)
        detections = []
        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            kps = r.keypoints.data.cpu().numpy()
            for i in range(len(boxes)):
                detections.append({
                    'bbox': boxes[i],
                    'keypoints': kps[i],
                    'score': float(scores[i]),
                })
        return detections

    def _run_model(self, frame, conf):
        """Run YOLO on a single image, return raw detections."""
        results = self.model(frame, device=self.device, conf=conf, verbose=False)
        detections = []
        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            kps = r.keypoints.data.cpu().numpy()
            for i in range(len(boxes)):
                detections.append({
                    'bbox': boxes[i],
                    'keypoints': kps[i],
                    'score': float(scores[i]),
                })
        return detections

    def detect(self, frame, conf=None):
        """Single-pass detection."""
        if conf is None:
            conf = self.conf
        detections = self._run_model(frame, conf)
        detections.sort(key=lambda d: d['score'], reverse=True)
        return detections

    def detect_dual_pass(self, frame, conf=None, crop_scale=4):
        """
        Two-pass detection:
        Pass 1: full frame → near player
        Pass 2: tight crop of far court area, upscale 4x → far player

        Returns deduplicated list of detections.
        """
        if conf is None:
            conf = self.conf
        h, w = frame.shape[:2]

        # Pass 1: full frame
        full_dets = self._run_model(frame, conf)

        # Pass 2: tight crop of far court area and upscale
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
            overlaps = False
            for fd in all_dets:
                fx = (fd['bbox'][0] + fd['bbox'][2]) / 2
                fy = (fd['bbox'][1] + fd['bbox'][3]) / 2
                if abs(cx - fx) < 50 and abs(cy - fy) < 50:
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

    def detect_players(self, frame, game_warp_matrix=None, conf=None):
        """
        Detect exactly 2 players (near and far) using separate strategies:
        - Near player: from full-frame pass, filtered to lower court area
        - Far player: from crop pass only (crop already targets far court area)

        Returns:
            dict with 'near' and 'far' keys, each containing a detection dict
            or None if that player wasn't found.
        """
        if conf is None:
            conf = self.conf
        h, w = frame.shape[:2]

        # --- Near player: full-frame detection ---
        full_dets = self._run_model(frame, conf)
        # Filter: at least 5 visible keypoints, in the lower-center court area
        near_candidates = []
        for d in full_dets:
            vis = np.sum(d['keypoints'][:, 2] > 0.3)
            if vis < 5:
                continue
            cx = (d['bbox'][0] + d['bbox'][2]) / 2
            cy = (d['bbox'][1] + d['bbox'][3]) / 2
            bh = d['bbox'][3] - d['bbox'][1]
            # Near player is in the court area, not in the top banner/crowd
            if cy > h * 0.22 and w * 0.10 < cx < w * 0.90 and bh > 40:
                d['_cy'] = cy
                near_candidates.append(d)

        # Pick the one with highest y and best score
        near = None
        if near_candidates:
            # Sort by y (highest first), then by score
            near_candidates.sort(key=lambda d: (d['_cy'], d['score']), reverse=True)
            near = near_candidates[0]
            near.pop('_cy', None)
            for d in near_candidates[1:]:
                d.pop('_cy', None)

        # --- Far player: multiple tight crop passes ---
        # The far player can be anywhere along the far baseline.
        # A single wide crop dilutes resolution. Instead, run 3 overlapping
        # tight crops (left, center, right) at 4x upscale.
        crop_scale = 4
        y_start = int(h * 0.08)
        y_end = int(h * 0.32)
        crop_regions = [
            (int(w * 0.10), int(w * 0.42)),   # left third
            (int(w * 0.27), int(w * 0.58)),   # center third
            (int(w * 0.50), int(w * 0.82)),   # right third
        ]

        all_crop_dets = []
        for x_start, x_end in crop_regions:
            crop = frame[y_start:y_end, x_start:x_end]
            ch, cw = crop.shape[:2]
            crop_up = cv2.resize(crop, (cw * crop_scale, ch * crop_scale),
                                 interpolation=cv2.INTER_CUBIC)
            crop_dets = self._run_crop_model(crop_up, conf)

            # Map back to original coordinates
            for det in crop_dets:
                det['bbox'][0] = det['bbox'][0] / crop_scale + x_start
                det['bbox'][1] = det['bbox'][1] / crop_scale + y_start
                det['bbox'][2] = det['bbox'][2] / crop_scale + x_start
                det['bbox'][3] = det['bbox'][3] / crop_scale + y_start
                det['keypoints'][:, 0] = det['keypoints'][:, 0] / crop_scale + x_start
                det['keypoints'][:, 1] = det['keypoints'][:, 1] / crop_scale + y_start
            all_crop_dets.extend(crop_dets)

        # Deduplicate overlapping crop detections (keep highest score)
        deduped = []
        for d in sorted(all_crop_dets, key=lambda x: x['score'], reverse=True):
            cx = (d['bbox'][0] + d['bbox'][2]) / 2
            cy = (d['bbox'][1] + d['bbox'][3]) / 2
            is_dup = False
            for existing in deduped:
                ex = (existing['bbox'][0] + existing['bbox'][2]) / 2
                ey = (existing['bbox'][1] + existing['bbox'][3]) / 2
                if abs(cx - ex) < 40 and abs(cy - ey) < 40:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(d)

        # Filter far candidates: at least 5 visible keypoints
        far_candidates = []
        for d in deduped:
            vis = np.sum(d['keypoints'][:, 2] > 0.3)
            if vis >= 5:
                far_candidates.append(d)

        # Sort by score descending
        far_candidates.sort(key=lambda d: d['score'], reverse=True)

        return {'near': near, 'far_candidates': far_candidates}

    @staticmethod
    def _get_foot_position(det):
        """Get foot position from a detection (average of visible ankles or bbox bottom)."""
        kps = det['keypoints']
        ankles = [kps[j] for j in [15, 16] if kps[j, 2] > 0.3]
        if ankles:
            return np.mean([a[:2] for a in ankles], axis=0)
        return np.array([(det['bbox'][0] + det['bbox'][2]) / 2, det['bbox'][3]])


class PlayerTracker:
    """
    Wraps PoseDetector with motion continuity tracking.

    For each player (near/far):
    - If a detection is found close to the last known position, accept it.
    - If multiple candidates, pick the closest to last known position.
    - If no detection this frame, hold the last known detection for up
      to `hold_frames` frames.
    - Max displacement per frame limits how far a player can "jump".
    """
    def __init__(self, detector, hold_frames=5, max_disp_near=120, max_disp_far=60):
        self.detector = detector
        self.hold_frames = hold_frames
        self.max_disp_near = max_disp_near  # near player moves more pixels/frame
        self.max_disp_far = max_disp_far    # far player moves fewer (perspective)

        # State per player
        self._last = {'near': None, 'far': None}
        self._prev = {'near': None, 'far': None}  # detection from 2 frames ago (for velocity)
        self._age = {'near': 0, 'far': 0}   # frames since last real detection

    @staticmethod
    def _bbox_center(det):
        b = det['bbox']
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])

    def _pick_best_candidate(self, candidates, last, prev, max_disp, age):
        """
        From a list of candidates, pick the one closest to the predicted
        position (based on velocity from last two frames). Falls back to
        closest to last position if no velocity info.

        Returns (detection, accepted) where accepted=True if picked.
        """
        if not candidates:
            return None, False

        if last is None:
            # No history — pick the candidate most likely to be the actual
            # far player. Prefer candidates that are:
            # - in the lower part of the crop (higher y = on court baseline)
            # - have reasonable width (player, not a thin sign or wide crowd)
            # Filter to only candidates with cy > 130 (on court, not in crowd)
            on_court = [c for c in candidates
                        if (c['bbox'][1] + c['bbox'][3]) / 2 > 130]
            pool = on_court if on_court else candidates
            # From the pool, pick the one with highest y (most on-court)
            best_init = max(pool, key=lambda c: (c['bbox'][1] + c['bbox'][3]) / 2)
            return best_init, True

        last_center = self._bbox_center(last)
        allowed = max_disp * (age + 1)

        # Predict position using velocity if we have 2 frames of history
        if prev is not None and age == 0:
            prev_center = self._bbox_center(prev)
            velocity = last_center - prev_center
            predicted = last_center + velocity
        else:
            predicted = last_center

        # Pick candidate closest to predicted position within allowed range
        best = None
        best_dist = float('inf')
        for c in candidates:
            c_center = self._bbox_center(c)
            dist_to_pred = np.linalg.norm(c_center - predicted)
            dist_to_last = np.linalg.norm(c_center - last_center)
            # Must be within displacement range of last position
            if dist_to_last <= allowed and dist_to_pred < best_dist:
                best = c
                best_dist = dist_to_pred

        if best is not None:
            return best, True

        # No candidate within range — return None (will trigger hold)
        return None, False

    def update(self, frame, conf=None):
        """
        Detect players in frame with motion continuity.

        Near player: single candidate from detect_players.
        Far player: multiple candidates — pick closest to last known position.
        This naturally rejects the umpire since they're stationary and far
        from the actual player's last position.

        Returns:
            dict with 'near' and 'far' keys, each a detection dict or None.
            Detections have an extra 'held' key (True if reusing last frame's
            detection because no new one was found).
        """
        raw = self.detector.detect_players(frame, conf=conf)
        result = {}

        # --- Near player (single candidate) ---
        role = 'near'
        max_disp = self.max_disp_near
        candidate = raw.get('near')
        last = self._last[role]

        if candidate is not None and last is not None:
            dist = np.linalg.norm(self._bbox_center(candidate) - self._bbox_center(last))
            if dist <= max_disp * (self._age[role] + 1):
                candidate['held'] = False
                result[role] = candidate
                self._last[role] = candidate
                self._age[role] = 0
            else:
                if self._age[role] < self.hold_frames:
                    held = self._copy_det(last)
                    held['held'] = True
                    result[role] = held
                    self._age[role] += 1
                else:
                    candidate['held'] = False
                    result[role] = candidate
                    self._last[role] = candidate
                    self._age[role] = 0
        elif candidate is not None:
            candidate['held'] = False
            result[role] = candidate
            self._last[role] = candidate
            self._age[role] = 0
        elif last is not None and self._age[role] < self.hold_frames:
            held = self._copy_det(last)
            held['held'] = True
            result[role] = held
            self._age[role] += 1
        else:
            result[role] = None
            self._last[role] = None
            self._age[role] = 0

        # --- Far player (multiple candidates, pick by proximity + velocity) ---
        role = 'far'
        max_disp = self.max_disp_far
        candidates = raw.get('far_candidates', [])
        last = self._last[role]
        prev = self._prev[role]

        best, accepted = self._pick_best_candidate(
            candidates, last, prev, max_disp, self._age[role])

        if accepted:
            best['held'] = False
            result[role] = best
            self._prev[role] = self._last[role]
            self._last[role] = best
            self._age[role] = 0
        elif last is not None and self._age[role] < self.hold_frames:
            held = self._copy_det(last)
            held['held'] = True
            result[role] = held
            self._age[role] += 1
        else:
            # No candidates and hold expired — reset with best court-center pick
            if candidates:
                frame_cx = 640
                pick = min(candidates, key=lambda c: (
                    abs((c['bbox'][0] + c['bbox'][2]) / 2 - frame_cx) * 2
                    - (c['bbox'][1] + c['bbox'][3]) / 2
                ))
                pick['held'] = False
                result[role] = pick
                self._prev[role] = None
                self._last[role] = pick
                self._age[role] = 0
            else:
                result[role] = None
                self._prev[role] = None
                self._last[role] = None
                self._age[role] = 0

        return result

    @staticmethod
    def _copy_det(det):
        """Shallow copy a detection dict."""
        return {
            'bbox': det['bbox'].copy(),
            'keypoints': det['keypoints'].copy(),
            'score': det['score'],
        }


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
    video_path = sys.argv[1] if len(sys.argv) > 1 else 'Original_HL_clip.mp4'

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f'Failed to read {video_path}')
        sys.exit(1)

    print(f'Frame shape: {frame.shape}')
    detector = PoseDetector(conf=0.1)
    detections = detector.detect_dual_pass(frame, conf=0.1)
    print(f'Detected {len(detections)} people')

    for i, det in enumerate(detections):
        bbox = det['bbox']
        kps = det['keypoints']
        print(f'  Person {i}: bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}] '
              f'score={det["score"]:.2f} '
              f'keypoints_visible={np.sum(kps[:, 2] > 0.3)}/17')
