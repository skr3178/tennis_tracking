"""
TrackNet wrapper for tennis ball detection.
Uses 3-frame temporal input (9 channels) and heatmap-based detection.
Includes outlier removal, gap-aware interpolation, and trajectory smoothing.
"""
import sys
import cv2
import numpy as np
import torch
from scipy.signal import savgol_filter

sys.path.insert(0, 'TrackNet')
from model import BallTrackerNet
from general import postprocess


class BallTracker:
    def __init__(self, weights_path='TrackNet/checkpoint/model_best.pt', device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = BallTrackerNet()
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        self.input_w = 640
        self.input_h = 360

    def detect_sequence(self, frames):
        """
        Detect ball in a sequence of frames using 3-frame sliding window.

        Args:
            frames: list of BGR frames (full resolution)

        Returns:
            list of (x, y) or None for each frame
        """
        n = len(frames)
        positions = [None] * n

        for i in range(2, n):
            imgs = []
            for j in range(i - 2, i + 1):
                img = cv2.resize(frames[j], (self.input_w, self.input_h))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs.append(img)

            # Stack 3 frames into 9-channel input
            inp = np.concatenate(imgs, axis=2)  # (360, 640, 9)
            inp = inp.astype(np.float32) / 255.0
            inp = np.rollaxis(inp, 2, 0)  # (9, 360, 640)
            inp = torch.from_numpy(inp).unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.model(inp, testing=True)

            output = out.argmax(dim=1).detach().cpu().numpy()
            x, y = postprocess(output[0], scale=2)
            if x is not None:
                positions[i] = (float(x), float(y))

        return positions

    def remove_outliers(self, positions, max_jump=120):
        """
        Remove positions that jump too far from their neighbors.
        A position is an outlier if it jumps > max_jump pixels from BOTH
        the previous and next valid positions.
        """
        result = list(positions)
        n = len(result)

        for i in range(1, n - 1):
            if result[i] is None:
                continue
            # Find previous valid
            prev = None
            for j in range(i - 1, max(i - 6, -1), -1):
                if result[j] is not None:
                    prev = result[j]
                    break
            # Find next valid
            nxt = None
            for j in range(i + 1, min(i + 6, n)):
                if result[j] is not None:
                    nxt = result[j]
                    break

            if prev is not None:
                d_prev = np.hypot(result[i][0] - prev[0], result[i][1] - prev[1])
            else:
                d_prev = 0
            if nxt is not None:
                d_next = np.hypot(result[i][0] - nxt[0], result[i][1] - nxt[1])
            else:
                d_next = 0

            # Outlier if it jumps far from both neighbors
            if prev is not None and nxt is not None:
                if d_prev > max_jump and d_next > max_jump:
                    result[i] = None
            elif prev is not None and d_prev > max_jump * 1.5:
                result[i] = None

        return result

    def interpolate_positions(self, positions, max_gap=10, max_interp_dist=200):
        """
        Fill gaps in ball positions with linear interpolation.
        Only fills gaps smaller than max_gap frames AND where the
        endpoints are within max_interp_dist pixels of each other.
        This prevents wild diagonal lines across the court.
        """
        result = list(positions)
        n = len(result)
        i = 0
        while i < n:
            if result[i] is None:
                j = i
                while j < n and result[j] is None:
                    j += 1
                gap = j - i
                if gap <= max_gap and i > 0 and j < n:
                    x1, y1 = result[i - 1]
                    x2, y2 = result[j]
                    dist = np.hypot(x2 - x1, y2 - y1)
                    # Only interpolate if endpoints are close enough
                    if dist < max_interp_dist:
                        for k in range(i, j):
                            t = (k - i + 1) / (gap + 1)
                            result[k] = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
                i = j
            else:
                i += 1
        return result

    def smooth_positions(self, positions, window=5, poly=2):
        """
        Smooth ball positions using Savitzky-Golay filter.
        Operates on contiguous runs of detections separately to avoid
        smoothing across gaps.
        """
        result = list(positions)
        n = len(result)

        # Find contiguous runs
        i = 0
        while i < n:
            if result[i] is None:
                i += 1
                continue
            # Start of a run
            j = i
            while j < n and result[j] is not None:
                j += 1
            run_len = j - i

            if run_len >= window:
                xs = np.array([result[k][0] for k in range(i, j)])
                ys = np.array([result[k][1] for k in range(i, j)])
                xs_smooth = savgol_filter(xs, window, poly)
                ys_smooth = savgol_filter(ys, window, poly)
                for k in range(i, j):
                    result[k] = (float(xs_smooth[k - i]), float(ys_smooth[k - i]))

            i = j

        return result


if __name__ == '__main__':
    video_path = sys.argv[1] if len(sys.argv) > 1 else 'Tennis_original.mp4'

    cap = cv2.VideoCapture(video_path)
    frames = []
    # Read first 30 frames for testing
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f'Read {len(frames)} frames')

    tracker = BallTracker()
    positions = tracker.detect_sequence(frames)

    detected = sum(1 for p in positions if p is not None)
    print(f'Ball detected in {detected}/{len(frames)} frames')

    for i, pos in enumerate(positions):
        if pos is not None:
            print(f'  Frame {i}: ball at ({pos[0]:.1f}, {pos[1]:.1f})')

    # Draw detections on frames and save a composite
    out = frames[len(frames) // 2].copy()
    for i, pos in enumerate(positions):
        if pos is not None:
            alpha = 0.3 + 0.7 * (i / len(frames))
            cv2.circle(out, (int(pos[0]), int(pos[1])), 4, (0, 255, 255), -1)
    # Mark the positions with trail
    trail_pts = [(int(p[0]), int(p[1])) for p in positions if p is not None]
    for j in range(1, len(trail_pts)):
        cv2.line(out, trail_pts[j - 1], trail_pts[j], (0, 200, 200), 2)
    cv2.imwrite('test_ball_tracker.png', out)
    print('Saved test_ball_tracker.png')
