"""
TrackNet wrapper for tennis ball detection.
Uses 3-frame temporal input (9 channels) and heatmap-based detection.
"""
import sys
import cv2
import numpy as np
import torch

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

    def interpolate_positions(self, positions, max_gap=10):
        """
        Fill gaps in ball positions with linear interpolation.
        Only fills gaps smaller than max_gap frames.
        """
        result = list(positions)
        n = len(result)
        i = 0
        while i < n:
            if result[i] is None:
                # Find gap end
                j = i
                while j < n and result[j] is None:
                    j += 1
                gap = j - i
                if gap <= max_gap and i > 0 and j < n:
                    x1, y1 = result[i - 1]
                    x2, y2 = result[j]
                    for k in range(i, j):
                        t = (k - i + 1) / (gap + 1)
                        result[k] = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
                i = j
            else:
                i += 1
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
