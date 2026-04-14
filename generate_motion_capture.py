"""
Tennis Motion Capture Pipeline
==============================
Three-phase execution:
  Phase A — Detection (all frames): court, players, rackets, ball
  Phase B — Post-processing: smoothing, outlier removal, coordinate transform
  Phase C — Rendering: schematic court with overlaid tracking data
"""
import sys
import os
import argparse
import time
import cv2
import numpy as np
from scipy.signal import savgol_filter

# Add tennis-tracking to path for court_detector, court_reference, sort
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tennis-tracking'))

from court_detector import CourtDetector
from pose_detector import PoseDetector
from ball_tracker import BallTracker
from racket_detector import RacketDetector, assign_rackets_to_players
from schematic_renderer import SchematicRenderer, NEAR_COLOR, FAR_COLOR, TRAIL_COLOR


def classify_near_far(detections, frame_h):
    """
    Classify detected players as near (bottom court) or far (top court).
    Near player = largest bbox in lower half of frame.
    Far player = highest bbox center in upper portion.

    Returns (near_det, far_det) or (None, None) if <2 detections.
    """
    if len(detections) < 2:
        if len(detections) == 1:
            # Single detection — assign based on y position
            det = detections[0]
            cy = (det['bbox'][1] + det['bbox'][3]) / 2
            if cy > frame_h * 0.45:
                return det, None
            else:
                return None, det
        return None, None

    # Sort by bbox bottom y (descending) — near player is lower in frame
    sorted_by_y = sorted(detections, key=lambda d: d['bbox'][3], reverse=True)

    # Near player: among detections in lower half, pick largest bbox area
    lower_half = [d for d in sorted_by_y if (d['bbox'][1] + d['bbox'][3]) / 2 > frame_h * 0.35]
    upper_half = [d for d in sorted_by_y if (d['bbox'][1] + d['bbox'][3]) / 2 <= frame_h * 0.35]

    if not lower_half:
        lower_half = [sorted_by_y[0]]
    if not upper_half:
        upper_half = [sorted_by_y[-1]]

    # Near = largest bbox area in lower half
    near_det = max(lower_half, key=lambda d: (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]))

    # Far = smallest bbox center y (highest in frame) excluding near
    remaining = [d for d in detections if d is not near_det]
    if remaining:
        far_det = min(remaining, key=lambda d: (d['bbox'][1] + d['bbox'][3]) / 2)
    else:
        far_det = None

    return near_det, far_det


def get_foot_position(keypoints):
    """
    Get foot (ankle) center from keypoints.
    Falls back to bbox bottom center if ankles not visible.
    """
    # Ankle indices: 15=left_ankle, 16=right_ankle
    visible_ankles = []
    for idx in [15, 16]:
        if keypoints[idx, 2] > 0.3:
            visible_ankles.append(keypoints[idx, :2])

    if visible_ankles:
        return np.mean(visible_ankles, axis=0)

    # Fallback: use lowest visible keypoint
    visible = keypoints[keypoints[:, 2] > 0.3]
    if len(visible) > 0:
        lowest = visible[np.argmax(visible[:, 1])]
        return lowest[:2]

    return None


def smooth_positions(positions, window=7, poly=2):
    """
    Smooth a list of (x, y) or None positions using Savitzky-Golay filter.
    Returns smoothed list with None entries preserved.
    """
    n = len(positions)
    xs = np.array([p[0] if p is not None else np.nan for p in positions])
    ys = np.array([p[1] if p is not None else np.nan for p in positions])

    # Only smooth if we have enough valid points
    valid = ~np.isnan(xs)
    if np.sum(valid) < window:
        return positions

    # Interpolate gaps for smoothing
    indices = np.arange(n)
    if np.any(~valid):
        xs[~valid] = np.interp(indices[~valid], indices[valid], xs[valid])
        ys[~valid] = np.interp(indices[~valid], indices[valid], ys[valid])

    # Apply Savitzky-Golay
    if len(xs) >= window:
        xs_smooth = savgol_filter(xs, window, poly)
        ys_smooth = savgol_filter(ys, window, poly)
    else:
        xs_smooth = xs
        ys_smooth = ys

    result = []
    for i in range(n):
        if positions[i] is not None:
            result.append((float(xs_smooth[i]), float(ys_smooth[i])))
        else:
            result.append(None)
    return result


def remove_ball_outliers(positions, dx_thresh=50, dy_thresh=50):
    """
    Remove ball position outliers where both dx and dy exceed thresholds.
    Replace with None for later interpolation.
    """
    result = list(positions)
    for i in range(1, len(result)):
        if result[i] is not None and result[i - 1] is not None:
            dx = abs(result[i][0] - result[i - 1][0])
            dy = abs(result[i][1] - result[i - 1][1])
            if dx > dx_thresh and dy > dy_thresh:
                result[i] = None
    return result


def main():
    parser = argparse.ArgumentParser(description='Tennis Motion Capture Generator')
    parser.add_argument('--input', default='Tennis_original.mp4', help='Input video path')
    parser.add_argument('--output', default='Motion_capture_output.mp4', help='Output video path')
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    # =========================================================================
    # Read all frames
    # =========================================================================
    print(f'Reading video: {input_path}')
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'  {frame_w}x{frame_h} @ {fps}fps, {total_frames} frames')

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    n_frames = len(frames)
    print(f'  Read {n_frames} frames')

    # =========================================================================
    # Phase A — Detection
    # =========================================================================
    print('\n=== Phase A: Detection ===')
    t0 = time.time()

    # Initialize detectors
    print('  Initializing detectors...')
    court_detector = CourtDetector()
    pose_detector = PoseDetector(device=args.device)
    ball_tracker = BallTracker(device=args.device)
    racket_detector = RacketDetector(device=args.device)
    renderer = SchematicRenderer()

    # --- Court detection ---
    print('  Detecting court (frame 0)...')
    court_detector.detect(frames[0])
    print(f'    Court score: {court_detector.court_score:.0f}')

    print('  Tracking court across frames...')
    for i in range(1, n_frames):
        court_detector.track_court(frames[i])
        if (i + 1) % 100 == 0:
            print(f'    Frame {i + 1}/{n_frames}')

    game_warp_matrices = court_detector.game_warp_matrix
    print(f'    Got {len(game_warp_matrices)} warp matrices')

    # --- Player detection (pose) ---
    print('  Detecting players...')
    all_player_detections = []
    for i in range(n_frames):
        dets = pose_detector.detect(frames[i])
        all_player_detections.append(dets)
        if (i + 1) % 100 == 0:
            print(f'    Frame {i + 1}/{n_frames}: {len(dets)} people')

    # --- Classify near/far and track ---
    print('  Classifying near/far players...')
    near_player_data = [None] * n_frames  # dict with bbox, keypoints, foot_pos
    far_player_data = [None] * n_frames

    for i in range(n_frames):
        dets = all_player_detections[i]
        near_det, far_det = classify_near_far(dets, frame_h)

        if near_det is not None:
            foot = get_foot_position(near_det['keypoints'])
            near_player_data[i] = {
                'bbox': near_det['bbox'],
                'keypoints': near_det['keypoints'],
                'score': near_det['score'],
                'foot_camera': foot,
            }

        if far_det is not None:
            foot = get_foot_position(far_det['keypoints'])
            far_player_data[i] = {
                'bbox': far_det['bbox'],
                'keypoints': far_det['keypoints'],
                'score': far_det['score'],
                'foot_camera': foot,
            }

    near_count = sum(1 for d in near_player_data if d is not None)
    far_count = sum(1 for d in far_player_data if d is not None)
    print(f'    Near player detected in {near_count}/{n_frames} frames')
    print(f'    Far player detected in {far_count}/{n_frames} frames')

    # --- Racket detection ---
    print('  Detecting rackets...')
    all_racket_data = []
    for i in range(n_frames):
        rackets = racket_detector.detect(frames[i])
        # Build player list for assignment
        players_for_assign = []
        if near_player_data[i] is not None:
            players_for_assign.append({
                'bbox': near_player_data[i]['bbox'],
                'keypoints': near_player_data[i]['keypoints'],
            })
        if far_player_data[i] is not None:
            players_for_assign.append({
                'bbox': far_player_data[i]['bbox'],
                'keypoints': far_player_data[i]['keypoints'],
            })

        assignments = assign_rackets_to_players(rackets, players_for_assign)

        # Map assignments back to near/far
        near_racket = None
        far_racket = None
        for ri, pi in assignments:
            # pi=0 could be near or far depending on what was added first
            if pi == 0 and near_player_data[i] is not None:
                near_racket = rackets[ri]
            elif pi == 1 and far_player_data[i] is not None:
                far_racket = rackets[ri]
            elif pi == 0 and near_player_data[i] is None and far_player_data[i] is not None:
                far_racket = rackets[ri]

        all_racket_data.append({'near': near_racket, 'far': far_racket})
        if (i + 1) % 100 == 0:
            print(f'    Frame {i + 1}/{n_frames}: {len(rackets)} rackets')

    # --- Ball detection ---
    print('  Detecting ball (TrackNet)...')
    ball_positions_raw = ball_tracker.detect_sequence(frames)
    detected_balls = sum(1 for p in ball_positions_raw if p is not None)
    print(f'    Ball detected in {detected_balls}/{n_frames} frames')

    t_detect = time.time() - t0
    print(f'  Detection complete in {t_detect:.1f}s')

    # =========================================================================
    # Phase B — Post-processing
    # =========================================================================
    print('\n=== Phase B: Post-processing ===')
    t1 = time.time()

    # Smooth player foot positions
    print('  Smoothing player positions...')
    near_feet_cam = [d['foot_camera'] if d is not None and d['foot_camera'] is not None else None
                     for d in near_player_data]
    far_feet_cam = [d['foot_camera'] if d is not None and d['foot_camera'] is not None else None
                    for d in far_player_data]

    near_feet_cam = smooth_positions(near_feet_cam, window=7, poly=2)
    far_feet_cam = smooth_positions(far_feet_cam, window=7, poly=2)

    # Write smoothed feet back
    for i in range(n_frames):
        if near_player_data[i] is not None and near_feet_cam[i] is not None:
            near_player_data[i]['foot_camera'] = np.array(near_feet_cam[i])
        if far_player_data[i] is not None and far_feet_cam[i] is not None:
            far_player_data[i]['foot_camera'] = np.array(far_feet_cam[i])

    # Ball outlier removal + interpolation
    print('  Processing ball positions...')
    ball_positions = remove_ball_outliers(ball_positions_raw, dx_thresh=50, dy_thresh=50)
    ball_positions = ball_tracker.interpolate_positions(ball_positions, max_gap=10)

    # Transform all coords to schematic space
    print('  Transforming coordinates to schematic space...')
    near_schematic = [None] * n_frames  # foot position in schematic
    far_schematic = [None] * n_frames
    ball_schematic = [None] * n_frames
    near_kps_schematic = [None] * n_frames
    far_kps_schematic = [None] * n_frames

    for i in range(n_frames):
        gwm = game_warp_matrices[i]
        if gwm is None:
            continue

        # Near player
        if near_player_data[i] is not None and near_player_data[i]['foot_camera'] is not None:
            foot_cam = near_player_data[i]['foot_camera']
            try:
                foot_schem = renderer.transform_foot_to_schematic(foot_cam, gwm)
                # Sanity check — foot should be on or near court
                if -100 < foot_schem[0] < 1380 and 100 < foot_schem[1] < 700:
                    near_schematic[i] = foot_schem
                    near_kps_schematic[i] = renderer.compute_upright_skeleton(
                        near_player_data[i]['keypoints'], foot_schem)
            except Exception:
                pass

        # Far player
        if far_player_data[i] is not None and far_player_data[i]['foot_camera'] is not None:
            foot_cam = far_player_data[i]['foot_camera']
            try:
                foot_schem = renderer.transform_foot_to_schematic(foot_cam, gwm)
                if -100 < foot_schem[0] < 1380 and 100 < foot_schem[1] < 700:
                    far_schematic[i] = foot_schem
                    far_kps_schematic[i] = renderer.compute_upright_skeleton(
                        far_player_data[i]['keypoints'], foot_schem)
            except Exception:
                pass

        # Ball
        if ball_positions[i] is not None:
            try:
                ball_cam = ball_positions[i]
                ball_schem = renderer.transform_foot_to_schematic(ball_cam, gwm)
                if -100 < ball_schem[0] < 1380 and 50 < ball_schem[1] < 750:
                    ball_schematic[i] = ball_schem
            except Exception:
                pass

    near_valid = sum(1 for p in near_schematic if p is not None)
    far_valid = sum(1 for p in far_schematic if p is not None)
    ball_valid = sum(1 for p in ball_schematic if p is not None)
    print(f'    Near player: {near_valid}/{n_frames} valid schematic positions')
    print(f'    Far player: {far_valid}/{n_frames} valid schematic positions')
    print(f'    Ball: {ball_valid}/{n_frames} valid schematic positions')

    t_post = time.time() - t1
    print(f'  Post-processing complete in {t_post:.1f}s')

    # =========================================================================
    # Phase C — Rendering
    # =========================================================================
    print('\n=== Phase C: Rendering ===')
    t2 = time.time()

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (1280, 720))

    # Build trails as we go
    near_trail = []
    far_trail = []
    ball_trail = []

    for i in range(n_frames):
        # Update trails
        if near_schematic[i] is not None:
            near_trail.append(near_schematic[i])
        if far_schematic[i] is not None:
            far_trail.append(far_schematic[i])
        if ball_schematic[i] is not None:
            ball_trail.append(ball_schematic[i])

        # Build player data for renderer
        players = []
        if near_kps_schematic[i] is not None and near_player_data[i] is not None:
            players.append({
                'keypoints_schematic': near_kps_schematic[i],
                'original_kps': near_player_data[i]['keypoints'],
                'id': 2,
                'color': NEAR_COLOR,
                'foot_pos': near_schematic[i],
            })
        if far_kps_schematic[i] is not None and far_player_data[i] is not None:
            players.append({
                'keypoints_schematic': far_kps_schematic[i],
                'original_kps': far_player_data[i]['keypoints'],
                'id': 1,
                'color': FAR_COLOR,
                'foot_pos': far_schematic[i],
            })

        # Player trails dict
        player_trails = {}
        if near_trail:
            player_trails[2] = near_trail[-30:]
        if far_trail:
            player_trails[1] = far_trail[-30:]

        # Ball trail (last 15)
        bt = ball_trail[-15:] if ball_trail else None

        # Render
        canvas = renderer.render_frame(
            frame_num=i,
            players=players if players else None,
            ball_pos=ball_schematic[i],
            player_trails=player_trails if player_trails else None,
            ball_trail=bt,
        )

        writer.write(canvas)

        if (i + 1) % 100 == 0:
            print(f'    Rendered frame {i + 1}/{n_frames}')

    writer.release()
    t_render = time.time() - t2
    total_time = time.time() - t0

    print(f'\n  Rendering complete in {t_render:.1f}s')
    print(f'  Total pipeline time: {total_time:.1f}s')
    print(f'\nOutput saved to: {output_path}')
    print(f'  Resolution: 1280x720 @ {fps}fps')
    print(f'  Frames: {n_frames}')


if __name__ == '__main__':
    main()
