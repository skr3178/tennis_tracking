"""
Generate schematic court video with upright player skeletons overlaid.
Two-pass pipeline:
  Pass 1 — detect players and collect camera-space data per frame
  Pass 2 — smooth positions, compute schematic projections, render
"""
import sys
import os
import cv2
import numpy as np
from scipy.signal import savgol_filter

sys.path.insert(0, 'tennis-tracking')
ORIG_DIR = os.getcwd()

from pose_detector import PoseDetector, PlayerTracker, SKELETON_CONNECTIONS
from schematic_renderer import (SchematicRenderer, NEAR_COLOR, FAR_COLOR,
                                 BALL_COLOR, BG_COLOR)

VIDEO = 'S_Original_HL_clip_cropped.mp4'
OUTPUT = 'schematic_output.mp4'
BALL_CSV = 'wasb_ball_positions.csv'

SMOOTH_WINDOW = 7
SMOOTH_POLY = 2
BALL_TRAIL_LEN = 15


def smooth_positions(positions, window=SMOOTH_WINDOW, poly=SMOOTH_POLY):
    """
    Smooth a list of (x, y) or None positions using Savitzky-Golay filter.
    Returns smoothed list with None entries preserved.
    """
    n = len(positions)
    xs = np.array([p[0] if p is not None else np.nan for p in positions])
    ys = np.array([p[1] if p is not None else np.nan for p in positions])

    valid = ~np.isnan(xs)
    if np.sum(valid) < window:
        return positions

    indices = np.arange(n)
    if np.any(~valid):
        xs[~valid] = np.interp(indices[~valid], indices[valid], xs[valid])
        ys[~valid] = np.interp(indices[~valid], indices[valid], ys[valid])

    if len(xs) >= window:
        xs_smooth = savgol_filter(xs, window, poly)
        ys_smooth = savgol_filter(ys, window, poly)
    else:
        xs_smooth, ys_smooth = xs, ys

    result = []
    for i in range(n):
        if positions[i] is not None:
            result.append((float(xs_smooth[i]), float(ys_smooth[i])))
        else:
            result.append(None)
    return result


def prepare_keypoints(det_data, role):
    """Extract keypoints and foot_cam from a detection, synthesize far-player lower body."""
    kps = det_data['keypoints'].copy()
    bbox = det_data['bbox']
    foot_cam = np.array([(bbox[0] + bbox[2]) / 2, bbox[3]])

    if role == 'far':
        bbox_cx = (bbox[0] + bbox[2]) / 2
        bbox_bottom = bbox[3]
        bbox_h = bbox[3] - bbox[1]
        synth = {
            11: (bbox_cx - bbox_h * 0.06, bbox[1] + bbox_h * 0.55),
            12: (bbox_cx + bbox_h * 0.06, bbox[1] + bbox_h * 0.55),
            13: (bbox_cx - bbox_h * 0.05, bbox[1] + bbox_h * 0.75),
            14: (bbox_cx + bbox_h * 0.05, bbox[1] + bbox_h * 0.75),
            15: (bbox_cx - bbox_h * 0.05, bbox_bottom),
            16: (bbox_cx + bbox_h * 0.05, bbox_bottom),
        }
        for idx, (sx, sy) in synth.items():
            if kps[idx, 2] < 0.3:
                kps[idx] = [sx, sy, 0.35]

    return kps, foot_cam


def main():
    # Init court detector
    os.chdir(os.path.join(ORIG_DIR, 'tennis-tracking'))
    from court_detector import CourtDetector
    os.chdir(ORIG_DIR)

    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'Input: {w}x{h}, {fps}fps, {total} frames')

    # Read all frames
    print('Reading frames...')
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    n_frames = len(frames)
    print(f'Read {n_frames} frames')

    # Court detection — use frame 0 warp for all frames (fixed camera)
    print('Detecting court...')
    os.chdir(os.path.join(ORIG_DIR, 'tennis-tracking'))
    cd = CourtDetector()
    cd.detect(frames[0])
    os.chdir(ORIG_DIR)
    cwm = cd.court_warp_matrix[0]  # fixed camera — single warp matrix
    print('  Court detected (fixed camera — using frame 0 homography)')

    # Init pose detector and tracker
    print('Detecting players...')
    det = PoseDetector(model_name='yolo26x-pose.pt',
                       crop_model_name='yolo11n-pose.pt', conf=0.1)
    tracker = PlayerTracker(det, hold_frames=75, max_disp_near=120, max_disp_far=60)

    # Init schematic renderer with video-derived court geometry
    renderer = SchematicRenderer()
    renderer.precompute_camera_court_geometry(cwm)

    # =========================================================================
    # Pass 1: Detect all players, collect camera-space data
    # =========================================================================
    print('Pass 1: Collecting detections...')
    far_data = [None] * n_frames   # {kps, foot_cam} per frame
    near_data = [None] * n_frames

    for i in range(n_frames):
        players = tracker.update(frames[i], conf=0.1)

        for role, storage in [('far', far_data), ('near', near_data)]:
            det_data = players.get(role)
            if det_data is not None:
                kps, foot_cam = prepare_keypoints(det_data, role)
                storage[i] = {'kps': kps, 'foot_cam': foot_cam}

        if i % 50 == 0:
            f = 'Y' if far_data[i] else 'N'
            n = 'Y' if near_data[i] else 'N'
            print(f'  Frame {i}/{n_frames}: far={f} near={n}')

    # =========================================================================
    # Pass 2: Smooth foot positions, project, render
    # =========================================================================
    print('Pass 2: Smoothing and rendering...')

    # Smooth camera-space foot positions
    far_feet = [d['foot_cam'].tolist() if d else None for d in far_data]
    near_feet = [d['foot_cam'].tolist() if d else None for d in near_data]
    far_feet_smooth = smooth_positions(far_feet)
    near_feet_smooth = smooth_positions(near_feet)

    # Write smoothed feet back
    for i in range(n_frames):
        if far_data[i] is not None and far_feet_smooth[i] is not None:
            far_data[i]['foot_cam'] = np.array(far_feet_smooth[i])
        if near_data[i] is not None and near_feet_smooth[i] is not None:
            near_data[i]['foot_cam'] = np.array(near_feet_smooth[i])

    # Project to schematic and smooth schematic positions too
    far_schem = [None] * n_frames
    near_schem = [None] * n_frames
    for i in range(n_frames):
        if far_data[i] is not None:
            far_schem[i] = renderer.transform_foot_to_schematic(far_data[i]['foot_cam'], cwm)
        if near_data[i] is not None:
            near_schem[i] = renderer.transform_foot_to_schematic(near_data[i]['foot_cam'], cwm)

    far_schem = smooth_positions(far_schem)
    near_schem = smooth_positions(near_schem)

    # Load ball positions from CSV and project to schematic
    print('  Loading ball positions...')
    import csv
    ball_cam = [None] * n_frames
    with open(BALL_CSV, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fi = int(row['frame'])
            if fi < n_frames and int(row['visible']):
                ball_cam[fi] = (float(row['x']), float(row['y']))

    # Project ball to schematic (ball is on/near ground plane)
    ball_schem = [None] * n_frames
    for i in range(n_frames):
        if ball_cam[i] is not None:
            ball_schem[i] = renderer.transform_foot_to_schematic(
                np.array(ball_cam[i]), cwm)

    ball_schem = smooth_positions(ball_schem, window=5, poly=2)

    n_ball = sum(1 for b in ball_schem if b is not None)
    print(f'  Ball positions: {n_ball}/{n_frames} frames')

    # Render
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('_temp_schematic.mp4', fourcc, fps, (1280, 720))

    for i in range(n_frames):
        player_list = []

        for role, data, schem, color, pid in [
            ('far', far_data, far_schem, FAR_COLOR, 1),
            ('near', near_data, near_schem, NEAR_COLOR, 2),
        ]:
            if data[i] is None or schem[i] is None:
                continue

            foot_schem = schem[i]
            foot_cam_y = data[i]['foot_cam'][1]
            kps = data[i]['kps']

            kps_schematic = renderer.compute_upright_skeleton(kps, foot_schem, foot_cam_y)

            player_list.append({
                'keypoints_schematic': kps_schematic,
                'original_kps': kps,
                'id': pid,
                'color': color,
                'foot_pos': foot_schem,
            })

        # Ball position and trail for this frame
        bp = ball_schem[i] if ball_schem[i] is not None else None
        bt = [b for b in ball_schem[max(0, i - BALL_TRAIL_LEN):i + 1] if b is not None]

        canvas = renderer.render_frame(
            frame_num=i, players=player_list,
            ball_pos=bp, ball_trail=bt if len(bt) >= 2 else None)
        writer.write(canvas)

        if i % 50 == 0:
            b = 'Y' if bp else 'N'
            print(f'  Rendering frame {i}/{n_frames}: {len(player_list)} players, ball={b}')

    writer.release()
    print(f'\nRe-encoding to H.264...')
    os.system(f'ffmpeg -y -i _temp_schematic.mp4 -c:v libx264 -preset fast -crf 23 '
              f'-pix_fmt yuv420p -movflags +faststart {OUTPUT} 2>/dev/null')
    os.remove('_temp_schematic.mp4')
    print(f'Done. Saved {OUTPUT}')


if __name__ == '__main__':
    main()
