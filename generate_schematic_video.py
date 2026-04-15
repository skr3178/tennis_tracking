"""
Generate schematic court video with upright player skeletons overlaid.
Uses court detection for homography, pose detection for players,
and the schematic renderer for the perspective court view.
"""
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, 'tennis-tracking')
ORIG_DIR = os.getcwd()

from pose_detector import PoseDetector, PlayerTracker, SKELETON_CONNECTIONS
from schematic_renderer import (SchematicRenderer, NEAR_COLOR, FAR_COLOR,
                                 BALL_COLOR, BG_COLOR)

VIDEO = 'S_Original_HL_clip_cropped.mp4'
OUTPUT = 'schematic_output.mp4'


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
    print(f'Read {len(frames)} frames')

    # Court detection
    print('Detecting court...')
    os.chdir(os.path.join(ORIG_DIR, 'tennis-tracking'))
    cd = CourtDetector()
    cd.detect(frames[0])
    for i in range(1, len(frames)):
        try:
            cd.track_court(frames[i])
        except Exception:
            cd.court_warp_matrix.append(cd.court_warp_matrix[-1])
            cd.game_warp_matrix.append(cd.game_warp_matrix[-1])
        if i % 100 == 0:
            print(f'  Court tracking: {i}/{len(frames)}')
    os.chdir(ORIG_DIR)
    print(f'  Court tracked, {len(cd.game_warp_matrix)} matrices')

    # Init pose detector and tracker
    print('Detecting players...')
    det = PoseDetector(model_name='yolo26x-pose.pt',
                       crop_model_name='yolo11n-pose.pt', conf=0.1)
    tracker = PlayerTracker(det, hold_frames=75, max_disp_near=120, max_disp_far=60)

    # Init schematic renderer with video-derived court geometry
    renderer = SchematicRenderer()
    renderer.precompute_camera_court_geometry(cd.court_warp_matrix[0])

    # Output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('_temp_schematic.mp4', fourcc, fps, (1280, 720))

    for i in range(len(frames)):
        frame = frames[i]
        cwm = cd.court_warp_matrix[i] if i < len(cd.court_warp_matrix) else cd.court_warp_matrix[-1]

        # Detect players
        players = tracker.update(frame, conf=0.1)

        # Build player data for renderer
        player_list = []
        for role, color, pid in [('far', FAR_COLOR, 1), ('near', NEAR_COLOR, 2)]:
            det_data = players.get(role)
            if det_data is None:
                continue

            kps = det_data['keypoints'].copy()
            bbox = det_data['bbox']

            # Always use bbox bottom-center as foot anchor (matches pose video)
            foot_cam = np.array([(bbox[0] + bbox[2]) / 2, bbox[3]])

            # For far player, lower-body keypoints often have low confidence.
            # Synthesize missing hips/knees/ankles from bbox so the skeleton
            # extends to the foot anchor instead of being cropped at the waist.
            if role == 'far':
                bbox_cx = (bbox[0] + bbox[2]) / 2
                bbox_bottom = bbox[3]
                bbox_h = bbox[3] - bbox[1]
                # Approximate lower-body positions from bbox geometry
                synth = {
                    11: (bbox_cx - bbox_h * 0.06, bbox[1] + bbox_h * 0.55),  # left_hip
                    12: (bbox_cx + bbox_h * 0.06, bbox[1] + bbox_h * 0.55),  # right_hip
                    13: (bbox_cx - bbox_h * 0.05, bbox[1] + bbox_h * 0.75),  # left_knee
                    14: (bbox_cx + bbox_h * 0.05, bbox[1] + bbox_h * 0.75),  # right_knee
                    15: (bbox_cx - bbox_h * 0.05, bbox_bottom),              # left_ankle
                    16: (bbox_cx + bbox_h * 0.05, bbox_bottom),              # right_ankle
                }
                for idx, (sx, sy) in synth.items():
                    if kps[idx, 2] < 0.3:
                        kps[idx] = [sx, sy, 0.35]

            # Project foot to schematic via homography
            foot_schem = renderer.transform_foot_to_schematic(foot_cam, cwm)

            # Compute upright skeleton in schematic space
            kps_schematic = renderer.compute_upright_skeleton(kps, foot_schem, foot_cam[1])

            player_list.append({
                'keypoints_schematic': kps_schematic,
                'original_kps': kps,
                'id': pid,
                'color': color,
                'foot_pos': foot_schem,
            })

        # Render schematic frame
        canvas = renderer.render_frame(
            frame_num=i,
            players=player_list,
        )

        writer.write(canvas)
        if i % 50 == 0:
            n_on = sum(1 for p in player_list if True)
            print(f'  Frame {i}/{len(frames)}: {n_on} players on schematic')

    writer.release()
    print(f'\nRe-encoding to H.264...')
    os.system(f'ffmpeg -y -i _temp_schematic.mp4 -c:v libx264 -preset fast -crf 23 '
              f'-pix_fmt yuv420p -movflags +faststart {OUTPUT} 2>/dev/null')
    os.remove('_temp_schematic.mp4')
    print(f'Done. Saved {OUTPUT}')


if __name__ == '__main__':
    main()
