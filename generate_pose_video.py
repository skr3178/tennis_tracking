"""
Generate pose estimation video from Original_HL_clip_cropped.mp4.
Uses dual-pass detection + court-area filtering for clean 2-player tracking.
"""
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, 'tennis-tracking')
ORIG_DIR = os.getcwd()

from pose_detector import PoseDetector, PlayerTracker, SKELETON_CONNECTIONS

VIDEO = 'S_Original_HL_clip_cropped.mp4'
OUTPUT = 'pose_estimation_output.mp4'

# Colors (BGR)
NEAR_COLOR = (0, 200, 0)       # green
FAR_COLOR = (50, 200, 220)     # gold
BALL_COLOR = (0, 255, 255)     # yellow

def draw_player(frame, det, color, label):
    if det is None:
        return
    bbox = det['bbox'].astype(int)
    kps = det['keypoints']

    # Bbox
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    cv2.putText(frame, label, (bbox[0], bbox[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    # Skeleton
    for a, b in SKELETON_CONNECTIONS:
        if kps[a, 2] > 0.3 and kps[b, 2] > 0.3:
            cv2.line(frame, (int(kps[a, 0]), int(kps[a, 1])),
                     (int(kps[b, 0]), int(kps[b, 1])), color, 2, cv2.LINE_AA)

    # Keypoints
    for j in range(17):
        if kps[j, 2] > 0.3:
            cv2.circle(frame, (int(kps[j, 0]), int(kps[j, 1])), 3,
                       (255, 255, 255), -1, cv2.LINE_AA)

    # Foot anchor dot — bottom center of bounding box
    foot = ((bbox[0] + bbox[2]) // 2, bbox[3])
    cv2.circle(frame, foot, 5, color, -1, cv2.LINE_AA)


def main():
    # Init court detector for filtering
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
            # Tracking failed — reuse previous warp matrices
            cd.court_warp_matrix.append(cd.court_warp_matrix[-1])
            cd.game_warp_matrix.append(cd.game_warp_matrix[-1])
        if i % 100 == 0:
            print(f'  Court tracking: {i}/{len(frames)}')
    os.chdir(ORIG_DIR)
    print(f'  Court tracked, {len(cd.game_warp_matrix)} matrices')

    # Pose detection with motion continuity tracking
    print('Detecting players...')
    det = PoseDetector(model_name='yolo26x-pose.pt', crop_model_name='yolo11n-pose.pt', conf=0.1)
    tracker = PlayerTracker(det, hold_frames=75, max_disp_near=120, max_disp_far=60)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('_temp_pose.mp4', fourcc, fps, (w, h))

    near_found = 0
    far_found = 0

    for i in range(len(frames)):
        frame = frames[i].copy()

        players = tracker.update(frame, conf=0.1)

        # Draw (dim held detections slightly)
        far_color = (30, 130, 160) if players.get('far') and players['far'].get('held') else FAR_COLOR
        near_color = (0, 130, 0) if players.get('near') and players['near'].get('held') else NEAR_COLOR
        draw_player(frame, players['far'], far_color, '#1 Far')
        draw_player(frame, players['near'], near_color, '#2 Near')

        # Status text
        status = f'Frame: {i}'
        if players['near'] is not None:
            near_found += 1
            status += f'  Near: {players["near"]["score"]:.2f}'
        if players['far'] is not None:
            far_found += 1
            status += f'  Far: {players["far"]["score"]:.2f}'
        cv2.putText(frame, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame)
        if i % 50 == 0:
            n = 'Y' if players['near'] else 'N'
            f = 'Y' if players['far'] else 'N'
            print(f'  Frame {i}/{len(frames)}: near={n} far={f}')

    writer.release()
    print(f'\nNear player found: {near_found}/{len(frames)} ({100*near_found/len(frames):.0f}%)')
    print(f'Far player found:  {far_found}/{len(frames)} ({100*far_found/len(frames):.0f}%)')

    # Re-encode to H.264 for Mac compatibility
    print('Re-encoding to H.264...')
    os.system(f'ffmpeg -y -i _temp_pose.mp4 -c:v libx264 -preset fast -crf 23 '
              f'-pix_fmt yuv420p -movflags +faststart {OUTPUT} 2>/dev/null')
    os.remove('_temp_pose.mp4')
    print(f'Done. Saved {OUTPUT}')


if __name__ == '__main__':
    main()
