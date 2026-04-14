"""
Export each pipeline component as a separate image for visual inspection.
Uses real data from Tennis_original.mp4.
"""
import sys
import os
import cv2
import numpy as np

ORIG_DIR = os.getcwd()
sys.path.insert(0, os.path.join(ORIG_DIR, 'tennis-tracking'))
os.chdir(os.path.join(ORIG_DIR, 'tennis-tracking'))
from court_detector import CourtDetector
from court_reference import CourtReference
os.chdir(ORIG_DIR)

from pose_detector import PoseDetector, SKELETON_CONNECTIONS
from ball_tracker import BallTracker
from schematic_renderer import SchematicRenderer, NEAR_COLOR, FAR_COLOR, BALL_COLOR

VIDEO = 'Tennis_original.mp4'
FRAME_NUM = 50  # frame with both players visible

# ── Read frames ──
cap = cv2.VideoCapture(VIDEO)
frames = []
for i in range(FRAME_NUM + 1):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
frame = frames[FRAME_NUM]
h, w = frame.shape[:2]
print(f'Using frame {FRAME_NUM}, shape {frame.shape}')

# ═══════════════════════════════════════════
# 1. Original frame (clean)
# ═══════════════════════════════════════════
cv2.imwrite('export_01_original.png', frame)
print('1. Saved export_01_original.png')

# ═══════════════════════════════════════════
# 2. Court detection
# ═══════════════════════════════════════════
os.chdir(os.path.join(ORIG_DIR, 'tennis-tracking'))
cd = CourtDetector()
cd.detect(frames[0])
for i in range(1, FRAME_NUM + 1):
    cd.track_court(frames[i])
os.chdir(ORIG_DIR)

court_img = frame.copy()
court_overlay = cd.add_court_overlay(court_img, frame_num=FRAME_NUM, overlay_color=(0, 255, 0))
cv2.imwrite('export_02_court_detection.png', court_overlay)
print(f'2. Saved export_02_court_detection.png (config={cd.best_conf})')

# ═══════════════════════════════════════════
# 3. Pose detection (skeletons on original)
# ═══════════════════════════════════════════
pose_det = PoseDetector(conf=0.1)
detections = pose_det.detect(frame, conf=0.1)
print(f'3. Detected {len(detections)} people')

pose_img = frame.copy()
colors = [(0, 255, 0), (0, 200, 200), (255, 0, 0), (255, 0, 255)]
for i, det in enumerate(detections):
    color = colors[i % len(colors)]
    bbox = det['bbox'].astype(int)
    kps = det['keypoints']
    cv2.rectangle(pose_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    cv2.putText(pose_img, f'P{i} ({det["score"]:.2f})', (bbox[0], bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    for j in range(17):
        if kps[j, 2] > 0.3:
            cv2.circle(pose_img, (int(kps[j, 0]), int(kps[j, 1])), 3, color, -1)
    for a, b in SKELETON_CONNECTIONS:
        if kps[a, 2] > 0.3 and kps[b, 2] > 0.3:
            cv2.line(pose_img, (int(kps[a, 0]), int(kps[a, 1])),
                     (int(kps[b, 0]), int(kps[b, 1])), color, 2)
cv2.imwrite('export_03_pose_detection.png', pose_img)
print('   Saved export_03_pose_detection.png')

# Also export each person's keypoints as separate images
for i, det in enumerate(detections[:4]):
    kp_img = frame.copy()
    color = colors[i % len(colors)]
    bbox = det['bbox'].astype(int)
    kps = det['keypoints']
    cv2.rectangle(kp_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    for j in range(17):
        if kps[j, 2] > 0.3:
            cv2.circle(kp_img, (int(kps[j, 0]), int(kps[j, 1])), 4, color, -1)
            cv2.putText(kp_img, str(j), (int(kps[j, 0]) + 5, int(kps[j, 1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    for a, b in SKELETON_CONNECTIONS:
        if kps[a, 2] > 0.3 and kps[b, 2] > 0.3:
            cv2.line(kp_img, (int(kps[a, 0]), int(kps[a, 1])),
                     (int(kps[b, 0]), int(kps[b, 1])), color, 2)
    cv2.imwrite(f'export_03_person_{i}.png', kp_img)
    print(f'   Saved export_03_person_{i}.png (score={det["score"]:.2f}, bbox={bbox})')

# ═══════════════════════════════════════════
# 4. Ball detection (first 30 frames)
# ═══════════════════════════════════════════
tracker = BallTracker()
ball_positions = tracker.detect_sequence(frames[:FRAME_NUM + 1])
detected_count = sum(1 for p in ball_positions if p is not None)
print(f'4. Ball detected in {detected_count}/{len(ball_positions)} frames')

ball_img = frame.copy()
# Draw all ball detections up to this frame
for i, pos in enumerate(ball_positions):
    if pos is not None:
        alpha = 0.3 + 0.7 * (i / FRAME_NUM)
        cv2.circle(ball_img, (int(pos[0]), int(pos[1])), 4, (0, 255, 255), -1)
        cv2.putText(ball_img, str(i), (int(pos[0]) + 5, int(pos[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
cv2.imwrite('export_04_ball_detection.png', ball_img)
print('   Saved export_04_ball_detection.png')

# ═══════════════════════════════════════════
# 5. Empty schematic court
# ═══════════════════════════════════════════
renderer = SchematicRenderer()
empty_court = renderer.render_frame(frame_num=FRAME_NUM)
cv2.imwrite('export_05_schematic_empty.png', empty_court)
print('5. Saved export_05_schematic_empty.png')

# ═══════════════════════════════════════════
# 6. Homography test — project foot positions onto schematic
# ═══════════════════════════════════════════
game_warp = cd.game_warp_matrix[FRAME_NUM]
homography_img = empty_court.copy()

for i, det in enumerate(detections[:2]):
    kps = det['keypoints']
    # Get foot position (average of visible ankles)
    ankles = [kps[j] for j in [15, 16] if kps[j, 2] > 0.3]
    if ankles:
        foot_cam = np.mean([a[:2] for a in ankles], axis=0)
    else:
        foot_cam = np.array([(det['bbox'][0] + det['bbox'][2]) / 2, det['bbox'][3]])

    foot_schem = renderer.transform_foot_to_schematic(foot_cam, game_warp)
    color = FAR_COLOR if i == 0 else NEAR_COLOR
    cv2.circle(homography_img, (int(foot_schem[0]), int(foot_schem[1])), 8, color, -1)
    cv2.putText(homography_img, f'P{i}', (int(foot_schem[0]) + 10, int(foot_schem[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

cv2.imwrite('export_06_homography_feet.png', homography_img)
print('6. Saved export_06_homography_feet.png')

# ═══════════════════════════════════════════
# 7. Upright skeleton on schematic (per player)
# ═══════════════════════════════════════════
for i, det in enumerate(detections[:2]):
    kps = det['keypoints']
    ankles = [kps[j] for j in [15, 16] if kps[j, 2] > 0.3]
    if ankles:
        foot_cam = np.mean([a[:2] for a in ankles], axis=0)
    else:
        foot_cam = np.array([(det['bbox'][0] + det['bbox'][2]) / 2, det['bbox'][3]])

    foot_schem = renderer.transform_foot_to_schematic(foot_cam, game_warp)
    kps_schematic = renderer.compute_upright_skeleton(kps, foot_schem)
    color = FAR_COLOR if i == 0 else NEAR_COLOR

    skel_img = empty_court.copy()
    renderer.draw_skeleton(skel_img, kps_schematic, color, original_kps=kps)
    renderer.draw_player_bbox(skel_img, kps_schematic, color, original_kps=kps)
    renderer.draw_player_marker(skel_img, foot_schem, i + 1, color)
    cv2.imwrite(f'export_07_skeleton_P{i}.png', skel_img)
    print(f'7. Saved export_07_skeleton_P{i}.png')

print('\nDone! Check all export_*.png files.')
