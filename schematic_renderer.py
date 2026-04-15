"""
Perspective court schematic renderer.
Draws a tennis court in perspective view and overlays upright player skeletons,
ball position, and tracking info.

Key insight: only feet touch the ground plane. Skeletons and the net stand
upright, so we project the foot position via the ground-plane homography,
then draw the figure *upward* from that anchor using camera-space proportions
scaled by perspective depth.
"""
import cv2
import numpy as np

# Output canvas size
CANVAS_W = 1280
CANVAS_H = 720

# Court reference corners (from court_reference.py)
REF_CORNERS = np.array([
    [286, 561],    # far-left (top-left of court)
    [1379, 561],   # far-right (top-right of court)
    [286, 2935],   # near-left (bottom-left of court)
    [1379, 2935],  # near-right (bottom-right of court)
], dtype=np.float32)

# Schematic output corners (measured from reference Motion_capture.mp4)
SCHEMATIC_CORNERS = np.array([
    [406, 210],    # far-left
    [874, 210],    # far-right
    [115, 590],    # near-left
    [1165, 590],   # near-right
], dtype=np.float32)

# Court reference line endpoints (from court_reference.py)
COURT_LINES_REF = {
    'baseline_top':     ((286, 561), (1379, 561)),
    'baseline_bottom':  ((286, 2935), (1379, 2935)),
    'left_court':       ((286, 561), (286, 2935)),
    'right_court':      ((1379, 561), (1379, 2935)),
    'left_inner':       ((423, 561), (423, 2935)),
    'right_inner':      ((1242, 561), (1242, 2935)),
    'middle':           ((832, 1110), (832, 2386)),
    'top_inner':        ((423, 1110), (1242, 1110)),
    'bottom_inner':     ((423, 2386), (1242, 2386)),
    'net':              ((286, 1748), (1379, 1748)),
}

# Colors (BGR)
BG_COLOR = (243, 242, 231)
COURT_COLOR = (207, 198, 181)
LINE_COLOR = (100, 100, 100)
DOT_COLOR = (80, 80, 80)
NET_COLOR = (83, 160, 220)       # orange in BGR
NEAR_COLOR = (0, 180, 0)
FAR_COLOR = (50, 180, 210)
BALL_COLOR = (50, 100, 230)
BALL_BBOX_COLOR = (150, 150, 200)
TRAIL_COLOR = (80, 80, 80)
TEXT_COLOR = (120, 120, 120)

# Skeleton connections (YOLO 17-keypoint format)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]

# Net height in reference court units (same coordinate system as REF_CORNERS)
NET_HEIGHT_REF = 107
REF_COURT_WIDTH = 1379 - 286  # = 1093 reference units


class SchematicRenderer:
    def __init__(self):
        # Compute homography: court reference → schematic output
        self.H_ref_to_schematic, _ = cv2.findHomography(REF_CORNERS, SCHEMATIC_CORNERS)

        # Pre-compute court lines in schematic space
        self.schematic_lines = {}
        for name, (p1, p2) in COURT_LINES_REF.items():
            sp1 = self._transform_point(p1)
            sp2 = self._transform_point(p2)
            self.schematic_lines[name] = (sp1, sp2)

        # Pre-compute court surface polygon (reorder to clockwise for proper fill)
        corners = self._transform_points(REF_CORNERS)
        self.court_poly = np.array([corners[0], corners[1], corners[3], corners[2]])

        # Pre-compute intersection dots
        intersection_pts_ref = set()
        for name, (p1, p2) in COURT_LINES_REF.items():
            if name == 'net':
                continue
            intersection_pts_ref.add(p1)
            intersection_pts_ref.add(p2)
        extra_intersections = [
            (423, 1748), (1242, 1748), (832, 1748),
            (832, 561), (832, 2935),
        ]
        for p in extra_intersections:
            intersection_pts_ref.add(p)
        self.intersection_dots = [self._transform_point(p) for p in intersection_pts_ref]

        # Pre-compute net endpoints in schematic space (for upright net)
        self.net_left = self._transform_point((186, 1748))
        self.net_right = self._transform_point((1479, 1748))

        # Pre-compute schematic sideline polylines for court width lookup
        n_samples = 200
        ref_ys = np.linspace(561, 2935, n_samples)
        left_pts = np.array([[[286, y]] for y in ref_ys], dtype=np.float32)
        right_pts = np.array([[[1379, y]] for y in ref_ys], dtype=np.float32)
        left_schem = cv2.perspectiveTransform(left_pts, self.H_ref_to_schematic).reshape(-1, 2)
        right_schem = cv2.perspectiveTransform(right_pts, self.H_ref_to_schematic).reshape(-1, 2)
        # Sort by schematic y for interpolation
        left_order = np.argsort(left_schem[:, 1])
        right_order = np.argsort(right_schem[:, 1])
        self._schem_left_y = left_schem[left_order, 1]
        self._schem_left_x = left_schem[left_order, 0]
        self._schem_right_y = right_schem[right_order, 1]
        self._schem_right_x = right_schem[right_order, 0]

        # Camera court geometry (set when precompute_camera_court_geometry is called)
        self._cam_left_y = None
        self._cam_left_x = None
        self._cam_right_y = None
        self._cam_right_x = None

    def _transform_point(self, pt):
        """Transform a single point from ref coords to schematic coords."""
        p = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        tp = cv2.perspectiveTransform(p, self.H_ref_to_schematic)
        return (int(tp[0, 0, 0]), int(tp[0, 0, 1]))

    def _transform_point_float(self, pt):
        """Transform a single point, return float coords."""
        p = np.array([[[pt[0], pt[1]]]], dtype=np.float32)
        tp = cv2.perspectiveTransform(p, self.H_ref_to_schematic)
        return (float(tp[0, 0, 0]), float(tp[0, 0, 1]))

    def _transform_points(self, pts):
        """Transform array of points from ref coords to schematic coords."""
        p = pts.reshape(-1, 1, 2).astype(np.float32)
        tp = cv2.perspectiveTransform(p, self.H_ref_to_schematic)
        return tp.reshape(-1, 2).astype(np.int32)

    def precompute_camera_court_geometry(self, court_warp_matrix):
        """
        Project left and right sidelines into camera space from frame 0.
        Call once — camera is fixed so this remains valid for all frames.
        """
        n_samples = 200
        ref_ys = np.linspace(561, 2935, n_samples)
        left_pts = np.array([[[286, y]] for y in ref_ys], dtype=np.float32)
        right_pts = np.array([[[1379, y]] for y in ref_ys], dtype=np.float32)
        left_cam = cv2.perspectiveTransform(left_pts, court_warp_matrix).reshape(-1, 2)
        right_cam = cv2.perspectiveTransform(right_pts, court_warp_matrix).reshape(-1, 2)
        # Sort by camera y for interpolation
        left_order = np.argsort(left_cam[:, 1])
        right_order = np.argsort(right_cam[:, 1])
        self._cam_left_y = left_cam[left_order, 1]
        self._cam_left_x = left_cam[left_order, 0]
        self._cam_right_y = right_cam[right_order, 1]
        self._cam_right_x = right_cam[right_order, 0]

    def get_camera_court_width(self, cam_y):
        """Return court width in camera pixels at the given camera y-coordinate."""
        left_x = np.interp(cam_y, self._cam_left_y, self._cam_left_x)
        right_x = np.interp(cam_y, self._cam_right_y, self._cam_right_x)
        return max(right_x - left_x, 10.0)

    def get_schematic_court_width(self, schem_y):
        """Return court width in schematic pixels at the given schematic y-coordinate."""
        left_x = np.interp(schem_y, self._schem_left_y, self._schem_left_x)
        right_x = np.interp(schem_y, self._schem_right_y, self._schem_right_x)
        return max(right_x - left_x, 10.0)

    def transform_foot_to_schematic(self, foot_camera, court_warp_matrix):
        """
        Transform a foot position from camera coords to schematic coords.
        Only the foot touches the ground plane, so only the foot goes through
        the homography.

        court_warp_matrix maps ref→camera, so we invert it to get camera→ref,
        then chain with H_ref_to_schematic.
        """
        cam_to_ref = cv2.invert(court_warp_matrix)[1]
        H_combined = self.H_ref_to_schematic @ cam_to_ref
        p = np.array([[[foot_camera[0], foot_camera[1]]]], dtype=np.float32)
        tp = cv2.perspectiveTransform(p, H_combined)
        return (float(tp[0, 0, 0]), float(tp[0, 0, 1]))

    def compute_upright_skeleton(self, keypoints_camera, foot_schematic, foot_cam_y=None):
        """
        Compute upright skeleton positions in schematic space.

        Scaling is derived from the court geometry visible in the video:
        pixel_scale = schematic_court_width / camera_court_width at the
        player's depth. This naturally handles perspective.

        Args:
            keypoints_camera: (17, 3) — x, y, conf in camera pixel coords
            foot_schematic: (x, y) — foot anchor in schematic coords
            foot_cam_y: float — camera y-coordinate of foot (for court width lookup)

        Returns:
            (17, 2) — keypoint positions in schematic coords
        """
        kps = keypoints_camera
        fx, fy = foot_schematic

        # Find foot center in camera space (average of visible ankles)
        ankle_indices = [15, 16]  # left_ankle, right_ankle
        visible_ankles = [kps[i] for i in ankle_indices if kps[i, 2] > 0.3]
        if len(visible_ankles) > 0:
            foot_cam = np.mean([a[:2] for a in visible_ankles], axis=0)
        else:
            visible = kps[kps[:, 2] > 0.3]
            if len(visible) > 0:
                foot_cam = np.array([np.mean(visible[:, 0]), np.max(visible[:, 1])])
            else:
                return np.full((17, 2), -1)

        # Use camera y for court width lookup (prefer explicit, fallback to foot_cam)
        cam_y = foot_cam_y if foot_cam_y is not None else foot_cam[1]

        # Video-derived scaling: ratio of schematic to camera court width
        if self._cam_left_y is not None:
            W_cam = self.get_camera_court_width(cam_y)
            W_schem = self.get_schematic_court_width(fy)
            pixel_scale = W_schem / W_cam
        else:
            # Fallback if camera geometry not precomputed (e.g. standalone tests)
            pixel_scale = 0.5

        # Compute schematic positions for each keypoint
        result = np.zeros((17, 2))
        for i in range(17):
            dx = kps[i, 0] - foot_cam[0]  # horizontal offset from foot
            dy = foot_cam[1] - kps[i, 1]  # height above foot (positive = up)
            result[i, 0] = fx + dx * pixel_scale      # x: same direction
            result[i, 1] = fy - dy * pixel_scale      # y: up = negative in image
        return result

    def draw_court(self, canvas):
        """Draw the court surface, lines, upright net band, and intersection dots."""
        # Court surface
        cv2.fillConvexPoly(canvas, self.court_poly, COURT_COLOR, cv2.LINE_AA)

        # Court lines (except net)
        for name, (p1, p2) in self.schematic_lines.items():
            if name == 'net':
                continue
            cv2.line(canvas, p1, p2, LINE_COLOR, 2, cv2.LINE_AA)

        # Upright net band — height derived from court proportions
        nl = self.net_left
        nr = self.net_right
        net_height_ratio = NET_HEIGHT_REF / REF_COURT_WIDTH
        net_h_left = int(net_height_ratio * self.get_schematic_court_width(nl[1]))
        net_h_right = int(net_height_ratio * self.get_schematic_court_width(nr[1]))
        net_poly = np.array([
            [nl[0], nl[1]],                          # bottom-left (on court)
            [nr[0], nr[1]],                          # bottom-right (on court)
            [nr[0], nr[1] - net_h_right],            # top-right
            [nl[0], nl[1] - net_h_left],             # top-left
        ], dtype=np.int32)

        # Draw as semi-transparent filled polygon
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [net_poly], NET_COLOR)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

        # Net line at the base
        net_p1, net_p2 = self.schematic_lines['net']
        cv2.line(canvas, net_p1, net_p2, LINE_COLOR, 2, cv2.LINE_AA)
        # Net top line
        cv2.line(canvas, (nl[0], nl[1] - net_h_left),
                 (nr[0], nr[1] - net_h_right), LINE_COLOR, 1, cv2.LINE_AA)

        # Intersection dots
        for dot in self.intersection_dots:
            cv2.circle(canvas, dot, 4, DOT_COLOR, -1, cv2.LINE_AA)

    def draw_skeleton(self, canvas, keypoints_schematic, color, conf_threshold=0.3,
                      original_kps=None):
        """Draw 17-keypoint stick figure on canvas (upright positions)."""
        kps = keypoints_schematic
        confs = original_kps[:, 2] if original_kps is not None else np.ones(17)

        for a, b in SKELETON_CONNECTIONS:
            if confs[a] > conf_threshold and confs[b] > conf_threshold:
                p1 = (int(kps[a, 0]), int(kps[a, 1]))
                p2 = (int(kps[b, 0]), int(kps[b, 1]))
                cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)

        for j in range(17):
            if confs[j] > conf_threshold:
                pt = (int(kps[j, 0]), int(kps[j, 1]))
                cv2.circle(canvas, pt, 3, (255, 255, 255), -1, cv2.LINE_AA)

    def draw_player_marker(self, canvas, position, player_id, color, radius=12):
        """Draw a filled circle at foot position."""
        x, y = int(position[0]), int(position[1])
        cv2.circle(canvas, (x, y), radius, color, -1, cv2.LINE_AA)

    def draw_player_bbox(self, canvas, keypoints_schematic, color, original_kps=None,
                         conf_threshold=0.3):
        """Draw a bounding box around visible upright keypoints."""
        confs = original_kps[:, 2] if original_kps is not None else np.ones(17)
        visible = keypoints_schematic[confs > conf_threshold]
        if len(visible) < 2:
            return
        x_min, y_min = visible.min(axis=0).astype(int)
        x_max, y_max = visible.max(axis=0).astype(int)
        pad = 8
        cv2.rectangle(canvas, (x_min - pad, y_min - pad), (x_max + pad, y_max + pad),
                      color, 1, cv2.LINE_AA)

    def draw_ball(self, canvas, position, radius=5):
        """Draw ball marker with small bbox."""
        x, y = int(position[0]), int(position[1])
        pad = 12
        cv2.rectangle(canvas, (x - pad, y - pad), (x + pad, y + pad),
                      BALL_BBOX_COLOR, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), radius, BALL_COLOR, -1, cv2.LINE_AA)

    def draw_trail(self, canvas, positions, color, max_len=30):
        """Draw polyline trail from recent positions."""
        pts = [(int(p[0]), int(p[1])) for p in positions[-max_len:] if p is not None]
        if len(pts) < 2:
            return
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], color, 1, cv2.LINE_AA)

    def render_frame(self, frame_num, players=None, ball_pos=None,
                     player_trails=None, ball_trail=None):
        """
        Render a complete schematic frame.

        Args:
            frame_num: int
            players: list of dicts with:
                'keypoints_schematic': (17, 2) upright positions
                'original_kps': (17, 3) with confidence
                'id': int
                'color': BGR tuple
                'foot_pos': (x, y) schematic foot anchor
            ball_pos: (x, y) in schematic coords or None
            player_trails: dict of player_id -> list of (x,y)
            ball_trail: list of (x,y)
        """
        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_COLOR, dtype=np.uint8)
        self.draw_court(canvas)

        if player_trails:
            for pid, trail in player_trails.items():
                self.draw_trail(canvas, trail, TRAIL_COLOR)

        if ball_trail:
            self.draw_trail(canvas, ball_trail, TRAIL_COLOR, max_len=15)

        if players:
            for p in players:
                color = p.get('color', NEAR_COLOR)
                kps_s = p['keypoints_schematic']
                orig_kps = p.get('original_kps', None)
                self.draw_player_bbox(canvas, kps_s, color, orig_kps)
                self.draw_skeleton(canvas, kps_s, color, original_kps=orig_kps)
                if 'foot_pos' in p:
                    self.draw_player_marker(canvas, p['foot_pos'], p['id'], color,
                                            radius=12 if p['id'] == 2 else 8)

        if ball_pos is not None:
            self.draw_ball(canvas, ball_pos)

        cv2.putText(canvas, f'frame: {frame_num}', (10, CANVAS_H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1, cv2.LINE_AA)

        return canvas


if __name__ == '__main__':
    renderer = SchematicRenderer()

    # --- Empty court test ---
    canvas = renderer.render_frame(frame_num=0)
    cv2.imwrite('test_schematic_empty.png', canvas)
    print('Saved test_schematic_empty.png')

    # --- Test with upright fake players ---
    # Simulate camera-space keypoints for a standing person
    # Offsets from foot center (camera pixels): dx, dy (dy negative = above foot)
    skeleton_offsets = [
        (0, -140),                     # 0  nose
        (-5, -148), (5, -148),         # 1  left_eye, 2 right_eye
        (-12, -142), (12, -142),       # 3  left_ear, 4 right_ear
        (-25, -110), (25, -110),       # 5  left_shoulder, 6 right_shoulder
        (-40, -70), (40, -70),         # 7  left_elbow, 8 right_elbow
        (-50, -30), (50, -30),         # 9  left_wrist, 10 right_wrist
        (-15, -50), (15, -50),         # 11 left_hip, 12 right_hip
        (-18, 0), (18, 0),             # 13 left_knee, 14 right_knee (at foot level)
        (-18, 30), (18, 30),           # 15 left_ankle, 16 right_ankle
    ]

    # Far player — foot at ref coords (832, 700), camera-like kps
    far_foot_ref = (832, 700)
    far_foot_schem = renderer._transform_point_float(far_foot_ref)
    far_kps_camera = np.zeros((17, 3), dtype=np.float32)
    for i, (dx, dy) in enumerate(skeleton_offsets):
        far_kps_camera[i] = [400 + dx, 300 - dy, 1.0]  # fake camera coords
    # Foot center in camera: (400, 330)
    far_kps_schematic = renderer.compute_upright_skeleton(far_kps_camera, far_foot_schem)

    # Near player — foot at ref coords (900, 2600)
    near_foot_ref = (900, 2600)
    near_foot_schem = renderer._transform_point_float(near_foot_ref)
    near_kps_camera = np.zeros((17, 3), dtype=np.float32)
    for i, (dx, dy) in enumerate(skeleton_offsets):
        near_kps_camera[i] = [600 + dx, 500 - dy, 1.0]
    near_kps_schematic = renderer.compute_upright_skeleton(near_kps_camera, near_foot_schem)

    players = [
        {
            'keypoints_schematic': far_kps_schematic,
            'original_kps': far_kps_camera,
            'id': 1,
            'color': FAR_COLOR,
            'foot_pos': far_foot_schem,
        },
        {
            'keypoints_schematic': near_kps_schematic,
            'original_kps': near_kps_camera,
            'id': 2,
            'color': NEAR_COLOR,
            'foot_pos': near_foot_schem,
        },
    ]

    ball_schem = renderer._transform_point((750, 1500))
    canvas = renderer.render_frame(frame_num=0, players=players, ball_pos=ball_schem)
    cv2.imwrite('test_schematic_with_players.png', canvas)
    print('Saved test_schematic_with_players.png')
