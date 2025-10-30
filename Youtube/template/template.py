#!/usr/bin/env python3
"""
Video Portrait Cropping Template with Face Tracking
Automatically crops landscape videos to portrait format with intelligent face tracking.
"""

import cv2
import json
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import insightface
from insightface.app import FaceAnalysis

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Output dimensions (9:16 portrait)
PORTRAIT_WIDTH = 1080
PORTRAIT_HEIGHT = 1920
TARGET_RATIO = PORTRAIT_HEIGHT / PORTRAIT_WIDTH

# Face detection & tracking
ZOOM_PADDING = 3.0  # Padding around face (multiplier)
SMOOTH_FACTOR = 0.08  # Smoothing between frames (0-1)
MIN_FACE_SIZE = 20  # Minimum face size in pixels
MAX_FACE_SIZE_RATIO = 0.9  # Maximum face size relative to frame
MIN_DETECTION_SCORE = 0.3  # Minimum confidence score
MIN_MOVEMENT_SCORE = 0.001  # Minimum movement threshold

# Segment analysis
ANALYSIS_SAMPLE_RATE = 10  # Sample every N frames
MIN_FACE_SAMPLES = 2  # Minimum samples needed per segment

# Jump cut detection
JUMP_CUT_THRESHOLD = 30.0  # Change percentage threshold
JUMP_CUT_MIN_FRAMES = 10  # Minimum frames between cuts

# Transition modes
ENABLE_TRANSITION_ANIMATION = False  # True = smooth animation, False = instant
JUMP_CUT_SMOOTH_FRAMES = 10  # Frames for smooth transition
JUMP_CUT_TRACK_FACE = True  # Track face during transition
FACE_TRACKING_WEIGHT_START = 0.7  # Initial face tracking weight
FACE_TRACKING_SMOOTHING = 0.3  # Face tracking smoothing factor
SMOOTH_JUMP_CUT_BLEND_FRAMES = 3  # Frames for instant blend

# InsightFace model
SCRFD_MODEL = 'buffalo_sc'
DETECTION_SIZE = 320

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_tmp_dir(output_dir: Path) -> Path:
    """Create and return temporary directory."""
    tmp_dir = output_dir / ".tmp"
    tmp_dir.mkdir(exist_ok=True)
    return tmp_dir

def ms_to_seconds(milliseconds: float) -> float:
    """Convert milliseconds to seconds."""
    return milliseconds / 1000.0

def format_time(milliseconds: float) -> str:
    """Convert milliseconds to HH:MM:SS.mmm format for FFmpeg."""
    seconds = milliseconds / 1000.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def get_video_info(video_path: Path) -> Dict[str, Any]:
    """Get video metadata (width, height, fps, frame_count)."""
    cap = cv2.VideoCapture(str(video_path))
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    cap.release()
    return info

# ============================================================================
# FFMPEG OPERATIONS
# ============================================================================

def run_ffmpeg_command(cmd: List[str], operation_name: str) -> bool:
    """Run FFmpeg command with error handling."""
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR {operation_name}: {e.stderr}")
        return False

def cut_video_segment(video_src: Path, start_ms: float, end_ms: float, output: Path) -> bool:
    """Cut video segment using milliseconds."""
    duration_ms = end_ms - start_ms
    cmd = [
        "ffmpeg", "-y",
        "-ss", format_time(start_ms),
        "-i", str(video_src),
        "-t", format_time(duration_ms),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-avoid_negative_ts", "1",
        str(output)
    ]
    return run_ffmpeg_command(cmd, "cutting video")

def add_audio_to_video(video_path: Path, audio_src: Path, output_path: Path,
                      start_time_ms: float, duration_ms: float) -> bool:
    """Add audio to video using milliseconds."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ss", format_time(start_time_ms),
        "-i", str(audio_src),
        "-t", format_time(duration_ms),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        str(output_path)
    ]
    return run_ffmpeg_command(cmd, "adding audio")

# ============================================================================
# FACE DETECTION
# ============================================================================

class SCRFDFaceDetector:
    """Face detector using InsightFace SCRFD model."""
    
    def __init__(self):
        print(f"    [INIT] Loading InsightFace SCRFD model: {SCRFD_MODEL}...")
        self.app = FaceAnalysis(
            name=SCRFD_MODEL,
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0, det_size=(DETECTION_SIZE, DETECTION_SIZE))
        # Keep only detection model
        self.app.models = {k: v for k, v in self.app.models.items() if 'det' in k}
        print(f"    [INIT] SCRFD model loaded successfully!")
    
    def detect(self, frame, prev_frame=None) -> List[Dict[str, Any]]:
        """Detect faces in frame and validate if they are real humans."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_faces = self.app.get(rgb_frame)
        
        faces = []
        h, w = frame.shape[:2]
        
        for face in detected_faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            x, y, w_face, h_face = x1, y1, x2 - x1, y2 - y1
            
            if w_face <= 0 or h_face <= 0:
                continue
            
            face_info = {
                'bbox': (x, y, w_face, h_face),
                'det_score': float(face.det_score),
                'landmarks': face.kps.astype(int) if hasattr(face, 'kps') else None,
                'movement_score': 0.0,
                'is_real_human': False
            }
            
            # Validate if face is real human
            face_info['is_real_human'] = self._validate_real_human(
                face_info, frame, prev_frame, w, h
            )
            
            faces.append(face_info)
        
        return faces
    
    def _validate_real_human(self, face: Dict[str, Any], frame, prev_frame, 
                            frame_w: int, frame_h: int) -> bool:
        """Validate if detected face is a real human."""
        x, y, w, h = face['bbox']
        
        # Check detection score
        if face['det_score'] < MIN_DETECTION_SCORE:
            return False
        
        # Check size constraints
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            return False
        
        face_area = w * h
        frame_area = frame_w * frame_h
        size_ratio = face_area / frame_area
        
        if size_ratio > MAX_FACE_SIZE_RATIO:
            return False
        
        # Calculate movement score
        if prev_frame is not None:
            face['movement_score'] = self._calculate_movement(face, frame, prev_frame)
        else:
            face['movement_score'] = 0.1
        
        return True
    
    def _calculate_movement(self, face: Dict[str, Any], frame, prev_frame) -> float:
        """Calculate movement score in face region."""
        x, y, w, h = face['bbox']
        
        # Clip coordinates
        x = max(0, x)
        y = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        
        face_region = frame[y:y2, x:x2]
        prev_face_region = prev_frame[y:y2, x:x2]
        
        if face_region.shape != prev_face_region.shape or face_region.size == 0:
            return 0.0
        
        # Calculate difference
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_face_region, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        
        return diff.mean() / 255.0

# ============================================================================
# JUMP CUT DETECTION
# ============================================================================

class JumpCutDetector:
    """Detect jump cuts (scene changes) in video."""
    
    def __init__(self, threshold: float = JUMP_CUT_THRESHOLD):
        self.threshold = threshold
        self.last_jump_frame = -JUMP_CUT_MIN_FRAMES
    
    def detect(self, frame, prev_frame, frame_num: int) -> bool:
        """Detect if current frame is a jump cut."""
        if prev_frame is None:
            return False
        
        # Enforce minimum frames between cuts
        if frame_num - self.last_jump_frame < JUMP_CUT_MIN_FRAMES:
            return False
        
        # Convert to grayscale and resize for performance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (160, 90))
        prev_gray_small = cv2.resize(prev_gray, (160, 90))
        
        # Calculate change percentage
        diff = cv2.absdiff(gray_small, prev_gray_small)
        total_pixels = gray_small.size
        changed_pixels = np.count_nonzero(diff > 30)
        change_percentage = (changed_pixels / total_pixels) * 100
        
        if change_percentage > self.threshold:
            self.last_jump_frame = frame_num
            return True
        
        return False

# ============================================================================
# SEGMENT ANALYSIS
# ============================================================================

class SegmentAnalyzer:
    """Analyze video segments to find optimal face positions."""
    
    def __init__(self, video_width: int, video_height: int):
        self.video_width = video_width
        self.video_height = video_height
    
    def analyze_segment(self, cap: cv2.VideoCapture, start_frame: int, 
                       end_frame: int, detector: SCRFDFaceDetector) -> Optional[Tuple[int, int, int, int]]:
        """Analyze segment and return optimal crop region."""
        print(f"\n    [ANALYSIS] Analyzing segment: frame {start_frame} to {end_frame}")
        
        # Collect face positions
        face_positions = self._collect_face_positions(
            cap, start_frame, end_frame, detector
        )
        
        if len(face_positions) < MIN_FACE_SAMPLES:
            print(f"    [ANALYSIS] Not enough samples (min: {MIN_FACE_SAMPLES})")
            return None
        
        # Filter outliers
        face_positions = self._filter_outliers(face_positions)
        if not face_positions:
            print(f"    [ANALYSIS] All samples filtered as outliers")
            return None
        
        # Calculate optimal crop
        crop = self._calculate_optimal_crop(face_positions)
        
        print(f"    [ANALYSIS] Final crop - X: {crop[0]}, Y: {crop[1]}, "
              f"W: {crop[2]}, H: {crop[3]}")
        
        return crop
    
    def _collect_face_positions(self, cap: cv2.VideoCapture, start_frame: int,
                               end_frame: int, detector: SCRFDFaceDetector) -> List[Dict]:
        """Sample frames and collect face positions."""
        positions = []
        prev_frame = None
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_offset in range(0, end_frame - start_frame, ANALYSIS_SAMPLE_RATE):
            current_frame_num = start_frame + frame_offset
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Detect faces
            faces = detector.detect(frame, prev_frame)
            real_humans = [f for f in faces if f['is_real_human']]
            
            if real_humans:
                # Use best detection score
                best_face = max(real_humans, key=lambda f: f['det_score'])
                x, y, w, h = best_face['bbox']
                
                positions.append({
                    'center_x': x + w // 2,
                    'center_y': y + h // 2,
                    'w': w,
                    'h': h,
                    'det_score': best_face['det_score'],
                    'movement': best_face['movement_score']
                })
            
            prev_frame = frame.copy()
        
        print(f"    [ANALYSIS] Found {len(positions)} REAL face samples")
        return positions
    
    def _filter_outliers(self, positions: List[Dict]) -> List[Dict]:
        """Filter outlier positions using standard deviation."""
        if len(positions) < 4:
            return positions
        
        x_values = [p['center_x'] for p in positions]
        median_x = np.median(x_values)
        std_x = np.std(x_values)
        
        # Keep positions within 2 standard deviations
        filtered = [p for p in positions if abs(p['center_x'] - median_x) <= 2 * std_x]
        return filtered if filtered else positions
    
    def _calculate_optimal_crop(self, positions: List[Dict]) -> Tuple[int, int, int, int]:
        """Calculate optimal crop region from face positions."""
        # Weight by detection score and take top 70%
        positions_weighted = sorted(positions, key=lambda p: p['det_score'], reverse=True)
        top_count = max(int(len(positions_weighted) * 0.7), MIN_FACE_SAMPLES)
        top_positions = positions_weighted[:top_count]
        
        # Calculate median position and size
        median_x = int(np.median([p['center_x'] for p in top_positions]))
        median_y = int(np.median([p['center_y'] for p in top_positions]))
        median_w = int(np.median([p['w'] for p in top_positions]))
        median_h = int(np.median([p['h'] for p in top_positions]))
        
        print(f"    [ANALYSIS] Best position - X: {median_x}, Y: {median_y}, "
              f"Size: {median_w}x{median_h}")
        
        # Calculate crop dimensions
        crop_height = int(median_h * ZOOM_PADDING * 1.8)
        crop_width = int(crop_height / TARGET_RATIO)
        
        # Constrain to video dimensions
        if crop_width > self.video_width:
            crop_width = self.video_width
            crop_height = int(crop_width * TARGET_RATIO)
        
        if crop_height > self.video_height:
            crop_height = self.video_height
            crop_width = int(crop_height / TARGET_RATIO)
        
        # Calculate position (face in upper third)
        crop_x = median_x - crop_width // 2
        crop_y = median_y - int(crop_height * 0.35)
        
        # Clamp to valid range
        crop_x = max(0, min(crop_x, self.video_width - crop_width))
        crop_y = max(0, min(crop_y, self.video_height - crop_height))
        
        return (crop_x, crop_y, crop_width, crop_height)

# ============================================================================
# CROP CALCULATOR
# ============================================================================

class CropCalculator:
    """Calculate crop regions with smooth transitions."""
    
    def __init__(self, video_width: int, video_height: int):
        self.video_width = video_width
        self.video_height = video_height
        self.last_crop = None
        self.current_segment_crop = None
        
        # Transition state
        self.transition_frames_remaining = 0
        self.transition_start_crop = None
        self.transition_target_crop = None
        self.in_face_tracking_transition = False
        self.tracking_progress = 0.0
        self.last_face_pos = None
        
        # Blend state (for instant mode)
        self.blend_frames_remaining = 0
        self.blend_start_crop = None
        self.blend_target_crop = None
    
    def set_segment_crop(self, crop: Tuple[int, int, int, int]):
        """Set new crop region for segment."""
        if self.last_crop is not None:
            if ENABLE_TRANSITION_ANIMATION:
                # Smooth animated transition
                self.transition_start_crop = self.last_crop
                self.transition_target_crop = crop
                self.transition_frames_remaining = JUMP_CUT_SMOOTH_FRAMES
                self.in_face_tracking_transition = JUMP_CUT_TRACK_FACE
                self.tracking_progress = 0.0
                self.last_face_pos = None
            else:
                # Instant transition with blend
                self.blend_start_crop = self.last_crop
                self.blend_target_crop = crop
                self.blend_frames_remaining = SMOOTH_JUMP_CUT_BLEND_FRAMES
                self.current_segment_crop = crop
        else:
            self.current_segment_crop = crop
            self.last_crop = crop
    
    def calculate(self, current_face_pos: Optional[Tuple[int, int]] = None) -> Tuple[int, int, int, int]:
        """Calculate current crop region."""
        # Handle animated transition
        if self.transition_frames_remaining > 0:
            crop = self._handle_animated_transition(current_face_pos)
            self.last_crop = crop
            return crop
        
        # Handle instant blend
        if self.blend_frames_remaining > 0:
            crop = self._handle_instant_blend()
            self.last_crop = crop
            return crop
        
        # Normal crop
        crop = self.current_segment_crop or self._center_crop()
        
        # Apply smoothing
        if self.last_crop is not None and crop != self.current_segment_crop:
            crop = self._smooth_transition(self.last_crop, crop)
        
        self.last_crop = crop
        return crop
    
    def _handle_animated_transition(self, face_pos: Optional[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """Handle smooth animated transition between segments."""
        progress = 1.0 - (self.transition_frames_remaining / JUMP_CUT_SMOOTH_FRAMES)
        self.tracking_progress = progress
        
        if self.in_face_tracking_transition and face_pos is not None:
            crop = self._track_face_transition(face_pos, progress)
        else:
            progress_eased = self._ease_in_out(progress)
            crop = self._interpolate_crop(
                self.transition_start_crop,
                self.transition_target_crop,
                progress_eased
            )
        
        self.transition_frames_remaining -= 1
        
        if self.transition_frames_remaining == 0:
            self.current_segment_crop = self.transition_target_crop
            self.in_face_tracking_transition = False
        
        return crop
    
    def _handle_instant_blend(self) -> Tuple[int, int, int, int]:
        """Handle instant transition with micro blend."""
        progress = 1.0 - (self.blend_frames_remaining / SMOOTH_JUMP_CUT_BLEND_FRAMES)
        crop = self._micro_blend_crop(
            self.blend_start_crop,
            self.blend_target_crop,
            progress
        )
        self.blend_frames_remaining -= 1
        return crop
    
    def _track_face_transition(self, face_center: Tuple[int, int], 
                              progress: float) -> Tuple[int, int, int, int]:
        """Track face during transition for natural movement."""
        face_x, face_y = face_center
        
        # Apply smoothing
        if self.last_face_pos is not None:
            last_x, last_y = self.last_face_pos
            face_x = int(last_x + (face_x - last_x) * FACE_TRACKING_SMOOTHING)
            face_y = int(last_y + (face_y - last_y) * FACE_TRACKING_SMOOTHING)
        
        self.last_face_pos = (face_x, face_y)
        
        # Interpolate crop size
        sx, sy, sw, sh = self.transition_start_crop
        tx, ty, tw, th = self.transition_target_crop
        eased_progress = self._ease_out_cubic(progress)
        
        current_w = int(sw + (tw - sw) * eased_progress)
        current_h = int(sh + (th - sh) * eased_progress)
        
        # Calculate ideal position from face
        ideal_x = face_x - current_w // 2
        ideal_y = face_y - int(current_h * 0.35)
        
        # Calculate target position
        target_x = int(sx + (tx - sx) * eased_progress)
        target_y = int(sy + (ty - sy) * eased_progress)
        
        # Blend face tracking with target
        face_weight = FACE_TRACKING_WEIGHT_START * (1.0 - eased_progress)
        target_weight = 1.0 - face_weight
        
        final_x = int(ideal_x * face_weight + target_x * target_weight)
        final_y = int(ideal_y * face_weight + target_y * target_weight)
        
        # Clamp to valid range
        final_x = max(0, min(final_x, self.video_width - current_w))
        final_y = max(0, min(final_y, self.video_height - current_h))
        
        return (final_x, final_y, current_w, current_h)
    
    def _center_crop(self) -> Tuple[int, int, int, int]:
        """Calculate center crop region."""
        crop_height = self.video_height
        crop_width = int(crop_height / TARGET_RATIO)
        
        if crop_width > self.video_width:
            crop_width = self.video_width
            crop_height = int(crop_width * TARGET_RATIO)
        
        crop_x = (self.video_width - crop_width) // 2
        crop_y = (self.video_height - crop_height) // 2
        
        return (crop_x, crop_y, crop_width, crop_height)
    
    def _smooth_transition(self, old_crop: Tuple[int, int, int, int],
                          new_crop: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Apply smooth transition between crops."""
        ox, oy, ow, oh = old_crop
        nx, ny, nw, nh = new_crop
        
        sx = int(ox + (nx - ox) * SMOOTH_FACTOR)
        sy = int(oy + (ny - oy) * SMOOTH_FACTOR)
        sw = int(ow + (nw - ow) * SMOOTH_FACTOR)
        sh = int(oh + (nh - oh) * SMOOTH_FACTOR)
        
        return (sx, sy, sw, sh)
    
    @staticmethod
    def _interpolate_crop(start: Tuple[int, int, int, int],
                         end: Tuple[int, int, int, int],
                         progress: float) -> Tuple[int, int, int, int]:
        """Linearly interpolate between two crops."""
        sx, sy, sw, sh = start
        ex, ey, ew, eh = end
        
        x = int(sx + (ex - sx) * progress)
        y = int(sy + (ey - sy) * progress)
        w = int(sw + (ew - sw) * progress)
        h = int(sh + (eh - sh) * progress)
        
        return (x, y, w, h)
    
    @staticmethod
    def _micro_blend_crop(start: Tuple[int, int, int, int],
                         end: Tuple[int, int, int, int],
                         progress: float) -> Tuple[int, int, int, int]:
        """Quick blend for instant transitions."""
        eased_progress = 1 - pow(1 - progress, 4)
        return CropCalculator._interpolate_crop(start, end, eased_progress)
    
    @staticmethod
    def _ease_in_out(t: float) -> float:
        """Ease in-out function."""
        return t * t * (3.0 - 2.0 * t)
    
    @staticmethod
    def _ease_out_cubic(t: float) -> float:
        """Ease out cubic function."""
        return 1 - pow(1 - t, 3)

# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================

def analyze_video_segments(input_path: Path) -> List[Dict[str, Any]]:
    """Analyze video to find segments and optimal face positions."""
    print(f"  [PHASE 1] Analyzing video segments...")
    
    info = get_video_info(input_path)
    cap = cv2.VideoCapture(str(input_path))
    
    if not cap.isOpened():
        print(f"    ERROR: Cannot open video file")
        return []
    
    # Initialize
    detector = SCRFDFaceDetector()
    jump_detector = JumpCutDetector()
    analyzer = SegmentAnalyzer(info['width'], info['height'])
    
    # Step 1: Detect jump cuts
    print(f"    [1/2] Detecting jump cuts...")
    jump_cuts = _detect_jump_cuts(cap, jump_detector, info['frame_count'])
    print(f"      Found {len(jump_cuts) - 1} segments")
    
    # Step 2: Analyze each segment
    print(f"    [2/2] Analyzing face positions per segment...")
    segments = _analyze_segments(cap, jump_cuts, analyzer, detector)
    
    cap.release()
    print(f"\n  [PHASE 1] Complete - {len(segments)} segments identified")
    
    return segments

def _detect_jump_cuts(cap: cv2.VideoCapture, detector: JumpCutDetector,
                     total_frames: int) -> List[int]:
    """Detect all jump cuts in video."""
    jump_cuts = [0]
    frame_num = 0
    prev_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        # Progress indicator
        if frame_num % 100 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"      Progress: {progress:.1f}%", end='\r')
        
        # Detect jump cut
        if detector.detect(frame, prev_frame, frame_num):
            jump_cuts.append(frame_num)
        
        prev_frame = frame.copy()
    
    jump_cuts.append(total_frames)
    print(f"\n      Found {len(jump_cuts) - 2} jump cuts")
    
    return jump_cuts

def _analyze_segments(cap: cv2.VideoCapture, jump_cuts: List[int],
                     analyzer: SegmentAnalyzer, detector: SCRFDFaceDetector) -> List[Dict]:
    """Analyze each segment for optimal crop."""
    segments = []
    
    for i in range(len(jump_cuts) - 1):
        start_frame = jump_cuts[i]
        end_frame = jump_cuts[i + 1]
        
        print(f"\n      Segment {i + 1}/{len(jump_cuts) - 1}: "
              f"frames {start_frame} to {end_frame}")
        
        crop = analyzer.analyze_segment(cap, start_frame, end_frame, detector)
        
        segments.append({
            'start_frame': start_frame,
            'end_frame': end_frame,
            'crop': crop,
            'is_valid': crop is not None
        })
        
        status = "✓ Valid" if crop else "✗ No valid face (fallback to center)"
        print(f"      {status}")
    
    return segments

def process_video_with_segments(input_path: Path, output_path: Path,
                                segments: List[Dict[str, Any]]) -> bool:
    """Process video with analyzed segments."""
    print(f"\n  [PHASE 2] Processing video...")
    
    info = get_video_info(input_path)
    cap = cv2.VideoCapture(str(input_path))
    
    if not cap.isOpened():
        print(f"    ERROR: Cannot open video file")
        return False
    
    # Initialize
    calculator = CropCalculator(info['width'], info['height'])
    detector = SCRFDFaceDetector()
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        info['fps'],
        (PORTRAIT_WIDTH, PORTRAIT_HEIGHT)
    )
    
    if not out.isOpened():
        print(f"    ERROR: Cannot create output video")
        cap.release()
        return False
    
    # Set initial crop
    if segments and segments[0]['is_valid']:
        calculator.set_segment_crop(segments[0]['crop'])
        print(f"    [SEGMENT 1] Using analyzed face position")
    else:
        print(f"    [SEGMENT 1] Using center crop")
    
    # Process frames
    frame_num = 0
    total_frames = info['frame_count']
    current_segment_idx = 0
    prev_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Progress indicator
        if frame_num % 30 == 0 or frame_num == total_frames - 1:
            _print_progress(frame_num, total_frames, current_segment_idx, 
                          len(segments), calculator)
        
        # Check for segment change
        if current_segment_idx < len(segments) - 1:
            next_segment = segments[current_segment_idx + 1]
            
            if frame_num >= next_segment['start_frame']:
                current_segment_idx += 1
                _handle_segment_transition(calculator, next_segment, current_segment_idx)
        
        # Detect face if needed for tracking
        face_center = None
        if ENABLE_TRANSITION_ANIMATION and calculator.in_face_tracking_transition:
            face_center = _get_face_center(frame, prev_frame, detector)
        
        # Calculate crop and apply
        crop_x, crop_y, crop_w, crop_h = calculator.calculate(face_center)
        cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        resized = cv2.resize(cropped, (PORTRAIT_WIDTH, PORTRAIT_HEIGHT))
        
        out.write(resized)
        
        prev_frame = frame.copy()
        frame_num += 1
    
    cap.release()
    out.release()
    
    print(f"\n    Progress: 100.0% - DONE")
    print(f"    Total segments: {len(segments)}")
    
    # Validate output
    if not output_path.exists() or output_path.stat().st_size < 1000:
        print(f"    ERROR: Output file is invalid")
        return False
    
    return True

def _print_progress(frame_num: int, total_frames: int, segment_idx: int,
                   total_segments: int, calculator: CropCalculator):
    """Print processing progress."""
    progress = (frame_num / total_frames) * 100
    seg_info = f"Segment {segment_idx + 1}/{total_segments}"
    trans_info = ""
    
    if ENABLE_TRANSITION_ANIMATION:
        if calculator.transition_frames_remaining > 0:
            trans_info = f" | Transition: {calculator.tracking_progress*100:.0f}%"
    else:
        if calculator.blend_frames_remaining > 0:
            trans_info = f" | Blending..."
    
    print(f"    Progress: {progress:.1f}% ({frame_num}/{total_frames}) | "
          f"{seg_info}{trans_info}", end='\r')

def _handle_segment_transition(calculator: CropCalculator, next_segment: Dict,
                              segment_idx: int):
    """Handle transition to next segment."""
    if next_segment['is_valid']:
        calculator.set_segment_crop(next_segment['crop'])
        mode = "smooth animation" if ENABLE_TRANSITION_ANIMATION else "instant blend"
        print(f"\n    [SEGMENT {segment_idx + 1}] Jump cut - {mode}")
    else:
        calculator.set_segment_crop(calculator._center_crop())
        print(f"\n    [SEGMENT {segment_idx + 1}] Jump cut - center crop")

def _get_face_center(frame, prev_frame, detector: SCRFDFaceDetector) -> Optional[Tuple[int, int]]:
    """Get center of best detected face."""
    faces = detector.detect(frame, prev_frame)
    real_humans = [f for f in faces if f['is_real_human']]
    
    if real_humans:
        best_face = max(real_humans, key=lambda f: f['det_score'])
        x, y, w, h = best_face['bbox']
        return (x + w // 2, y + h // 2)
    
    return None

# ============================================================================
# MAIN TEMPLATE FUNCTION
# ============================================================================

def apply(paths: Dict[str, Path], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main template function called by wrapper.
    
    Args:
        paths: Dictionary containing:
            - video_src: Source video path
            - output_dir: Output directory
            - clips_dir: Clips directory
        metadata: Dictionary containing:
            - candidates: List of clips with start/stop times in milliseconds
    
    Returns:
        Dictionary with processing results
    """
    print("=" * 60)
    print("VIDEO PORTRAIT CROPPING TEMPLATE")
    mode = "Smooth Animation" if ENABLE_TRANSITION_ANIMATION else "Instant + Blend"
    print(f"Mode: {mode}")
    print("=" * 60)
    
    # Extract paths
    video_src = paths["video_src"]
    output_dir = paths["output_dir"]
    clips_dir = paths["clips_dir"]
    
    # Setup temp directory
    tmp_dir = ensure_tmp_dir(output_dir)
    print(f"\n[SETUP] Temp directory: {tmp_dir}")
    
    # Get candidates
    candidates = metadata.get("candidates", [])
    if not candidates:
        return {"status": "error", "message": "No candidates found"}
    
    print(f"\n[INFO] Processing {len(candidates)} clips")
    
    # Process each clip
    results = []
    
    for idx, candidate in enumerate(candidates, 1):
        print(f"\n{'=' * 60}")
        print(f"CLIP {idx}/{len(candidates)}")
        print(f"{'=' * 60}")
        
        result = _process_clip(idx, candidate, video_src, tmp_dir, clips_dir)
        
        if result:
            results.append(result)
    
    # Cleanup
    try:
        tmp_dir.rmdir()
    except:
        pass
    
    # Print summary
    _print_summary(len(candidates), len(results), clips_dir)
    
    return {
        "status": "success",
        "total_clips": len(candidates),
        "processed_clips": len(results),
        "clips": results,
        "output_dir": str(clips_dir)
    }

def _process_clip(idx: int, candidate: Dict, video_src: Path,
                 tmp_dir: Path, clips_dir: Path) -> Optional[Dict]:
    """Process a single clip."""
    # Get timestamps
    start_ms = candidate.get("start")
    stop_ms = candidate.get("stop")
    
    if start_ms is None or stop_ms is None:
        print(f"  [SKIP] Missing timestamps")
        return None
    
    duration_ms = stop_ms - start_ms
    
    print(f"  Time: {start_ms}ms - {stop_ms}ms ({duration_ms}ms)")
    print(f"       = {start_ms/1000:.2f}s - {stop_ms/1000:.2f}s ({duration_ms/1000:.2f}s)")
    
    # Define file paths
    tmp_cut = tmp_dir / f"clip_{idx:03d}_cut.mp4"
    tmp_processed = tmp_dir / f"clip_{idx:03d}_processed.mp4"
    final_output = clips_dir / f"clip_{idx:03d}_final.mp4"
    
    # Step 1: Cut segment
    print(f"\n  [STEP 1] Cutting video segment...")
    if not cut_video_segment(video_src, start_ms, stop_ms, tmp_cut):
        print(f"  [ERROR] Failed to cut video")
        return None
    print(f"  [SUCCESS] Cut saved")
    
    # Step 2: Analyze segments
    print(f"\n  [STEP 2] Analyzing segments...")
    segments = analyze_video_segments(tmp_cut)
    
    if not segments:
        print(f"  [ERROR] Failed to analyze")
        return None
    
    valid_segments = sum(1 for s in segments if s['is_valid'])
    print(f"  [SUCCESS] {len(segments)} segments ({valid_segments} with faces)")
    
    # Step 3: Process video
    print(f"\n  [STEP 3] Processing video...")
    if not process_video_with_segments(tmp_cut, tmp_processed, segments):
        print(f"  [ERROR] Failed to process")
        return None
    print(f"  [SUCCESS] Video processed")
    
    # Step 4: Add audio
    print(f"\n  [STEP 4] Adding audio...")
    if not add_audio_to_video(tmp_processed, video_src, final_output, start_ms, duration_ms):
        print(f"  [ERROR] Failed to add audio")
        return None
    print(f"  [SUCCESS] Final clip saved: {final_output.name}")
    
    # Cleanup temp files
    tmp_cut.unlink(missing_ok=True)
    tmp_processed.unlink(missing_ok=True)
    
    return {
        "clip_id": idx,
        "output": str(final_output),
        "timestart_ms": start_ms,
        "timestop_ms": stop_ms,
        "duration_ms": duration_ms,
        "timestart_s": start_ms / 1000.0,
        "timestop_s": stop_ms / 1000.0,
        "duration_s": duration_ms / 1000.0,
        "segments": len(segments),
        "valid_segments": valid_segments,
        "status": "success"
    }

def _print_summary(total: int, processed: int, clips_dir: Path):
    """Print processing summary."""
    print(f"\n{'=' * 60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total clips: {total}")
    print(f"Processed: {processed}")
    print(f"Success rate: {(processed/total*100):.1f}%")
    print(f"Output: {clips_dir}")
    print(f"{'=' * 60}")


# ============================================================================
# ENTRY POINT (for testing)
# ============================================================================

if __name__ == "__main__":
    """
    Test script - run template directly without wrapper.
    """
    print("This is a template file, use via EditingWrapper instead.")
    print("\nExample usage:")
    print("  from editing.editing import EditingWrapper")
    print("  wrapper = EditingWrapper()")
    print("  result = wrapper.execute(...)")