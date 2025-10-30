#!/usr/bin/env python3

import cv2
import json
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import insightface
from insightface.app import FaceAnalysis

PORTRAIT_WIDTH = 1080
PORTRAIT_HEIGHT = 1920
TARGET_RATIO = PORTRAIT_HEIGHT / PORTRAIT_WIDTH

ZOOM_PADDING = 3.0
SMOOTH_FACTOR = 0.08

# OPSI TRANSISI - Set False untuk disable animasi transisi
ENABLE_TRANSITION_ANIMATION = False  # True = animasi smooth, False = instant jump cut

JUMP_CUT_SMOOTH_FRAMES = 10  # Cepat: 10 frames (~0.33s di 30fps, ~0.4s di 25fps)
JUMP_CUT_TRACK_FACE = True  # Track face during jump cut transition
FACE_TRACKING_WEIGHT_START = 0.7  # Start: 70% face, 30% target (lebih stabil)
FACE_TRACKING_SMOOTHING = 0.3  # Smoothing factor untuk mengurangi shake

# Smooth jump cut tanpa animasi (analisis frame untuk mengurangi glitch)
SMOOTH_JUMP_CUT_BLEND_FRAMES = 3  # Blend 3 frame untuk mengurangi glitch visual

JUMP_CUT_THRESHOLD = 30.0
JUMP_CUT_MIN_FRAMES = 10

ANALYSIS_SAMPLE_RATE = 10
MIN_FACE_SAMPLES = 2

SCRFD_MODEL = 'buffalo_sc'
DETECTION_SIZE = 320

MIN_FACE_SIZE = 20
MAX_FACE_SIZE_RATIO = 0.9
MIN_DETECTION_SCORE = 0.3
MIN_MOVEMENT_SCORE = 0.001

def ensure_tmp_dir(output_dir: Path) -> Path:
    tmp_dir = output_dir / ".tmp"
    tmp_dir.mkdir(exist_ok=True)
    return tmp_dir

def ms_to_seconds(milliseconds: float) -> float:
    """Convert milliseconds to seconds"""
    return milliseconds / 1000.0

def format_time(milliseconds: float) -> str:
    """Convert milliseconds to HH:MM:SS.mmm format for FFmpeg"""
    seconds = milliseconds / 1000.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def cut_video_segment(video_src: Path, start_ms: float, end_ms: float, output: Path) -> bool:
    """Cut video segment using milliseconds"""
    duration_ms = end_ms - start_ms
    cmd = [
        "ffmpeg", "-y",
        "-ss", format_time(start_ms),
        "-i", str(video_src),
        "-t", format_time(duration_ms),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "128k",
        "-avoid_negative_ts", "1",
        str(output)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR cutting video: {e.stderr}")
        return False

def get_video_info(video_path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    
    cap.release()
    return info

class SCRFDFaceDetector:
    
    def __init__(self):
        print(f"    [INIT] Loading InsightFace SCRFD model: {SCRFD_MODEL}...")
        
        self.app = FaceAnalysis(
            name=SCRFD_MODEL,
            providers=['CPUExecutionProvider']
        )
        
        self.app.prepare(ctx_id=0, det_size=(DETECTION_SIZE, DETECTION_SIZE))
        self.app.models = {k: v for k, v in self.app.models.items() if 'det' in k}
        
        print(f"    [INIT] SCRFD model loaded successfully!")
        
        self.face_history = {}
    
    def detect(self, frame, prev_frame=None) -> List[Dict[str, Any]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_faces = self.app.get(rgb_frame)
        
        faces = []
        
        for face in detected_faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1
            
            if w <= 0 or h <= 0:
                continue
            
            det_score = float(face.det_score)
            landmarks = face.kps.astype(int) if hasattr(face, 'kps') else None
            
            face_info = {
                'bbox': (x, y, w, h),
                'det_score': det_score,
                'landmarks': landmarks,
                'is_real_human': False,
                'movement_score': 0.0
            }
            
            faces.append(face_info)
        
        h, w = frame.shape[:2]
        for face in faces:
            face['is_real_human'] = self._validate_real_human(face, frame, prev_frame, w, h)
        
        return faces
    
    def _validate_real_human(self, face: Dict[str, Any], frame, prev_frame, 
                            frame_w: int, frame_h: int) -> bool:
        x, y, w, h = face['bbox']
        det_score = face['det_score']
        
        if det_score < MIN_DETECTION_SCORE:
            return False
        
        face_area = w * h
        frame_area = frame_w * frame_h
        size_ratio = face_area / frame_area
        
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            return False
        
        if size_ratio > MAX_FACE_SIZE_RATIO:
            return False
        
        if prev_frame is not None:
            movement_score = self._calculate_movement(face, frame, prev_frame)
            face['movement_score'] = movement_score
        else:
            face['movement_score'] = 0.1
        
        return True
    
    def _calculate_movement(self, face: Dict[str, Any], frame, prev_frame) -> float:
        x, y, w, h = face['bbox']
        
        x = max(0, x)
        y = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        
        face_region = frame[y:y2, x:x2]
        prev_face_region = prev_frame[y:y2, x:x2]
        
        if face_region.shape != prev_face_region.shape or face_region.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_face_region, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(gray, prev_gray)
        movement = diff.mean() / 255.0
        
        return movement

class JumpCutDetector:
    
    def __init__(self, threshold: float = JUMP_CUT_THRESHOLD):
        self.threshold = threshold
        self.last_jump_frame = -JUMP_CUT_MIN_FRAMES
    
    def detect(self, frame, prev_frame, frame_num: int) -> bool:
        if prev_frame is None:
            return False
        
        if frame_num - self.last_jump_frame < JUMP_CUT_MIN_FRAMES:
            return False
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        gray_small = cv2.resize(gray, (160, 90))
        prev_gray_small = cv2.resize(prev_gray, (160, 90))
        
        diff = cv2.absdiff(gray_small, prev_gray_small)
        
        total_pixels = gray_small.size
        changed_pixels = np.count_nonzero(diff > 30)
        change_percentage = (changed_pixels / total_pixels) * 100
        
        if change_percentage > self.threshold:
            self.last_jump_frame = frame_num
            return True
        
        return False

class SegmentAnalyzer:
    
    def __init__(self, video_width: int, video_height: int):
        self.video_width = video_width
        self.video_height = video_height
    
    def analyze_segment(self, cap: cv2.VideoCapture, start_frame: int, end_frame: int, 
                       detector: SCRFDFaceDetector) -> Optional[Tuple[int, int, int, int]]:
        print(f"\n    [ANALYSIS] Analyzing segment: frame {start_frame} to {end_frame}")
        
        real_face_positions = []
        sample_count = 0
        prev_frame = None
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_offset in range(0, end_frame - start_frame, ANALYSIS_SAMPLE_RATE):
            current_frame_num = start_frame + frame_offset
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_num)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            sample_count += 1
            
            faces = detector.detect(frame, prev_frame)
            real_humans = [f for f in faces if f['is_real_human']]
            
            if real_humans:
                sorted_faces = sorted(real_humans, 
                                    key=lambda f: f['det_score'], 
                                    reverse=True)
                
                best_face = sorted_faces[0]
                x, y, w, h = best_face['bbox']
                
                real_face_positions.append({
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'center_x': x + w // 2,
                    'center_y': y + h // 2,
                    'det_score': best_face['det_score'],
                    'movement': best_face['movement_score']
                })
            
            prev_frame = frame.copy()
        
        print(f"    [ANALYSIS] Sampled {sample_count} frames, found {len(real_face_positions)} REAL faces")
        
        if len(real_face_positions) < MIN_FACE_SAMPLES:
            print(f"    [ANALYSIS] Not enough real human samples (min: {MIN_FACE_SAMPLES})")
            return None
        
        real_face_positions = self._filter_outliers(real_face_positions)
        
        if not real_face_positions:
            print(f"    [ANALYSIS] All samples filtered as outliers")
            return None
        
        positions_with_weights = []
        for pos in real_face_positions:
            weight = pos['det_score']
            positions_with_weights.append((pos, weight))
        
        positions_with_weights.sort(key=lambda x: x[1], reverse=True)
        top_count = max(int(len(positions_with_weights) * 0.7), MIN_FACE_SAMPLES)
        top_positions = [p[0] for p in positions_with_weights[:top_count]]
        
        median_x = int(np.median([p['center_x'] for p in top_positions]))
        median_y = int(np.median([p['center_y'] for p in top_positions]))
        median_w = int(np.median([p['w'] for p in top_positions]))
        median_h = int(np.median([p['h'] for p in top_positions]))
        
        print(f"    [ANALYSIS] Best position - X: {median_x}, Y: {median_y}, Size: {median_w}x{median_h}")
        
        crop_height = int(median_h * ZOOM_PADDING * 1.8)
        crop_width = int(crop_height / TARGET_RATIO)
        
        if crop_width > self.video_width:
            crop_width = self.video_width
            crop_height = int(crop_width * TARGET_RATIO)
        
        if crop_height > self.video_height:
            crop_height = self.video_height
            crop_width = int(crop_height / TARGET_RATIO)
        
        crop_x = median_x - crop_width // 2
        crop_y = median_y - int(crop_height * 0.35)
        
        crop_x = max(0, min(crop_x, self.video_width - crop_width))
        crop_y = max(0, min(crop_y, self.video_height - crop_height))
        
        print(f"    [ANALYSIS] Final crop - X: {crop_x}, Y: {crop_y}, W: {crop_width}, H: {crop_height}")
        
        return (crop_x, crop_y, crop_width, crop_height)
    
    def _filter_outliers(self, positions: List[Dict]) -> List[Dict]:
        if len(positions) < 4:
            return positions
        
        x_values = [p['center_x'] for p in positions]
        median_x = np.median(x_values)
        std_x = np.std(x_values)
        
        filtered = [p for p in positions if abs(p['center_x'] - median_x) <= 2 * std_x]
        
        return filtered if filtered else positions

class CropCalculator:
    
    def __init__(self, video_width: int, video_height: int):
        self.video_width = video_width
        self.video_height = video_height
        self.last_crop = None
        self.current_segment_crop = None
        self.transition_frames_remaining = 0
        self.transition_start_crop = None
        self.transition_target_crop = None
        self.in_face_tracking_transition = False
        self.tracking_progress = 0.0
        self.last_face_pos = None
        
        self.blend_frames_remaining = 0
        self.blend_start_crop = None
        self.blend_target_crop = None
    
    def set_segment_crop(self, crop: Tuple[int, int, int, int]):
        if self.last_crop is not None:
            if ENABLE_TRANSITION_ANIMATION:
                self.transition_start_crop = self.last_crop
                self.transition_target_crop = crop
                self.transition_frames_remaining = JUMP_CUT_SMOOTH_FRAMES
                self.in_face_tracking_transition = JUMP_CUT_TRACK_FACE
                self.tracking_progress = 0.0
                self.last_face_pos = None
            else:
                self.blend_start_crop = self.last_crop
                self.blend_target_crop = crop
                self.blend_frames_remaining = SMOOTH_JUMP_CUT_BLEND_FRAMES
                self.current_segment_crop = crop
        else:
            self.current_segment_crop = crop
            self.last_crop = crop
    
    def calculate(self, current_face_pos: Optional[Tuple[int, int]] = None) -> Tuple[int, int, int, int]:
        if self.transition_frames_remaining > 0:
            progress = 1.0 - (self.transition_frames_remaining / JUMP_CUT_SMOOTH_FRAMES)
            self.tracking_progress = progress
            
            if self.in_face_tracking_transition and current_face_pos is not None:
                crop = self._track_face_transition(current_face_pos, progress)
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
            
            self.last_crop = crop
            return crop
        
        if self.blend_frames_remaining > 0:
            progress = 1.0 - (self.blend_frames_remaining / SMOOTH_JUMP_CUT_BLEND_FRAMES)
            
            crop = self._micro_blend_crop(
                self.blend_start_crop,
                self.blend_target_crop,
                progress
            )
            
            self.blend_frames_remaining -= 1
            self.last_crop = crop
            return crop
        
        if self.current_segment_crop is None:
            crop = self._center_crop()
        else:
            crop = self.current_segment_crop
        
        if self.last_crop is not None and crop != self.current_segment_crop:
            crop = self._smooth_transition(self.last_crop, crop)
        
        self.last_crop = crop
        return crop
    
    def _ease_in_out(self, t: float) -> float:
        return t * t * (3.0 - 2.0 * t)
    
    def _track_face_transition(self, face_center: Tuple[int, int], progress: float) -> Tuple[int, int, int, int]:
        face_x, face_y = face_center
        
        if self.last_face_pos is not None:
            last_x, last_y = self.last_face_pos
            face_x = int(last_x + (face_x - last_x) * FACE_TRACKING_SMOOTHING)
            face_y = int(last_y + (face_y - last_y) * FACE_TRACKING_SMOOTHING)
        
        self.last_face_pos = (face_x, face_y)
        
        sx, sy, sw, sh = self.transition_start_crop
        tx, ty, tw, th = self.transition_target_crop
        
        eased_progress = self._ease_out_cubic(progress)
        
        current_w = int(sw + (tw - sw) * eased_progress)
        current_h = int(sh + (th - sh) * eased_progress)
        
        ideal_x = face_x - current_w // 2
        ideal_y = face_y - int(current_h * 0.35)
        
        target_x = int(sx + (tx - sx) * eased_progress)
        target_y = int(sy + (ty - sy) * eased_progress)
        
        face_weight = FACE_TRACKING_WEIGHT_START * (1.0 - eased_progress)
        target_weight = 1.0 - face_weight
        
        final_x = int(ideal_x * face_weight + target_x * target_weight)
        final_y = int(ideal_y * face_weight + target_y * target_weight)
        
        final_x = max(0, min(final_x, self.video_width - current_w))
        final_y = max(0, min(final_y, self.video_height - current_h))
        
        return (final_x, final_y, current_w, current_h)
    
    def _ease_out_cubic(self, t: float) -> float:
        return 1 - pow(1 - t, 3)
    
    def _micro_blend_crop(self, start_crop: Tuple[int, int, int, int],
                          end_crop: Tuple[int, int, int, int],
                          progress: float) -> Tuple[int, int, int, int]:
        eased_progress = 1 - pow(1 - progress, 4)
        
        sx, sy, sw, sh = start_crop
        ex, ey, ew, eh = end_crop
        
        x = int(sx + (ex - sx) * eased_progress)
        y = int(sy + (ey - sy) * eased_progress)
        w = int(sw + (ew - sw) * eased_progress)
        h = int(sh + (eh - sh) * eased_progress)
        
        return (x, y, w, h)
    
    def _interpolate_crop(self, start_crop: Tuple[int, int, int, int],
                         end_crop: Tuple[int, int, int, int],
                         progress: float) -> Tuple[int, int, int, int]:
        sx, sy, sw, sh = start_crop
        ex, ey, ew, eh = end_crop
        
        x = int(sx + (ex - sx) * progress)
        y = int(sy + (ey - sy) * progress)
        w = int(sw + (ew - sw) * progress)
        h = int(sh + (eh - sh) * progress)
        
        return (x, y, w, h)
    
    def _center_crop(self) -> Tuple[int, int, int, int]:
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
        ox, oy, ow, oh = old_crop
        nx, ny, nw, nh = new_crop
        
        sx = int(ox + (nx - ox) * SMOOTH_FACTOR)
        sy = int(oy + (ny - oy) * SMOOTH_FACTOR)
        sw = int(ow + (nw - ow) * SMOOTH_FACTOR)
        sh = int(oh + (nh - oh) * SMOOTH_FACTOR)
        
        return (sx, sy, sw, sh)

def analyze_video_segments(input_path: Path) -> List[Dict[str, Any]]:
    print(f"  [PHASE 1] Analyzing video segments (InsightFace SCRFD)...")
    
    info = get_video_info(input_path)
    cap = cv2.VideoCapture(str(input_path))
    
    if not cap.isOpened():
        print(f"    ERROR: Cannot open video file")
        return []
    
    detector = SCRFDFaceDetector()
    jump_detector = JumpCutDetector()
    analyzer = SegmentAnalyzer(info['width'], info['height'])
    
    print(f"    [1/2] Detecting jump cuts...")
    jump_cuts = [0]
    frame_num = 0
    prev_frame = None
    total_frames = info['frame_count']
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        if frame_num % 100 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"      Progress: {progress:.1f}% ({frame_num}/{total_frames})", end='\r')
        
        is_jump_cut = jump_detector.detect(frame, prev_frame, frame_num)
        
        if is_jump_cut:
            jump_cuts.append(frame_num)
        
        prev_frame = frame.copy()
    
    jump_cuts.append(total_frames)
    
    print(f"\n      Found {len(jump_cuts) - 1} segments (jump cuts: {len(jump_cuts) - 2})")
    
    print(f"    [2/2] Analyzing REAL HUMAN positions per segment...")
    segments = []
    
    for i in range(len(jump_cuts) - 1):
        start_frame = jump_cuts[i]
        end_frame = jump_cuts[i + 1]
        
        print(f"\n      Segment {i + 1}/{len(jump_cuts) - 1}: frames {start_frame} to {end_frame}")
        
        crop = analyzer.analyze_segment(cap, start_frame, end_frame, detector)
        
        segments.append({
            'start_frame': start_frame,
            'end_frame': end_frame,
            'crop': crop,
            'is_valid': crop is not None
        })
        
        if crop:
            print(f"      ✓ Valid REAL HUMAN detected")
        else:
            print(f"      ✗ No valid real human, will use center crop")
    
    cap.release()
    
    print(f"\n  [PHASE 1] Analysis complete - {len(segments)} segments identified")
    
    return segments

def process_video_with_segments(input_path: Path, output_path: Path,
                                segments: List[Dict[str, Any]]) -> bool:
    print(f"\n  [PHASE 2] Processing video with analyzed positions...")
    
    info = get_video_info(input_path)
    cap = cv2.VideoCapture(str(input_path))
    
    if not cap.isOpened():
        print(f"    ERROR: Cannot open video file")
        return False
    
    calculator = CropCalculator(info['width'], info['height'])
    detector = SCRFDFaceDetector()
    
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
    
    frame_num = 0
    total_frames = info['frame_count']
    current_segment_idx = 0
    prev_frame = None
    
    if segments and segments[0]['is_valid']:
        calculator.set_segment_crop(segments[0]['crop'])
        print(f"    [SEGMENT 1] Using analyzed REAL HUMAN position")
    else:
        print(f"    [SEGMENT 1] Using center crop (fallback)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num % 30 == 0 or frame_num == total_frames - 1:
            progress = (frame_num / total_frames) * 100
            seg_info = f"Segment {current_segment_idx + 1}/{len(segments)}"
            trans_info = ""
            
            if ENABLE_TRANSITION_ANIMATION:
                if calculator.transition_frames_remaining > 0:
                    trans_info = f" | Transition: {calculator.tracking_progress*100:.0f}%"
            else:
                if calculator.blend_frames_remaining > 0:
                    trans_info = f" | Blending..."
                    
            print(f"    Progress: {progress:.1f}% ({frame_num}/{total_frames}) | {seg_info}{trans_info}", end='\r')
        
        if current_segment_idx < len(segments) - 1:
            next_segment = segments[current_segment_idx + 1]
            
            if frame_num >= next_segment['start_frame']:
                current_segment_idx += 1
                
                if next_segment['is_valid']:
                    calculator.set_segment_crop(next_segment['crop'])
                    if ENABLE_TRANSITION_ANIMATION:
                        print(f"\n    [SEGMENT {current_segment_idx + 1}] Jump cut - natural face tracking transition")
                    else:
                        print(f"\n    [SEGMENT {current_segment_idx + 1}] Jump cut - instant with smooth blend")
                else:
                    calculator.set_segment_crop(calculator._center_crop())
                    if ENABLE_TRANSITION_ANIMATION:
                        print(f"\n    [SEGMENT {current_segment_idx + 1}] Jump cut - smooth transition to center crop")
                    else:
                        print(f"\n    [SEGMENT {current_segment_idx + 1}] Jump cut - instant to center crop")
        
        face_center = None
        if ENABLE_TRANSITION_ANIMATION and calculator.in_face_tracking_transition:
            faces = detector.detect(frame, prev_frame)
            real_humans = [f for f in faces if f['is_real_human']]
            
            if real_humans:
                best_face = max(real_humans, key=lambda f: f['det_score'])
                x, y, w, h = best_face['bbox']
                face_center = (x + w // 2, y + h // 2)
        
        crop_x, crop_y, crop_w, crop_h = calculator.calculate(face_center)
        
        cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        resized = cv2.resize(cropped, (PORTRAIT_WIDTH, PORTRAIT_HEIGHT))
        
        out.write(resized)
        
        prev_frame = frame.copy()
        frame_num += 1
    
    cap.release()
    out.release()
    
    print(f"\n    Progress: 100.0% - DONE")
    print(f"    Total segments processed: {len(segments)}")
    
    if not output_path.exists() or output_path.stat().st_size < 1000:
        print(f"    ERROR: Output file is invalid or too small")
        return False
    
    return True

def add_audio_to_video(video_path: Path, audio_src: Path, output_path: Path,
                      start_time_ms: float, duration_ms: float) -> bool:
    """Add audio to video using milliseconds"""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ss", format_time(start_time_ms),
        "-i", str(audio_src),
        "-t", format_time(duration_ms),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR adding audio: {e.stderr}")
        return False

def apply(paths: Dict[str, Path], metadata: Dict[str, Any]) -> Dict[str, Any]:
    print("=" * 60)
    print("VIDEO EDITING TEMPLATE - INSIGHTFACE SCRFD")
    if ENABLE_TRANSITION_ANIMATION:
        print("Mode: Natural Face Tracking + Smooth Transitions")
    else:
        print("Mode: Instant Jump Cuts + Smooth Blend (No Animation)")
    print("Time Unit: MILLISECONDS (ms)")
    print("=" * 60)
    
    video_src = paths["video_src"]
    output_dir = paths["output_dir"]
    clips_dir = paths["clips_dir"]
    
    tmp_dir = ensure_tmp_dir(output_dir)
    print(f"\n[SETUP] Temp directory: {tmp_dir}")
    
    candidates = metadata.get("candidates", [])
    if not candidates:
        return {
            "status": "error",
            "message": "No candidates found in metadata"
        }
    
    print(f"\n[INFO] Found {len(candidates)} clips to process")
    
    results = []
    
    for idx, candidate in enumerate(candidates, 1):
        print(f"\n{'=' * 60}")
        print(f"CLIP {idx}/{len(candidates)}")
        print(f"{'=' * 60}")
        
        timestart_ms = candidate.get("start")
        timestop_ms = candidate.get("stop")
        
        if timestart_ms is None or timestop_ms is None:
            print(f"  [SKIP] Missing timestart/timestop")
            continue
        
        timestart_s = timestart_ms / 1000.0
        timestop_s = timestop_ms / 1000.0
        duration_ms = timestop_ms - timestart_ms
        duration_s = duration_ms / 1000.0
        
        print(f"  Time: {timestart_ms}ms - {timestop_ms}ms ({duration_ms}ms)")
        print(f"       = {timestart_s:.2f}s - {timestop_s:.2f}s ({duration_s:.2f}s)")
        
        tmp_cut = tmp_dir / f"clip_{idx:03d}_cut.mp4"
        tmp_processed = tmp_dir / f"clip_{idx:03d}_processed.mp4"
        final_output = clips_dir / f"clip_{idx:03d}_final.mp4"
        
        print(f"\n  [STEP 1] Cutting video segment...")
        if not cut_video_segment(video_src, timestart_ms, timestop_ms, tmp_cut):
            print(f"  [ERROR] Failed to cut video")
            continue
        print(f"  [SUCCESS] Cut saved to: {tmp_cut.name}")
        
        print(f"\n  [STEP 2] Analyzing video segments with SCRFD...")
        segments = analyze_video_segments(tmp_cut)
        
        if not segments:
            print(f"  [ERROR] Failed to analyze segments")
            continue
        
        valid_segments = sum(1 for s in segments if s['is_valid'])
        print(f"  [SUCCESS] Found {len(segments)} segments ({valid_segments} with REAL humans)")
        
        print(f"\n  [STEP 3] Processing with natural face tracking...")
        if not process_video_with_segments(tmp_cut, tmp_processed, segments):
            print(f"  [ERROR] Failed to process video")
            continue
        print(f"  [SUCCESS] Processed saved to: {tmp_processed.name}")
        
        print(f"\n  [STEP 4] Adding audio...")
        if not add_audio_to_video(tmp_processed, video_src, final_output, timestart_ms, duration_ms):
            print(f"  [ERROR] Failed to add audio")
            continue
        print(f"  [SUCCESS] Final clip saved to: {final_output.name}")
        
        tmp_cut.unlink(missing_ok=True)
        tmp_processed.unlink(missing_ok=True)
        
        results.append({
            "clip_id": idx,
            "output": str(final_output),
            "timestart_ms": timestart_ms,
            "timestop_ms": timestop_ms,
            "duration_ms": duration_ms,
            "timestart_s": timestart_s,
            "timestop_s": timestop_s,
            "duration_s": duration_s,
            "segments": len(segments),
            "valid_segments": valid_segments,
            "status": "success"
        })
    
    try:
        tmp_dir.rmdir()
    except:
        pass
    
    print(f"\n{'=' * 60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total clips processed: {len(results)}/{len(candidates)}")
    print(f"Output directory: {clips_dir}")
    
    return {
        "status": "success",
        "total_clips": len(candidates),
        "processed_clips": len(results),
        "clips": results,
        "output_dir": str(clips_dir)
    }