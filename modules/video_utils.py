"""
video_utils.py
動画の読み込みと1秒ごとのフレーム抽出を担当するモジュール
"""

import cv2
import numpy as np
from typing import List, Tuple
import tempfile
import os


def save_uploaded_video(uploaded_file) -> str:
    """
    StreamlitのUploadedFileを一時ファイルに保存し、パスを返す
    """
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def extract_frames_per_second(video_path: str) -> Tuple[List[np.ndarray], float, int]:
    """
    動画から1秒ごとにフレームを抽出する
    
    Returns:
        frames: 抽出されたフレームのリスト (BGR numpy array)
        fps: 動画のFPS
        total_frames: 総フレーム数
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"動画ファイルを開けませんでした: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0:
        fps = 30.0  # デフォルトFPS
    
    frames = []
    timestamps = []
    frame_indices = []
    
    # 1秒ごとのフレームインデックスを計算
    interval = int(fps)
    current_second = 0
    
    while True:
        target_frame = current_second * interval
        if target_frame >= total_frames:
            break
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frames.append(frame)
        timestamps.append(current_second)
        frame_indices.append(target_frame)
        current_second += 1
    
    cap.release()
    
    return frames, fps, total_frames, timestamps


def get_video_info(video_path: str) -> dict:
    """
    動画の基本情報を取得する
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration_seconds": duration
    }


def frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    """
    BGRフレームをRGBに変換する（Streamlit表示用）
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def resize_frame_for_display(frame: np.ndarray, max_width: int = 640) -> np.ndarray:
    """
    表示用にフレームをリサイズする
    """
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
    return frame
