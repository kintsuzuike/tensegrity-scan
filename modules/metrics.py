"""
metrics.py
各フレームから身体バランス指標を算出するモジュール
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import math
from modules.pose_utils import calculate_angle_degrees, midpoint, distance


def extract_frame_metrics(landmarks: Dict) -> Optional[Dict]:
    """
    1フレームのランドマークから各種バランス指標を算出する
    """
    if landmarks is None:
        return None
    
    try:
        # --- 各ポイント取得 ---
        nose = landmarks["nose"][:2]
        left_eye = landmarks["left_eye"][:2]
        right_eye = landmarks["right_eye"][:2]
        left_ear = landmarks["left_ear"][:2]
        right_ear = landmarks["right_ear"][:2]
        left_shoulder = landmarks["left_shoulder"][:2]
        right_shoulder = landmarks["right_shoulder"][:2]
        left_hip = landmarks["left_hip"][:2]
        right_hip = landmarks["right_hip"][:2]
        left_knee = landmarks["left_knee"][:2]
        right_knee = landmarks["right_knee"][:2]
        left_ankle = landmarks["left_ankle"][:2]
        right_ankle = landmarks["right_ankle"][:2]
        
        # --- 頭部中心X座標 ---
        head_center_x = (left_ear[0] + right_ear[0]) / 2 if (
            landmarks["left_ear"][2] > 0.3 and landmarks["right_ear"][2] > 0.3
        ) else nose[0]
        
        # --- 肩ライン角度 ---
        shoulder_angle = calculate_angle_degrees(left_shoulder, right_shoulder)
        
        # --- 骨盤ライン角度 ---
        pelvis_angle = calculate_angle_degrees(left_hip, right_hip)
        
        # --- 肩中心 ---
        shoulder_center = midpoint(left_shoulder, right_shoulder)
        
        # --- 骨盤中心 ---
        pelvis_center = midpoint(left_hip, right_hip)
        
        # --- 膝中心 ---
        knee_center = midpoint(left_knee, right_knee)
        knee_center_x = knee_center[0]
        
        # --- 足首中心 ---
        ankle_center = midpoint(left_ankle, right_ankle)
        ankle_center_x = ankle_center[0]
        
        # --- 肩幅（正規化） ---
        shoulder_width = distance(left_shoulder, right_shoulder)
        
        # --- 骨盤幅（正規化） ---
        pelvis_width = distance(left_hip, right_hip)
        
        # --- 頭部・肩・骨盤・膝・足首の垂直関係（X座標のズレ） ---
        # 足首中心を基準（0）として各部位のX位置を正規化
        ref_x = ankle_center_x
        norm_factor = shoulder_width if shoulder_width > 0.01 else 0.2
        
        head_offset_x = (head_center_x - ref_x) / norm_factor
        shoulder_offset_x = (shoulder_center[0] - ref_x) / norm_factor
        pelvis_offset_x = (pelvis_center[0] - ref_x) / norm_factor
        knee_offset_x = (knee_center_x - ref_x) / norm_factor
        
        # --- 左右差（膝・足首） ---
        knee_lr_diff = left_knee[0] - right_knee[0]
        ankle_lr_diff = left_ankle[0] - right_ankle[0]
        
        # --- 左右の肩高さ差 ---
        shoulder_height_diff = left_shoulder[1] - right_shoulder[1]  # 正規化済み
        
        # --- 左右の骨盤高さ差 ---
        pelvis_height_diff = left_hip[1] - right_hip[1]
        
        # --- 垂直アライメントスコア（各部位の連鎖のズレ） ---
        # 足首→膝→骨盤→肩→頭部のX座標偏差合計
        vertical_chain_deviation = (
            abs(head_offset_x) + 
            abs(shoulder_offset_x) + 
            abs(pelvis_offset_x) + 
            abs(knee_offset_x)
        ) / 4.0
        
        return {
            "head_center_x": head_center_x,
            "head_offset_x": head_offset_x,
            "shoulder_angle": shoulder_angle,
            "shoulder_angle_abs": abs(shoulder_angle),
            "pelvis_angle": pelvis_angle,
            "pelvis_angle_abs": abs(pelvis_angle),
            "shoulder_center_x": shoulder_center[0],
            "shoulder_offset_x": shoulder_offset_x,
            "pelvis_center_x": pelvis_center[0],
            "pelvis_offset_x": pelvis_offset_x,
            "knee_center_x": knee_center_x,
            "knee_offset_x": knee_offset_x,
            "ankle_center_x": ankle_center_x,
            "knee_lr_diff": knee_lr_diff,
            "ankle_lr_diff": ankle_lr_diff,
            "shoulder_width": shoulder_width,
            "pelvis_width": pelvis_width,
            "shoulder_height_diff": shoulder_height_diff,
            "pelvis_height_diff": pelvis_height_diff,
            "vertical_chain_deviation": vertical_chain_deviation,
        }
    
    except (KeyError, IndexError, ZeroDivisionError):
        return None


def build_metrics_dataframe(
    all_landmarks: List[Optional[Dict]], 
    timestamps: List[float]
) -> pd.DataFrame:
    """
    全フレームの指標をDataFrameとして構築する
    """
    rows = []
    for t, lm in zip(timestamps, all_landmarks):
        metrics = extract_frame_metrics(lm)
        if metrics is not None:
            metrics["timestamp"] = t
            rows.append(metrics)
    
    if not rows:
        return pd.DataFrame()
    
    return pd.DataFrame(rows).set_index("timestamp")


def compute_fluctuation_stats(df: pd.DataFrame) -> Dict:
    """
    時系列データから揺らぎ統計を算出する
    """
    if df.empty or len(df) < 2:
        return {}
    
    stats = {}
    
    key_cols = [
        "shoulder_angle", "pelvis_angle", 
        "head_center_x", "shoulder_center_x",
        "pelvis_center_x", "knee_center_x", "ankle_center_x",
        "knee_lr_diff", "ankle_lr_diff",
        "vertical_chain_deviation"
    ]
    
    for col in key_cols:
        if col in df.columns:
            series = df[col].dropna()
            if len(series) > 0:
                stats[col] = {
                    "mean": float(series.mean()),
                    "std": float(series.std()) if len(series) > 1 else 0.0,
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "range": float(series.max() - series.min()),
                    # 固定化指数：標準偏差が小さいほど固定化している
                    "fixation_index": float(1.0 / (series.std() + 0.001)) if len(series) > 1 else 99.0,
                }
    
    return stats


def detect_fixed_deviations(df: pd.DataFrame, stats: Dict) -> Dict:
    """
    固定化された偏位を検出する
    左右差そのものではなく、時間経過で変化しない固定偏位をリスクと判定
    """
    risks = {}
    
    if df.empty:
        return risks
    
    # 肩ライン固定化チェック
    if "shoulder_angle" in stats:
        s = stats["shoulder_angle"]
        mean_angle = abs(s["mean"])
        std_angle = s["std"]
        if mean_angle > 3.0 and std_angle < 1.5:
            severity = "high" if mean_angle > 6.0 else "medium"
            risks["shoulder_fixation"] = {
                "label": "肩ライン固定化",
                "severity": severity,
                "detail": f"平均角度 {s['mean']:.1f}°、変動幅 {s['std']:.2f}°（小さい＝固定化）",
                "comment": "肩ラインの固定化傾向があります"
            }
    
    # 骨盤ライン固定化チェック
    if "pelvis_angle" in stats:
        s = stats["pelvis_angle"]
        mean_angle = abs(s["mean"])
        std_angle = s["std"]
        if mean_angle > 2.5 and std_angle < 1.2:
            severity = "high" if mean_angle > 5.0 else "medium"
            risks["pelvis_fixation"] = {
                "label": "骨盤ライン固定化",
                "severity": severity,
                "detail": f"平均角度 {s['mean']:.1f}°、変動幅 {s['std']:.2f}°",
                "comment": "骨盤の揺らぎが小さいです"
            }
    
    # 頭部偏位チェック
    if "head_center_x" in stats:
        s = stats["head_center_x"]
        # 肩中心との比較
        if "shoulder_center_x" in stats:
            head_shoulder_diff = abs(s["mean"] - stats["shoulder_center_x"]["mean"])
            shoulder_w = df["shoulder_width"].mean() if "shoulder_width" in df.columns else 0.2
            norm_diff = head_shoulder_diff / shoulder_w if shoulder_w > 0 else 0
            if norm_diff > 0.15 and s["std"] < 0.01:
                risks["head_lateral_shift"] = {
                    "label": "頭部側方偏位",
                    "severity": "medium",
                    "detail": f"肩中心との偏差 {norm_diff:.2f}（正規化値）",
                    "comment": "頭部が肩の中心からズレた位置に固定されています"
                }
    
    # 足部での張力停止チェック
    if "vertical_chain_deviation" in stats:
        s = stats["vertical_chain_deviation"]
        if s["mean"] > 0.3:
            # どの部位でズレが大きいか特定
            worst_part = _find_worst_chain_part(df)
            risks["load_path_break"] = {
                "label": f"張力通路の遮断（{worst_part}付近）",
                "severity": "high" if s["mean"] > 0.5 else "medium",
                "detail": f"垂直チェーン偏差平均 {s['mean']:.2f}",
                "comment": f"{worst_part}で張力が止まっている可能性があります"
            }
    
    # 膝・足首の固定化左右差チェック
    for key, label in [("knee_lr_diff", "膝"), ("ankle_lr_diff", "足首")]:
        if key in stats:
            s = stats[key]
            if abs(s["mean"]) > 0.05 and s["std"] < 0.015:
                side = "右" if s["mean"] > 0 else "左"
                risks[f"{key}_fixation"] = {
                    "label": f"{label}軸の固定偏位",
                    "severity": "medium",
                    "detail": f"平均偏差 {s['mean']:.3f}、変動幅 {s['std']:.4f}",
                    "comment": f"{side}足部で張力が止まっている可能性があります"
                }
    
    return risks


def _find_worst_chain_part(df: pd.DataFrame) -> str:
    """
    垂直チェーンの中で最もズレが大きい部位を特定する
    """
    offsets = {}
    for col, label in [
        ("head_offset_x", "頭部"),
        ("shoulder_offset_x", "肩"),
        ("pelvis_offset_x", "骨盤"),
        ("knee_offset_x", "膝"),
    ]:
        if col in df.columns:
            offsets[label] = abs(df[col].mean())
    
    if not offsets:
        return "骨盤"
    
    return max(offsets, key=offsets.get)
