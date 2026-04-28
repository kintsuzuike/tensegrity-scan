"""
scoring.py
TBI（Tensegrity Balance Index）算出ロジック
TBI = 0.35S + 0.30F + 0.25L + 0.10R
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List


def compute_structure_score(df: pd.DataFrame, stats: Dict) -> Tuple[float, Dict]:
    """
    S = Structure Score（0〜100）
    静止立位における構造スコア。
    肩ライン、骨盤ライン、頭部位置、膝軸、足首軸、垂直関係から算出。
    左右対称性ではなく、張力が通る配置かどうかを評価。
    """
    if df.empty:
        return 50.0, {}
    
    detail = {}
    scores = []
    
    # 1. 垂直チェーン偏差（重要）
    if "vertical_chain_deviation" in df.columns:
        mean_dev = df["vertical_chain_deviation"].mean()
        # 偏差0=100点、偏差0.5=50点、偏差1.0=0点
        s = max(0, 100 - mean_dev * 100)
        scores.append(("vertical_chain", s, 0.40))
        detail["vertical_chain_score"] = s
    
    # 2. 肩ライン角度（固定化ではなく平均角度の評価）
    if "shoulder_angle" in stats:
        mean_abs = abs(stats["shoulder_angle"]["mean"])
        # 0度=100点、10度=0点
        s = max(0, 100 - mean_abs * 10)
        scores.append(("shoulder_angle", s, 0.20))
        detail["shoulder_angle_score"] = s
    
    # 3. 骨盤ライン角度
    if "pelvis_angle" in stats:
        mean_abs = abs(stats["pelvis_angle"]["mean"])
        s = max(0, 100 - mean_abs * 10)
        scores.append(("pelvis_angle", s, 0.20))
        detail["pelvis_angle_score"] = s
    
    # 4. 頭部位置（肩中心との相対的なズレ）
    if "head_offset_x" in df.columns and "shoulder_width" in df.columns:
        mean_head_offset = abs(df["head_offset_x"].mean())
        s = max(0, 100 - mean_head_offset * 80)
        scores.append(("head_position", s, 0.20))
        detail["head_position_score"] = s
    
    if not scores:
        return 50.0, detail
    
    # 重み付き平均
    total_weight = sum(w for _, _, w in scores)
    weighted_sum = sum(s * w for _, s, w in scores)
    structure_score = weighted_sum / total_weight if total_weight > 0 else 50.0
    
    return float(np.clip(structure_score, 0, 100)), detail


def compute_fluctuation_score(df: pd.DataFrame, stats: Dict) -> Tuple[float, Dict]:
    """
    F = Fluctuation Score（0〜100）
    揺らぎスコア。
    微細に揺れて中心へ戻る場合は高評価。
    同じ方向に固定されたまま変化が少ない場合は低評価。
    """
    if df.empty or len(df) < 3:
        return 50.0, {}
    
    detail = {}
    scores = []
    
    # 評価対象の指標と「健全な揺らぎの標準偏差範囲」
    fluctuation_targets = [
        ("shoulder_angle", 0.5, 4.0, 0.25),   # (列名, 理想最小std, 理想最大std, 重み)
        ("pelvis_angle", 0.3, 3.5, 0.25),
        ("head_center_x", 0.003, 0.025, 0.20),
        ("knee_center_x", 0.003, 0.025, 0.15),
        ("ankle_center_x", 0.002, 0.020, 0.15),
    ]
    
    for col, min_std, max_std, weight in fluctuation_targets:
        if col in stats and "std" in stats[col]:
            std = stats[col]["std"]
            
            if std < min_std:
                # 揺らぎが小さすぎる（固定化）→ 低スコア
                s = (std / min_std) * 50
            elif std > max_std:
                # 揺らぎが大きすぎる（不安定）→ 低スコア
                excess = (std - max_std) / max_std
                s = max(20, 80 - excess * 40)
            else:
                # 健全な揺らぎ範囲 → 高スコア
                s = 75 + 25 * ((std - min_std) / (max_std - min_std))
            
            scores.append((col, s, weight))
            detail[f"{col}_fluctuation_score"] = s
    
    if not scores:
        return 50.0, detail
    
    total_weight = sum(w for _, _, w in scores)
    weighted_sum = sum(s * w for _, s, w in scores)
    fluctuation_score = weighted_sum / total_weight if total_weight > 0 else 50.0
    
    return float(np.clip(fluctuation_score, 0, 100)), detail


def compute_load_path_score(df: pd.DataFrame, stats: Dict) -> Tuple[float, Dict]:
    """
    L = Load Path Score（0〜100）
    張力通路スコア。
    足部→膝→骨盤→胸郭→頭部のラインが大きく遮断されていないかを評価。
    """
    if df.empty:
        return 50.0, {}
    
    detail = {}
    scores = []
    
    # 各セグメントのX方向ズレを評価
    # セグメント: 足首→膝、膝→骨盤、骨盤→肩、肩→頭
    segment_pairs = [
        ("ankle_center_x", "knee_center_x", "足首-膝", 0.30),
        ("knee_center_x", "pelvis_center_x", "膝-骨盤", 0.30),
        ("pelvis_center_x", "shoulder_center_x", "骨盤-肩", 0.25),
        ("shoulder_center_x", "head_center_x", "肩-頭", 0.15),
    ]
    
    shoulder_w = df["shoulder_width"].mean() if "shoulder_width" in df.columns else 0.2
    
    for col_lower, col_upper, label, weight in segment_pairs:
        if col_lower in df.columns and col_upper in df.columns:
            diff = (df[col_upper] - df[col_lower]).abs().mean()
            norm_diff = diff / shoulder_w if shoulder_w > 0 else diff
            
            # 正規化偏差0=100点、0.5=0点
            s = max(0, 100 - norm_diff * 200)
            scores.append((label, s, weight))
            detail[f"segment_{label}_score"] = s
    
    if not scores:
        return 50.0, detail
    
    total_weight = sum(w for _, _, w in scores)
    weighted_sum = sum(s * w for _, s, w in scores)
    load_path_score = weighted_sum / total_weight if total_weight > 0 else 50.0
    
    return float(np.clip(load_path_score, 0, 100)), detail


def compute_recovery_score(df: pd.DataFrame, stats: Dict) -> Tuple[float, Dict]:
    """
    R = Recovery Score（0〜100）
    復元力スコア。
    ズレた構造が時系列の中で中心へ戻る傾向があるかを評価。
    """
    if df.empty or len(df) < 3:
        return 50.0, {}
    
    detail = {}
    scores = []
    
    # 評価対象列
    recovery_targets = [
        ("head_center_x", 0.25),
        ("shoulder_center_x", 0.25),
        ("pelvis_center_x", 0.25),
        ("vertical_chain_deviation", 0.25),
    ]
    
    for col, weight in recovery_targets:
        if col not in df.columns:
            continue
        
        series = df[col].dropna().values
        if len(series) < 3:
            continue
        
        mean_val = np.mean(series)
        deviations = series - mean_val
        
        # 復元傾向の評価：
        # 偏差の符号変化回数が多い → 中心への復帰が多い → 高スコア
        sign_changes = np.sum(np.diff(np.sign(deviations)) != 0)
        max_possible_changes = len(deviations) - 1
        
        if max_possible_changes > 0:
            change_ratio = sign_changes / max_possible_changes
        else:
            change_ratio = 0.5
        
        # 変化なし（固定化）→ 50点、頻繁な復帰 → 90点
        s = 50 + change_ratio * 40
        
        # さらに、最終フレームが平均に近い場合は加点
        if len(series) > 0:
            std = np.std(series)
            last_dev = abs(series[-1] - mean_val)
            if std > 0:
                closeness = max(0, 1 - last_dev / (std + 0.001))
                s = s * 0.8 + closeness * 20
        
        scores.append((col, s, weight))
        detail[f"{col}_recovery_score"] = s
    
    if not scores:
        return 50.0, detail
    
    total_weight = sum(w for _, _, w in scores)
    weighted_sum = sum(s * w for _, s, w in scores)
    recovery_score = weighted_sum / total_weight if total_weight > 0 else 50.0
    
    return float(np.clip(recovery_score, 0, 100)), detail


def compute_tbi(
    structure_score: float,
    fluctuation_score: float,
    load_path_score: float,
    recovery_score: float
) -> float:
    """
    TBI = 0.35S + 0.30F + 0.25L + 0.10R
    """
    tbi = (
        0.35 * structure_score +
        0.30 * fluctuation_score +
        0.25 * load_path_score +
        0.10 * recovery_score
    )
    return float(np.clip(tbi, 0, 100))


def get_tbi_label(tbi: float) -> Dict:
    """
    TBIスコアから判定ラベルと色を返す
    """
    if tbi >= 85:
        return {
            "label": "High Flow",
            "emoji": "🟢",
            "color": "#00d4a1",
            "description": "非常に良好なテンセグリティバランスです。身体の張力が自然に循環しています。",
            "bg_color": "#0d2b24"
        }
    elif tbi >= 70:
        return {
            "label": "Good Flow",
            "emoji": "🟡",
            "color": "#7ecfff",
            "description": "良好なバランスです。一部に改善の余地はありますが、全体的な流れは良好です。",
            "bg_color": "#0d1f2d"
        }
    elif tbi >= 55:
        return {
            "label": "Caution",
            "emoji": "🟠",
            "color": "#ffd166",
            "description": "注意が必要な部位があります。固定化傾向のある箇所を意識してみましょう。",
            "bg_color": "#2b2109"
        }
    elif tbi >= 40:
        return {
            "label": "Risk",
            "emoji": "🔴",
            "color": "#ff6b6b",
            "description": "リスク状態です。複数の部位で固定化や張力の遮断が見られます。",
            "bg_color": "#2b0d0d"
        }
    else:
        return {
            "label": "Collapse",
            "emoji": "⚫",
            "color": "#9b59b6",
            "description": "テンセグリティ構造の崩れが顕著です。専門家への相談を検討してください。",
            "bg_color": "#1a0d2b"
        }


def generate_comments(df: pd.DataFrame, stats: Dict, risks: Dict) -> List[str]:
    """
    スコアとリスクに基づいてコメントリストを生成する
    """
    comments = []
    
    # リスクからのコメント
    for risk_key, risk_info in risks.items():
        comments.append(risk_info["comment"])
    
    # 揺らぎ関連コメント
    if "shoulder_angle" in stats:
        std = stats["shoulder_angle"].get("std", 0)
        if std > 3.0:
            comments.append("肩ラインに大きな揺らぎがありますが、左右差がある場合も自然な揺らぎの範囲です")
        elif std < 0.5:
            comments.append("肩ラインの揺らぎが非常に小さく、固定化傾向があります")
    
    if "pelvis_angle" in stats:
        std = stats["pelvis_angle"].get("std", 0)
        mean = abs(stats["pelvis_angle"].get("mean", 0))
        if std < 0.4 and mean > 2.0:
            comments.append("骨盤の揺らぎが小さいです")
        elif std > 2.0 and mean < 3.0:
            comments.append("左右差はありますが、時系列変動があるため自然な揺らぎの範囲です")
    
    # デフォルトコメント
    if not comments:
        if len(df) < 3:
            comments.append("フレーム数が少ないため、詳細な分析には長い動画が必要です")
        else:
            comments.append("全体的なバランスは安定しています")
    
    return comments[:5]  # 最大5件


def run_full_analysis(df: pd.DataFrame, stats: Dict, risks: Dict) -> Dict:
    """
    全スコアを一括算出してresult dictを返す
    """
    S, s_detail = compute_structure_score(df, stats)
    F, f_detail = compute_fluctuation_score(df, stats)
    L, l_detail = compute_load_path_score(df, stats)
    R, r_detail = compute_recovery_score(df, stats)
    
    tbi = compute_tbi(S, F, L, R)
    tbi_info = get_tbi_label(tbi)
    comments = generate_comments(df, stats, risks)
    
    return {
        "tbi": tbi,
        "tbi_info": tbi_info,
        "structure_score": S,
        "fluctuation_score": F,
        "load_path_score": L,
        "recovery_score": R,
        "structure_detail": s_detail,
        "fluctuation_detail": f_detail,
        "load_path_detail": l_detail,
        "recovery_detail": r_detail,
        "comments": comments,
        "risks": risks,
        "frame_count": len(df),
    }
