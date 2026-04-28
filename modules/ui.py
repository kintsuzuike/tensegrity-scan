"""
ui.py
Streamlit UIコンポーネントのモジュール
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import math


# ===== カラーパレット =====
COLORS = {
    "bg_primary": "#0a0e1a",
    "bg_secondary": "#0f1629",
    "bg_card": "#141c33",
    "accent_cyan": "#00d4a1",
    "accent_blue": "#7ecfff",
    "accent_orange": "#ffd166",
    "accent_red": "#ff6b6b",
    "accent_purple": "#c77dff",
    "text_primary": "#e8f4fd",
    "text_secondary": "#8ba4c0",
    "border": "#1e2d4a",
    "grid": "#1a2540",
}


def apply_global_styles():
    """グローバルCSSを適用する"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');
    
    :root {
        --bg-primary: #0a0e1a;
        --bg-card: #141c33;
        --accent-cyan: #00d4a1;
        --accent-blue: #7ecfff;
        --text-primary: #e8f4fd;
        --text-secondary: #8ba4c0;
        --border: #1e2d4a;
    }
    
    .stApp {
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }
    
    .main .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }
    
    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        color: var(--text-primary) !important;
    }
    
    p, div, span, label {
        color: var(--text-primary) !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00d4a1, #7ecfff) !important;
        color: #0a0e1a !important;
        font-family: 'Space Mono', monospace !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.6rem 1.8rem !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.05em !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(0, 212, 161, 0.4) !important;
    }
    
    .stFileUploader {
        background: var(--bg-card) !important;
        border: 1px dashed var(--border) !important;
        border-radius: 8px !important;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #00d4a1, #7ecfff) !important;
    }
    
    [data-testid="stMetricValue"] {
        font-family: 'Space Mono', monospace !important;
        color: var(--accent-cyan) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Space Mono', monospace !important;
        color: var(--text-secondary) !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--accent-cyan) !important;
    }
    
    hr {
        border-color: var(--border) !important;
    }
    
    .element-container {
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """ヘッダーを描画する"""
    st.markdown("""
    <div style="
        text-align: center; 
        padding: 2rem 0 1.5rem;
        border-bottom: 1px solid #1e2d4a;
        margin-bottom: 2rem;
    ">
        <div style="
            font-family: 'Space Mono', monospace;
            font-size: 0.75rem;
            letter-spacing: 0.25em;
            color: #00d4a1;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
        ">PROTOTYPE · NON-MEDICAL</div>
        <h1 style="
            font-family: 'Syne', sans-serif;
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #00d4a1, #7ecfff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 0 0 0.5rem;
            letter-spacing: -0.02em;
        ">Tensegrity Scan™</h1>
        <p style="
            font-family: 'Space Mono', monospace;
            font-size: 0.85rem;
            color: #8ba4c0;
            margin: 0;
        ">身体バランス可視化プロトタイプ · TBI解析システム</p>
    </div>
    """, unsafe_allow_html=True)


def render_upload_section(label: str = "立位動画をアップロード", key: str = "video") -> object:
    """動画アップロードUIを描画する"""
    st.markdown(f"""
    <div style="
        background: #141c33;
        border: 1px solid #1e2d4a;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    ">
        <div style="
            font-family: 'Syne', sans-serif;
            font-size: 1rem;
            font-weight: 700;
            color: #e8f4fd;
            margin-bottom: 0.5rem;
        ">{label}</div>
        <div style="
            font-family: 'Space Mono', monospace;
            font-size: 0.72rem;
            color: #8ba4c0;
            margin-bottom: 1rem;
        ">対応形式: MP4 · MOV · AVI · 推奨: 5〜30秒の立位動画</div>
    </div>
    """, unsafe_allow_html=True)
    
    return st.file_uploader(
        label,
        type=["mp4", "mov", "avi", "m4v"],
        key=key,
        label_visibility="collapsed"
    )


def render_analysis_progress(current: int, total: int, message: str = ""):
    """解析進捗を描画する"""
    progress = current / total if total > 0 else 0
    
    st.markdown(f"""
    <div style="
        background: #141c33;
        border: 1px solid #1e2d4a;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    ">
        <div style="
            font-family: 'Space Mono', monospace;
            font-size: 0.8rem;
            color: #00d4a1;
            margin-bottom: 0.8rem;
            letter-spacing: 0.1em;
        ">ANALYZING · フレーム {current}/{total}</div>
        <div style="
            font-family: 'Space Mono', monospace;
            font-size: 0.75rem;
            color: #8ba4c0;
        ">{message}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.progress(progress)


def render_tbi_gauge(tbi: float, tbi_info: Dict):
    """TBIゲージを描画する"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=tbi,
        number={
            'font': {'size': 56, 'family': 'Space Mono', 'color': tbi_info["color"]},
            'suffix': ''
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': "#1e2d4a",
                'tickfont': {'color': '#8ba4c0', 'size': 10, 'family': 'Space Mono'}
            },
            'bar': {'color': tbi_info["color"], 'thickness': 0.3},
            'bgcolor': "#141c33",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 39], 'color': '#1a0d2b'},
                {'range': [39, 54], 'color': '#2b0d0d'},
                {'range': [54, 69], 'color': '#2b2109'},
                {'range': [69, 84], 'color': '#0d1f2d'},
                {'range': [84, 100], 'color': '#0d2b24'},
            ],
            'threshold': {
                'line': {'color': tbi_info["color"], 'width': 3},
                'thickness': 0.8,
                'value': tbi
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e8f4fd'},
        margin=dict(t=20, b=0, l=30, r=30),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_score_cards(result: Dict):
    """4つのサブスコアカードを描画する"""
    scores = [
        ("S", "Structure", result["structure_score"], "構造スコア", "#00d4a1"),
        ("F", "Fluctuation", result["fluctuation_score"], "揺らぎスコア", "#7ecfff"),
        ("L", "Load Path", result["load_path_score"], "張力通路スコア", "#ffd166"),
        ("R", "Recovery", result["recovery_score"], "復元力スコア", "#c77dff"),
    ]
    
    weights = [0.35, 0.30, 0.25, 0.10]
    
    cols = st.columns(4)
    for col, (abbr, name, score, desc, color), weight in zip(cols, scores, weights):
        with col:
            bar_width = int(score)
            st.markdown(f"""
            <div style="
                background: #141c33;
                border: 1px solid #1e2d4a;
                border-radius: 10px;
                padding: 1.2rem 1rem;
                text-align: center;
                position: relative;
                overflow: hidden;
            ">
                <div style="
                    position: absolute;
                    bottom: 0; left: 0;
                    width: {bar_width}%;
                    height: 3px;
                    background: {color};
                    opacity: 0.8;
                "></div>
                <div style="
                    font-family: 'Space Mono', monospace;
                    font-size: 0.65rem;
                    color: {color};
                    letter-spacing: 0.15em;
                    margin-bottom: 0.3rem;
                ">×{weight}</div>
                <div style="
                    font-family: 'Syne', sans-serif;
                    font-size: 2rem;
                    font-weight: 800;
                    color: {color};
                    line-height: 1;
                ">{score:.1f}</div>
                <div style="
                    font-family: 'Space Mono', monospace;
                    font-size: 0.7rem;
                    font-weight: 700;
                    color: #e8f4fd;
                    margin-top: 0.3rem;
                ">{abbr} · {name}</div>
                <div style="
                    font-family: 'Space Mono', monospace;
                    font-size: 0.62rem;
                    color: #8ba4c0;
                    margin-top: 0.15rem;
                ">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


def render_risk_list(risks: Dict):
    """リスク部位リストを描画する"""
    if not risks:
        st.markdown("""
        <div style="
            background: #141c33;
            border: 1px solid #1e2d4a;
            border-left: 3px solid #00d4a1;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            font-family: 'Space Mono', monospace;
            font-size: 0.8rem;
            color: #00d4a1;
        ">✓ 検出されたリスク部位はありません</div>
        """, unsafe_allow_html=True)
        return
    
    severity_colors = {
        "high": "#ff6b6b",
        "medium": "#ffd166",
        "low": "#7ecfff"
    }
    severity_labels = {"high": "HIGH", "medium": "MED", "low": "LOW"}
    
    for key, risk in risks.items():
        color = severity_colors.get(risk["severity"], "#7ecfff")
        sev_label = severity_labels.get(risk["severity"], "MED")
        st.markdown(f"""
        <div style="
            background: #141c33;
            border: 1px solid #1e2d4a;
            border-left: 3px solid {color};
            border-radius: 8px;
            padding: 0.9rem 1.2rem;
            margin-bottom: 0.5rem;
        ">
            <div style="display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.3rem;">
                <span style="
                    background: {color}22;
                    color: {color};
                    font-family: 'Space Mono', monospace;
                    font-size: 0.6rem;
                    padding: 0.15rem 0.4rem;
                    border-radius: 2px;
                    letter-spacing: 0.1em;
                ">{sev_label}</span>
                <span style="
                    font-family: 'Syne', sans-serif;
                    font-weight: 700;
                    font-size: 0.9rem;
                    color: #e8f4fd;
                ">{risk['label']}</span>
            </div>
            <div style="
                font-family: 'Space Mono', monospace;
                font-size: 0.72rem;
                color: #8ba4c0;
            ">{risk['detail']}</div>
        </div>
        """, unsafe_allow_html=True)


def render_comments(comments: List[str]):
    """コメントリストを描画する"""
    for comment in comments:
        st.markdown(f"""
        <div style="
            background: #141c33;
            border: 1px solid #1e2d4a;
            border-radius: 8px;
            padding: 0.8rem 1.2rem;
            margin-bottom: 0.4rem;
            font-family: 'Space Mono', monospace;
            font-size: 0.78rem;
            color: #e8f4fd;
            display: flex;
            align-items: flex-start;
            gap: 0.6rem;
        ">
            <span style="color: #7ecfff; flex-shrink: 0;">›</span>
            <span>{comment}</span>
        </div>
        """, unsafe_allow_html=True)


def render_landmark_figure(landmarks_list: List, frame_idx: int = 0):
    """
    人体ランドマーク図をPlotlyで描画する
    """
    if not landmarks_list or frame_idx >= len(landmarks_list):
        return
    
    lm = landmarks_list[frame_idx]
    if lm is None:
        st.info("このフレームはランドマーク検出に失敗しました")
        return
    
    # 表示するランドマークと接続
    points = {
        "nose": lm["nose"][:2],
        "left_shoulder": lm["left_shoulder"][:2],
        "right_shoulder": lm["right_shoulder"][:2],
        "left_elbow": lm["left_elbow"][:2],
        "right_elbow": lm["right_elbow"][:2],
        "left_wrist": lm["left_wrist"][:2],
        "right_wrist": lm["right_wrist"][:2],
        "left_hip": lm["left_hip"][:2],
        "right_hip": lm["right_hip"][:2],
        "left_knee": lm["left_knee"][:2],
        "right_knee": lm["right_knee"][:2],
        "left_ankle": lm["left_ankle"][:2],
        "right_ankle": lm["right_ankle"][:2],
    }
    
    connections = [
        ("nose", "left_shoulder"), ("nose", "right_shoulder"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ]
    
    fig = go.Figure()
    
    # 接続線を描画
    for p1_name, p2_name in connections:
        if p1_name in points and p2_name in points:
            p1, p2 = points[p1_name], points[p2_name]
            fig.add_trace(go.Scatter(
                x=[p1[0], p2[0]],
                y=[-p1[1], -p2[1]],  # Y軸反転（画像座標系）
                mode='lines',
                line=dict(color='#00d4a1', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # 各ランドマーク点を描画
    for name, (x, y) in points.items():
        is_key = name in ["nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        fig.add_trace(go.Scatter(
            x=[x],
            y=[-y],
            mode='markers+text',
            marker=dict(
                size=10 if is_key else 7,
                color='#7ecfff' if is_key else '#ffffff',
                line=dict(color='#0a0e1a', width=1.5)
            ),
            text=[name.replace("_", "<br>")],
            textposition="top center",
            textfont=dict(size=7, color='#8ba4c0', family='Space Mono'),
            name=name,
            showlegend=False,
            hovertemplate=f"{name}<br>x: {x:.3f}, y: {y:.3f}<extra></extra>"
        ))
    
    # 垂直線（中心軸）
    all_x = [p[0] for p in points.values()]
    center_x = np.mean(all_x)
    all_y = [-p[1] for p in points.values()]
    
    fig.add_shape(
        type="line",
        x0=center_x, y0=min(all_y) - 0.05,
        x1=center_x, y1=max(all_y) + 0.05,
        line=dict(color='#1e2d4a', width=1, dash='dot')
    )
    
    fig.update_layout(
        height=420,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0f1629',
        xaxis=dict(
            showgrid=False, zeroline=False,
            showticklabels=False, range=[0, 1]
        ),
        yaxis=dict(
            showgrid=False, zeroline=False,
            showticklabels=False,
            scaleanchor='x', scaleratio=1
        ),
        margin=dict(t=10, b=10, l=10, r=10),
        title=dict(
            text=f"Frame {frame_idx} · ランドマーク図",
            font=dict(family='Space Mono', size=11, color='#8ba4c0'),
            x=0.5
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_fluctuation_charts(df: pd.DataFrame):
    """
    揺らぎ解析の時系列グラフを描画する
    """
    if df.empty:
        st.info("データがありません")
        return
    
    timestamps = df.index.tolist()
    
    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=[
            "肩ライン角度 (°)", 
            "骨盤ライン角度 (°)",
            "頭部中心X座標",
            "膝中心の左右差",
            "足首中心の左右差"
        ],
        vertical_spacing=0.06,
        shared_xaxes=True
    )
    
    plot_configs = [
        ("shoulder_angle", "#00d4a1", 1),
        ("pelvis_angle", "#7ecfff", 2),
        ("head_center_x", "#ffd166", 3),
        ("knee_lr_diff", "#c77dff", 4),
        ("ankle_lr_diff", "#ff6b6b", 5),
    ]
    
    for col, color, row in plot_configs:
        if col in df.columns:
            values = df[col].tolist()
            mean_val = np.mean(values)
            
            # メインライン
            fig.add_trace(
                go.Scatter(
                    x=timestamps, y=values,
                    mode='lines+markers',
                    line=dict(color=color, width=2),
                    marker=dict(size=5, color=color),
                    name=col,
                    showlegend=False,
                    hovertemplate=f"t=%{{x}}s<br>value=%{{y:.4f}}<extra></extra>"
                ),
                row=row, col=1
            )
            
            # 平均線
            fig.add_trace(
                go.Scatter(
                    x=timestamps, y=[mean_val] * len(timestamps),
                    mode='lines',
                    line=dict(color=color, width=1, dash='dot'),
                    opacity=0.4,
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=1
            )
    
    fig.update_layout(
        height=900,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0f1629',
        font=dict(color='#8ba4c0', family='Space Mono', size=10),
        margin=dict(t=40, b=40, l=60, r=20),
    )
    
    fig.update_xaxes(
        gridcolor='#1a2540',
        linecolor='#1e2d4a',
        title_text="時間 (秒)",
        title_font=dict(size=9),
        row=5
    )
    fig.update_yaxes(gridcolor='#1a2540', linecolor='#1e2d4a')
    
    # サブタイトルのフォント調整
    for annotation in fig.layout.annotations:
        annotation.font = dict(family='Space Mono', size=10, color='#8ba4c0')
    
    st.plotly_chart(fig, use_container_width=True)


def render_comparison_view(result_before: Dict, result_after: Dict):
    """
    Before / After比較UIを描画する
    """
    tbi_b = result_before["tbi"]
    tbi_a = result_after["tbi"]
    delta = tbi_a - tbi_b
    delta_color = "#00d4a1" if delta >= 0 else "#ff6b6b"
    delta_symbol = "+" if delta >= 0 else ""
    
    # TBI変化表示
    st.markdown(f"""
    <div style="
        background: #141c33;
        border: 1px solid #1e2d4a;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
    ">
        <div style="
            font-family: 'Space Mono', monospace;
            font-size: 0.7rem;
            color: #8ba4c0;
            letter-spacing: 0.2em;
            margin-bottom: 1rem;
        ">TBI CHANGE</div>
        <div style="
            font-family: 'Syne', sans-serif;
            font-size: 3rem;
            font-weight: 800;
            color: #e8f4fd;
        ">
            <span style="color: #ff6b6b;">{tbi_b:.0f}</span>
            <span style="color: #8ba4c0; font-size: 1.5rem; margin: 0 1rem;">→</span>
            <span style="color: #00d4a1;">{tbi_a:.0f}</span>
        </div>
        <div style="
            font-family: 'Space Mono', monospace;
            font-size: 1.2rem;
            color: {delta_color};
            margin-top: 0.5rem;
            font-weight: 700;
        ">{delta_symbol}{delta:.1f} pts</div>
    </div>
    """, unsafe_allow_html=True)
    
    # スコア比較テーブル
    col1, col2 = st.columns(2)
    
    score_items = [
        ("Structure Score", "structure_score", "#00d4a1"),
        ("Fluctuation Score", "fluctuation_score", "#7ecfff"),
        ("Load Path Score", "load_path_score", "#ffd166"),
        ("Recovery Score", "recovery_score", "#c77dff"),
    ]
    
    with col1:
        st.markdown("""
        <div style="
            font-family: 'Space Mono', monospace;
            font-size: 0.7rem;
            color: #ff6b6b;
            letter-spacing: 0.15em;
            margin-bottom: 0.8rem;
        ">◀ BEFORE</div>
        """, unsafe_allow_html=True)
        _render_score_column(result_before, score_items)
    
    with col2:
        st.markdown("""
        <div style="
            font-family: 'Space Mono', monospace;
            font-size: 0.7rem;
            color: #00d4a1;
            letter-spacing: 0.15em;
            margin-bottom: 0.8rem;
        ">AFTER ▶</div>
        """, unsafe_allow_html=True)
        _render_score_column(result_after, score_items)
    
    # 判定ラベル比較
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        info_b = result_before["tbi_info"]
        st.markdown(f"""
        <div style="
            background: {info_b['bg_color']};
            border: 1px solid {info_b['color']}44;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        ">
            <div style="font-size: 1.5rem;">{info_b['emoji']}</div>
            <div style="
                font-family: 'Syne', sans-serif;
                font-size: 1.1rem;
                font-weight: 700;
                color: {info_b['color']};
            ">{info_b['label']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        info_a = result_after["tbi_info"]
        st.markdown(f"""
        <div style="
            background: {info_a['bg_color']};
            border: 1px solid {info_a['color']}44;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        ">
            <div style="font-size: 1.5rem;">{info_a['emoji']}</div>
            <div style="
                font-family: 'Syne', sans-serif;
                font-size: 1.1rem;
                font-weight: 700;
                color: {info_a['color']};
            ">{info_a['label']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # コメント比較
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#8ba4c0;margin-bottom:0.5rem;">BEFORE コメント</div>""", unsafe_allow_html=True)
        for c in result_before.get("comments", [])[:3]:
            st.markdown(f"""<div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#e8f4fd;padding:0.4rem 0;border-bottom:1px solid #1e2d4a;">› {c}</div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#8ba4c0;margin-bottom:0.5rem;">AFTER コメント</div>""", unsafe_allow_html=True)
        for c in result_after.get("comments", [])[:3]:
            st.markdown(f"""<div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#e8f4fd;padding:0.4rem 0;border-bottom:1px solid #1e2d4a;">› {c}</div>""", unsafe_allow_html=True)


def _render_score_column(result: Dict, score_items: List):
    """スコアカラムを描画するヘルパー"""
    for label, key, color in score_items:
        score = result.get(key, 0)
        bar_w = int(score)
        st.markdown(f"""
        <div style="
            background: #141c33;
            border: 1px solid #1e2d4a;
            border-radius: 6px;
            padding: 0.7rem 1rem;
            margin-bottom: 0.4rem;
        ">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem;">
                <span style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#8ba4c0;">{label}</span>
                <span style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:{color};">{score:.1f}</span>
            </div>
            <div style="background:#0a0e1a;border-radius:2px;height:3px;">
                <div style="background:{color};width:{bar_w}%;height:3px;border-radius:2px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
