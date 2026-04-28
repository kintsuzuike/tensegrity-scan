"""
app.py
Tensegrity Scan™ MVP
身体バランス可視化プロトタイプ - TBI（Tensegrity Balance Index）解析システム

医療診断ではありません。身体バランスの可視化プロトタイプです。
"""

import streamlit as st
import os
import tempfile
import numpy as np
import cv2

# ページ設定（必ず最初に）
st.set_page_config(
    page_title="Tensegrity Scan™",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# モジュールインポート
from modules.video_utils import (
    save_uploaded_video,
    extract_frames_per_second,
    get_video_info,
    frame_to_rgb,
    resize_frame_for_display
)
from modules.pose_utils import PoseAnalyzer
from modules.metrics import (
    extract_frame_metrics,
    build_metrics_dataframe,
    compute_fluctuation_stats,
    detect_fixed_deviations
)
from modules.scoring import run_full_analysis
from modules.ui import (
    apply_global_styles,
    render_header,
    render_upload_section,
    render_tbi_gauge,
    render_score_cards,
    render_risk_list,
    render_comments,
    render_landmark_figure,
    render_fluctuation_charts,
    render_comparison_view
)


def analyze_video(video_path: str, progress_bar=None, status_text=None) -> dict:
    """
    動画を解析してresult dictを返すメイン処理
    """
    # 1. フレーム抽出
    if status_text:
        status_text.markdown(
            '<div style="font-family:Space Mono,monospace;font-size:0.8rem;color:#7ecfff;">フレーム抽出中...</div>',
            unsafe_allow_html=True
        )
    
    frames, fps, total_frames, timestamps = extract_frames_per_second(video_path)
    
    if not frames:
        st.error("動画からフレームを抽出できませんでした。")
        return None
    
    # 2. Pose解析
    analyzer = PoseAnalyzer()
    all_landmarks = []
    annotated_frames = []
    
    for i, (frame, t) in enumerate(zip(frames, timestamps)):
        if progress_bar:
            progress_bar.progress((i + 1) / len(frames))
        if status_text:
            status_text.markdown(
                f'<div style="font-family:Space Mono,monospace;font-size:0.8rem;color:#7ecfff;">MediaPipe Pose解析中... {i+1}/{len(frames)} フレーム (t={t}s)</div>',
                unsafe_allow_html=True
            )
        
        results = analyzer.process_frame(frame)
        landmarks = analyzer.extract_landmarks_dict(results)
        all_landmarks.append(landmarks)
        
        # アノテーション付きフレームを保存
        annotated = analyzer.draw_landmarks_on_frame(frame, results)
        annotated_frames.append(frame_to_rgb(resize_frame_for_display(annotated, 480)))
    
    analyzer.close()
    
    # 3. 指標算出
    if status_text:
        status_text.markdown(
            '<div style="font-family:Space Mono,monospace;font-size:0.8rem;color:#7ecfff;">指標算出中...</div>',
            unsafe_allow_html=True
        )
    
    df = build_metrics_dataframe(all_landmarks, timestamps)
    
    if df.empty:
        st.error("ランドマークの検出に失敗しました。人物が明確に映っている立位動画を使用してください。")
        return None
    
    stats = compute_fluctuation_stats(df)
    risks = detect_fixed_deviations(df, stats)
    
    # 4. スコア算出
    if status_text:
        status_text.markdown(
            '<div style="font-family:Space Mono,monospace;font-size:0.8rem;color:#7ecfff;">TBI算出中...</div>',
            unsafe_allow_html=True
        )
    
    result = run_full_analysis(df, stats, risks)
    result["df"] = df
    result["stats"] = stats
    result["all_landmarks"] = all_landmarks
    result["annotated_frames"] = annotated_frames
    result["timestamps"] = timestamps
    result["video_info"] = get_video_info(video_path)
    
    return result


def show_result_page(result: dict, prefix: str = ""):
    """
    解析結果ページを表示する
    """
    tbi_info = result["tbi_info"]
    
    # TBI判定バナー
    st.markdown(f"""
    <div style="
        background: {tbi_info['bg_color']};
        border: 1px solid {tbi_info['color']}44;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    ">
        <span style="font-size: 2rem;">{tbi_info['emoji']}</span>
        <div>
            <div style="
                font-family: 'Syne', sans-serif;
                font-size: 1.4rem;
                font-weight: 800;
                color: {tbi_info['color']};
            ">{tbi_info['label']}</div>
            <div style="
                font-family: 'Space Mono', monospace;
                font-size: 0.75rem;
                color: #8ba4c0;
                margin-top: 0.2rem;
            ">{tbi_info['description']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # TBIゲージ + サブスコア
    col_gauge, col_scores = st.columns([1, 1.5])
    
    with col_gauge:
        st.markdown(f"""
        <div style="
            font-family: 'Space Mono', monospace;
            font-size: 0.7rem;
            color: #8ba4c0;
            letter-spacing: 0.2em;
            text-align: center;
            margin-bottom: 0.3rem;
        ">TBI TOTAL SCORE</div>
        """, unsafe_allow_html=True)
        render_tbi_gauge(result["tbi"], tbi_info)
        
        # フレーム情報
        st.markdown(f"""
        <div style="
            text-align: center;
            font-family: 'Space Mono', monospace;
            font-size: 0.68rem;
            color: #8ba4c0;
        ">解析フレーム数: {result['frame_count']} フレーム</div>
        """, unsafe_allow_html=True)
    
    with col_scores:
        st.markdown("""
        <div style="
            font-family: 'Space Mono', monospace;
            font-size: 0.7rem;
            color: #8ba4c0;
            letter-spacing: 0.2em;
            margin-bottom: 0.8rem;
        ">SUB SCORES · TBI = 0.35S + 0.30F + 0.25L + 0.10R</div>
        """, unsafe_allow_html=True)
        render_score_cards(result)
    
    st.markdown("---")
    
    # ランドマーク図 + リスク・コメント
    col_lm, col_info = st.columns([1, 1.2])
    
    with col_lm:
        st.markdown("""<div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#8ba4c0;letter-spacing:0.15em;margin-bottom:0.5rem;">LANDMARK FIGURE · 代表フレーム</div>""", unsafe_allow_html=True)
        
        # アノテーション済みフレーム表示
        if result.get("annotated_frames"):
            mid_idx = len(result["annotated_frames"]) // 2
            st.image(
                result["annotated_frames"][mid_idx],
                caption=f"t={result['timestamps'][mid_idx]}s · MediaPipe Pose",
                use_container_width=True
            )
        
        render_landmark_figure(result["all_landmarks"], frame_idx=len(result["all_landmarks"]) // 2)
    
    with col_info:
        st.markdown("""<div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#8ba4c0;letter-spacing:0.15em;margin-bottom:0.8rem;">RISK AREAS · リスク部位</div>""", unsafe_allow_html=True)
        render_risk_list(result["risks"])
        
        st.markdown("---")
        
        st.markdown("""<div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#8ba4c0;letter-spacing:0.15em;margin-bottom:0.8rem;">COMMENTS · コメント</div>""", unsafe_allow_html=True)
        render_comments(result["comments"])
    
    # 免責事項
    st.markdown("""
    <div style="
        background: #141c33;
        border: 1px solid #1e2d4a;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin-top: 1rem;
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        color: #4a6080;
    ">
    ⚠️ このアプリは身体バランスの可視化プロトタイプです。医療診断・治療の代替にはなりません。
    健康上の問題がある場合は必ず医療専門家にご相談ください。
    </div>
    """, unsafe_allow_html=True)


# ===== メインアプリ =====

apply_global_styles()
render_header()

# ナビゲーション
tab1, tab2, tab3 = st.tabs([
    "📹  解析",
    "📊  揺らぎ解析",
    "⚖️  Before / After 比較"
])


# ===== Tab 1: 解析 =====
with tab1:
    
    if "result" not in st.session_state:
        st.session_state.result = None
    
    # アップロードセクション
    st.markdown("""
    <div style="
        font-family: 'Syne', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #e8f4fd;
        margin-bottom: 0.5rem;
    ">動画アップロード · 解析開始</div>
    <div style="
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #8ba4c0;
        margin-bottom: 1.5rem;
    ">正面または側面からの立位動画を5〜30秒程度アップロードしてください</div>
    """, unsafe_allow_html=True)
    
    uploaded_video = render_upload_section("立位動画をアップロード", key="main_video")
    
    if uploaded_video is not None:
        st.markdown(f"""
        <div style="
            background: #141c33;
            border: 1px solid #00d4a1;
            border-radius: 8px;
            padding: 0.8rem 1.2rem;
            margin-bottom: 1rem;
            font-family: 'Space Mono', monospace;
            font-size: 0.75rem;
            color: #00d4a1;
        ">✓ ファイル受信: {uploaded_video.name} ({uploaded_video.size / 1024:.1f} KB)</div>
        """, unsafe_allow_html=True)
        
        analyze_btn = st.button("🔬  解析開始", key="analyze_main", use_container_width=False)
        
        if analyze_btn:
            # セッション状態をリセット
            st.session_state.result = None
            
            with st.spinner(""):
                progress_container = st.container()
                with progress_container:
                    st.markdown("""
                    <div style="
                        font-family: 'Syne', sans-serif;
                        font-size: 1rem;
                        font-weight: 700;
                        color: #e8f4fd;
                        margin-bottom: 1rem;
                    ">解析中 — しばらくお待ちください</div>
                    """, unsafe_allow_html=True)
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                try:
                    # 一時ファイルに保存
                    video_path = save_uploaded_video(uploaded_video)
                    
                    # 解析実行
                    result = analyze_video(video_path, progress_bar, status_text)
                    
                    # 一時ファイル削除
                    try:
                        os.unlink(video_path)
                    except Exception:
                        pass
                    
                    if result:
                        st.session_state.result = result
                        progress_bar.progress(1.0)
                        status_text.markdown(
                            '<div style="font-family:Space Mono,monospace;font-size:0.8rem;color:#00d4a1;">✓ 解析完了</div>',
                            unsafe_allow_html=True
                        )
                
                except Exception as e:
                    st.error(f"解析中にエラーが発生しました: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # 結果表示
    if st.session_state.result is not None:
        st.markdown("---")
        st.markdown("""
        <div style="
            font-family: 'Syne', sans-serif;
            font-size: 1.2rem;
            font-weight: 700;
            color: #e8f4fd;
            margin-bottom: 1rem;
        ">解析結果</div>
        """, unsafe_allow_html=True)
        show_result_page(st.session_state.result)


# ===== Tab 2: 揺らぎ解析 =====
with tab2:
    
    if st.session_state.get("result") is None:
        st.markdown("""
        <div style="
            background: #141c33;
            border: 1px solid #1e2d4a;
            border-radius: 12px;
            padding: 3rem;
            text-align: center;
        ">
            <div style="font-size: 2rem; margin-bottom: 1rem;">📊</div>
            <div style="
                font-family: 'Space Mono', monospace;
                font-size: 0.85rem;
                color: #8ba4c0;
            ">「解析」タブで動画を解析してください</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        result = st.session_state.result
        df = result["df"]
        
        st.markdown("""
        <div style="
            font-family: 'Syne', sans-serif;
            font-size: 1.2rem;
            font-weight: 700;
            color: #e8f4fd;
            margin-bottom: 0.5rem;
        ">揺らぎ解析 · 時系列グラフ</div>
        <div style="
            font-family: 'Space Mono', monospace;
            font-size: 0.75rem;
            color: #8ba4c0;
            margin-bottom: 1.5rem;
        ">各指標の時系列変動を可視化します。点線は平均値を示します。</div>
        """, unsafe_allow_html=True)
        
        # 統計サマリー
        stats = result["stats"]
        
        summary_cols = st.columns(4)
        summary_items = [
            ("肩角度 変動幅", "shoulder_angle", "range", "°"),
            ("骨盤角度 変動幅", "pelvis_angle", "range", "°"),
            ("頭部X 変動幅", "head_center_x", "range", ""),
            ("膝LR差 変動幅", "knee_lr_diff", "range", ""),
        ]
        
        for col, (label, key, stat_key, unit) in zip(summary_cols, summary_items):
            with col:
                if key in stats:
                    val = stats[key].get(stat_key, 0)
                    std = stats[key].get("std", 0)
                    
                    # 固定化度合い（stdが小さいほど固定化）
                    if std < 0.5 and key in ["shoulder_angle", "pelvis_angle"]:
                        color = "#ff6b6b"
                        note = "固定化"
                    elif std > 3.0 and key in ["shoulder_angle", "pelvis_angle"]:
                        color = "#ffd166"
                        note = "大きな揺れ"
                    else:
                        color = "#00d4a1"
                        note = "健全"
                    
                    st.markdown(f"""
                    <div style="
                        background: #141c33;
                        border: 1px solid #1e2d4a;
                        border-radius: 8px;
                        padding: 0.8rem;
                        text-align: center;
                    ">
                        <div style="font-family:'Space Mono',monospace;font-size:0.62rem;color:#8ba4c0;margin-bottom:0.3rem;">{label}</div>
                        <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;color:{color};">{val:.3f}{unit}</div>
                        <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:{color};margin-top:0.2rem;">{note}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        render_fluctuation_charts(df)


# ===== Tab 3: Before / After 比較 =====
with tab3:
    
    st.markdown("""
    <div style="
        font-family: 'Syne', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #e8f4fd;
        margin-bottom: 0.5rem;
    ">Before / After 比較</div>
    <div style="
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #8ba4c0;
        margin-bottom: 1.5rem;
    ">施術・トレーニング前後の2動画を比較分析します</div>
    """, unsafe_allow_html=True)
    
    if "result_before" not in st.session_state:
        st.session_state.result_before = None
    if "result_after" not in st.session_state:
        st.session_state.result_after = None
    
    col_b, col_a = st.columns(2)
    
    with col_b:
        st.markdown("""<div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#ff6b6b;letter-spacing:0.15em;margin-bottom:0.5rem;">◀ BEFORE</div>""", unsafe_allow_html=True)
        uploaded_before = render_upload_section("Before動画", key="before_video")
        
        if uploaded_before is not None:
            st.markdown(f"""<div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#00d4a1;margin-bottom:0.5rem;">✓ {uploaded_before.name}</div>""", unsafe_allow_html=True)
            
            if st.button("解析 (Before)", key="analyze_before"):
                with st.spinner("Before動画を解析中..."):
                    try:
                        video_path = save_uploaded_video(uploaded_before)
                        result = analyze_video(video_path)
                        try:
                            os.unlink(video_path)
                        except Exception:
                            pass
                        if result:
                            st.session_state.result_before = result
                            st.success("Before解析完了")
                    except Exception as e:
                        st.error(f"エラー: {str(e)}")
        
        if st.session_state.result_before:
            r = st.session_state.result_before
            st.markdown(f"""
            <div style="
                background: #141c33;
                border: 1px solid #ff6b6b44;
                border-radius: 8px;
                padding: 0.8rem;
                text-align: center;
                margin-top: 0.5rem;
            ">
                <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#8ba4c0;">TBI</div>
                <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#ff6b6b;">{r['tbi']:.1f}</div>
                <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#ff6b6b;">{r['tbi_info']['label']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_a:
        st.markdown("""<div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#00d4a1;letter-spacing:0.15em;margin-bottom:0.5rem;">AFTER ▶</div>""", unsafe_allow_html=True)
        uploaded_after = render_upload_section("After動画", key="after_video")
        
        if uploaded_after is not None:
            st.markdown(f"""<div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#00d4a1;margin-bottom:0.5rem;">✓ {uploaded_after.name}</div>""", unsafe_allow_html=True)
            
            if st.button("解析 (After)", key="analyze_after"):
                with st.spinner("After動画を解析中..."):
                    try:
                        video_path = save_uploaded_video(uploaded_after)
                        result = analyze_video(video_path)
                        try:
                            os.unlink(video_path)
                        except Exception:
                            pass
                        if result:
                            st.session_state.result_after = result
                            st.success("After解析完了")
                    except Exception as e:
                        st.error(f"エラー: {str(e)}")
        
        if st.session_state.result_after:
            r = st.session_state.result_after
            st.markdown(f"""
            <div style="
                background: #141c33;
                border: 1px solid #00d4a144;
                border-radius: 8px;
                padding: 0.8rem;
                text-align: center;
                margin-top: 0.5rem;
            ">
                <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#8ba4c0;">TBI</div>
                <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#00d4a1;">{r['tbi']:.1f}</div>
                <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#00d4a1;">{r['tbi_info']['label']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # 比較表示
    if st.session_state.result_before and st.session_state.result_after:
        st.markdown("---")
        st.markdown("""
        <div style="
            font-family: 'Syne', sans-serif;
            font-size: 1rem;
            font-weight: 700;
            color: #e8f4fd;
            margin-bottom: 1rem;
        ">比較結果</div>
        """, unsafe_allow_html=True)
        render_comparison_view(
            st.session_state.result_before,
            st.session_state.result_after
        )
    
    elif not (st.session_state.result_before and st.session_state.result_after):
        st.markdown("""
        <div style="
            background: #141c33;
            border: 1px solid #1e2d4a;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            margin-top: 1rem;
        ">
            <div style="font-size: 2rem; margin-bottom: 0.8rem;">⚖️</div>
            <div style="
                font-family: 'Space Mono', monospace;
                font-size: 0.8rem;
                color: #8ba4c0;
            ">BeforeとAfterの両動画を解析すると比較結果が表示されます</div>
        </div>
        """, unsafe_allow_html=True)
