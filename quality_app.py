"""
quality_app.py

Streamlit dashboard for the fingerprint QC gate. Upload an image, see the
composite score, a pass/fail badge, and per-metric breakdowns. Sidebar
sliders let you tune thresholds live and immediately re-run the gate.
"""

import cv2
import numpy as np
import streamlit as st

from quality_assessment import quality_gate

st.set_page_config(page_title="Fingerprint QC Gate", layout="wide")
st.title("📱 Contactless Fingerprint Quality Control System")

# --- Sidebar: live threshold tuning -----------------------------------------
st.sidebar.header("QC Threshold Settings")
blur_thresh = st.sidebar.slider("Blur Threshold (Laplacian Var)", 5.0, 50.0, 10.0)
min_bright = st.sidebar.slider("Min Brightness", 20, 80, 50)
max_bright = st.sidebar.slider("Max Brightness", 180, 240, 210)
max_glare = st.sidebar.slider("Max Glare Fraction", 0.01, 0.15, 0.05)
min_roi = st.sidebar.slider("Min ROI Fraction", 0.05, 0.40, 0.15)
ridge_thresh = st.sidebar.slider("Ridge Clarity Threshold", 5.0, 40.0, 15.0)

thresholds = {
    "blur_threshold": blur_thresh,
    "brightness_min": min_bright,
    "brightness_max": max_bright,
    "glare_max": max_glare,
    "roi_min": min_roi,
    "ridge_threshold": ridge_thresh,
}

uploaded_file = st.file_uploader("Upload Fingerprint Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # NOTE: dtype must be np.uint8, not the bare builtin `uint8`.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    res = quality_gate(image_bgr, thresholds=thresholds)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            caption="Uploaded Image",
            use_container_width=True,
        )

    with col2:
        score = res["composite_score"]
        if res["passed"]:
            st.success(f"### Composite Score: {score} / 100 -- PASSED")
        else:
            st.error(f"### Composite Score: {score} / 100 -- REJECTED")

        st.info(f"**Guidance:** {res['guidance']}")

        st.markdown("#### Quality Checks Breakdown")
        st.write(
            f"Blur Check: {'✅ PASS' if not res['blur']['is_blurry'] else '❌ FAIL'} "
            f"(Score: {res['blur']['blur_score']})"
        )
        st.write(
            f"Brightness Check: "
            f"{'✅ PASS' if not (res['brightness']['too_dark'] or res['brightness']['too_bright']) else '❌ FAIL'} "
            f"(Value: {res['brightness']['brightness']})"
        )
        st.write(
            f"Glare Check: {'✅ PASS' if not res['glare']['has_glare'] else '❌ FAIL'} "
            f"(Ratio: {res['glare']['glare_fraction']})"
        )
        st.write(
            f"ROI Check: {'✅ PASS' if res['roi']['roi_complete'] else '❌ FAIL'} "
            f"(Ratio: {res['roi']['roi_fraction']})"
        )
        st.write(
            f"Ridge Clarity: {'✅ PASS' if res['ridge']['ridges_clear'] else '❌ FAIL'} "
            f"(Score: {res['ridge']['ridge_score']})"
        )
else:
    st.caption("Upload a fingerprint photo to run it through the QC gate.")
