"""
RetailVision AI — Production Dashboard
Shelf gap detection powered by YOLO11n for retail operations.
"""

import streamlit as st
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent

with open(PROJECT_ROOT / "configs" / "base_config.yaml") as f:
    config = yaml.safe_load(f)

CONF_THRESHOLD = config["inference"]["confidence_threshold"]
IOU_THRESHOLD = config["inference"]["iou_threshold"]

# ================================================================
# PAGE CONFIG & CUSTOM CSS
# ================================================================
st.set_page_config(page_title="RetailVision AI", page_icon="🛒", layout="wide")

st.markdown("""
<style>
    .main-title {
        font-size: 8rem;
        font-weight: 700;
        color: #FF6D28;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #888;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #FF6D28;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FF6D28;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .alert-critical {
        background: #ff000020;
        border-left: 4px solid #ff0000;
        padding: 15px;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .alert-warning {
        background: #ffa50020;
        border-left: 4px solid #ffa500;
        padding: 15px;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .alert-ok {
        background: #00ff0010;
        border-left: 4px solid #00cc00;
        padding: 15px;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #FF6D28;
        border-bottom: 2px solid #FF6D2840;
        padding-bottom: 8px;
        margin-top: 20px;
    }
    .info-box {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.85rem;
        color: #ccc;
        margin: 10px 0;
    }
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.8rem;
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid #333;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ================================================================
# LOAD MODEL
# ================================================================
@st.cache_resource
def load_model():
    model_path = PROJECT_ROOT / "models" / "yolo11n_baseline_v1" / "weights" / "best.pt"
    return YOLO(str(model_path))


model = load_model()


# ================================================================
# HEADER
# ================================================================
st.markdown('<h1 style="font-size: 3.5rem !important; font-weight: 700; color: #FF6D28; margin-bottom: 0;">RetailVision AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time shelf gap detection for retail operations | Powered by YOLO11n</p>', unsafe_allow_html=True)


# ================================================================
# TABS (3 tabs, concise)
# ================================================================
tab1, tab2, tab3 = st.tabs([
    "🔍 Live Detection",
    "💰 ROI Calculator",
    "📊 Model & Architecture",
])


# ================================================================
# TAB 1: LIVE DETECTION
# ================================================================
with tab1:

    # Domain shift notice
    st.markdown(
        '<div class="info-box">'
        '⚠️ <strong>Note:</strong> This model was trained on US big-box retail shelves '
        'Performance may vary on different store formats '
        '(small shops, different shelving, natural lighting). '
        'Production deployment would require fine-tuning per store type.'
        '</div>',
        unsafe_allow_html=True,
    )

    col_upload, col_settings = st.columns([3, 1])

    with col_settings:
        st.markdown('<p class="section-header">Settings</p>', unsafe_allow_html=True)
        conf = st.slider(
            "Confidence", 0.1, 0.9, CONF_THRESHOLD, 0.05,
            help="Lower = more detections (more false alarms). Higher = fewer but more confident.",
        )
        iou = st.slider(
            "IoU (NMS)", 0.1, 0.9, IOU_THRESHOLD, 0.05,
            help="Controls overlap filtering. Higher = allows more overlapping boxes.",
        )

    with col_upload:
        uploaded_file = st.file_uploader("Upload a shelf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)

        with st.spinner("Analyzing shelf..."):
            results = model(img_array, conf=conf, iou=iou, verbose=False)
            boxes = results[0].boxes
            annotated = results[0].plot()[..., ::-1]

        total_gaps = len(boxes)

        # Alert banner
        if total_gaps >= 5:
            st.markdown(
                '<div class="alert-critical">🚨 CRITICAL — '
                + str(total_gaps) + ' gaps detected. Immediate restocking required.</div>',
                unsafe_allow_html=True,
            )
        elif total_gaps >= 2:
            st.markdown(
                '<div class="alert-warning">⚠️ WARNING — '
                + str(total_gaps) + ' gaps detected. Schedule restocking.</div>',
                unsafe_allow_html=True,
            )
        elif total_gaps >= 1:
            st.markdown(
                '<div class="alert-ok">✅ MINOR — '
                + str(total_gaps) + ' gap detected. Monitor.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="alert-ok">✅ ALL CLEAR — Shelf fully stocked.</div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        col_img, col_metrics = st.columns([2, 1])

        with col_img:
            st.image(annotated, caption="Detection result (conf=" + str(conf) + ")", use_container_width=True)

        with col_metrics:
            if total_gaps > 0:
                avg_conf = float(boxes.conf.mean())

                st.markdown(
                    '<div class="metric-card">'
                    '<p class="metric-label">Gaps Found</p>'
                    '<p class="metric-value">' + str(total_gaps) + '</p>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="metric-card">'
                    '<p class="metric-label">Avg Confidence</p>'
                    '<p class="metric-value">' + str(round(avg_conf * 100, 1)) + '%</p>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="metric-card">'
                    '<p class="metric-label">Inference Time</p>'
                    '<p class="metric-value">~2ms</p>'
                    '</div>',
                    unsafe_allow_html=True,
                )

                with st.expander("Detection details"):
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        c = float(box.conf)
                        st.markdown(
                            "**Gap " + str(i + 1) + "** | "
                            "Confidence: " + str(round(c * 100, 1)) + "% | "
                            "Size: " + str(round(x2 - x1)) + "x" + str(round(y2 - y1)) + "px"
                        )

                # Quick impact
                st.markdown("")
                revenue_per_gap = 188
                st.markdown(
                    '<div class="info-box">'
                    '💰 <strong>' + str(total_gaps) + ' gaps</strong> detected early = '
                    '<strong>$' + str(total_gaps * revenue_per_gap) + '</strong> revenue protected '
                    '(see ROI tab for methodology)'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="metric-card">'
                    '<p class="metric-label">Status</p>'
                    '<p class="metric-value">Full</p>'
                    '</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("Upload a shelf image to start detecting gaps. Try any retail shelf photo.")


# ================================================================
# TAB 2: ROI CALCULATOR
# ================================================================
with tab2:
    st.markdown('<p class="section-header">ROI Calculator — Business Impact Estimation</p>', unsafe_allow_html=True)
    st.caption("All projections are simulated estimates. Model metrics are real from the test set. Adjust assumptions below.")

    st.markdown("")

    col_r1, col_r2, col_r3 = st.columns(3)

    with col_r1:
        st.markdown("### Stockout Cost")
        avg_price = st.number_input("Avg product price ($)", value=25, min_value=1, key="roi_price")
        walkaway = st.slider("Customer walkaway rate (%)", 10, 50, 30, key="roi_walk") / 100
        cust_hr = st.number_input("Customers per gap/hour", value=10, min_value=1, key="roi_cust")

    with col_r2:
        st.markdown("### Detection Speed")
        hrs_manual = st.number_input("Hours undetected (manual)", value=3.0, min_value=0.5, step=0.5, key="roi_manual")
        hrs_ai = st.number_input("Hours undetected (AI)", value=0.5, min_value=0.1, step=0.1, key="roi_ai")
        gaps_day = st.slider("Estimated gaps per store/day", 10, 200, 50, key="roi_gaps")

    with col_r3:
        st.markdown("### Operations Cost")
        wage = st.number_input("Associate wage ($/hr)", value=20, min_value=10, key="roi_wage")
        verify_min = st.number_input("False alert verify (min)", value=5, min_value=1, key="roi_verify")
        n_stores = st.number_input("Number of stores", value=100, min_value=1, key="roi_stores")

    # Calculations
    rev_per_gap_hr = avg_price * walkaway * cust_hr
    hrs_saved = hrs_manual - hrs_ai
    rev_saved_per_gap = rev_per_gap_hr * hrs_saved
    cost_per_fa = (verify_min / 60) * wage

    detected = round(gaps_day * 0.90)
    missed = gaps_day - detected
    false_alerts = round(detected * 0.083 / 0.917)
    daily_value = round(detected * rev_saved_per_gap)
    daily_fa_cost = round(false_alerts * cost_per_fa, 2)
    daily_net = daily_value - daily_fa_cost

    st.markdown("---")

    # Step by step
    st.markdown("### Step-by-Step Calculation")

    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        st.markdown(
            "**Revenue lost per gap per hour:**\n\n"
            "$" + str(avg_price) + " x " + str(int(walkaway * 100))
            + "% x " + str(cust_hr) + " customers\n\n"
            "= **$" + str(round(rev_per_gap_hr)) + "/hour**"
        )

    with col_s2:
        st.markdown(
            "**Revenue saved per detected gap:**\n\n"
            "$" + str(round(rev_per_gap_hr)) + "/hour x " + str(hrs_saved) + "h saved\n\n"
            "= **$" + str(round(rev_saved_per_gap)) + "/gap**"
        )

    with col_s3:
        st.markdown(
            "**False alert cost:**\n\n"
            + str(verify_min) + " min = " + str(round(verify_min/60, 2)) + "h x $" + str(wage) + "/hr\n\n"
            "= **$" + str(round(cost_per_fa, 2)) + "/false alert**\n\n"
            "(associate walks to shelf, verifies, returns)"
        )

    st.markdown("---")
    st.markdown("### Daily Store Impact (" + str(gaps_day) + " gaps/day)")

    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    col_d1.metric("Gaps Detected", str(detected), help="gaps/day x 90% recall")
    col_d2.metric("Revenue Saved", "$" + "{:,}".format(daily_value),
                  help=str(detected) + " gaps detected x $" + str(round(rev_saved_per_gap)) + " saved per gap")
    col_d3.metric("False Alert Cost", "-$" + str(daily_fa_cost),
                  help=str(detected) + " detections x 8.3% false positive rate (1-91.7% precision) = " + str(false_alerts) + " false alerts. Each costs $" + str(round(cost_per_fa, 2)) + " in associate time.")
    col_d4.metric("NET Value/Day", "$" + "{:,}".format(daily_net))

    st.markdown("---")
    st.markdown("### Scaled Projections")

    monthly = daily_net * n_stores * 30
    yearly = daily_net * n_stores * 365

    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("Monthly (" + str(n_stores) + " stores)", "$" + "{:,}".format(monthly),
                  help="$" + "{:,}".format(daily_net) + "/day x " + str(n_stores) + " stores x 30 days")
    col_p2.metric("Yearly (" + str(n_stores) + " stores)", "$" + "{:,}".format(yearly),
                  help="$" + "{:,}".format(daily_net) + "/day x " + str(n_stores) + " stores x 365 days")
    col_p3.metric(
        "Missed gaps/day/store", str(missed),
        help=str(gaps_day) + " total gaps x 10% miss rate (1 - 90% recall) = " + str(missed) + " gaps undetected per day."
    )

    st.markdown("---")
    st.caption(
        "**Disclaimer:** All dollar values are SIMULATED estimates. "
        "Model metrics (precision 91.7%, recall 90.0%) are REAL from the held-out test set. "
        "Source: IHL Group reports 21-43% customer walkaway rate on stockouts. "
        "Real ROI requires A/B testing in pilot stores with actual operational data."
    )


# ================================================================
# TAB 3: MODEL & ARCHITECTURE
# ================================================================
with tab3:

    # --- Production Pipeline ---
    st.markdown('<p class="section-header">Production Pipeline — How It Works</p>', unsafe_allow_html=True)
    st.markdown("")

    # Clean linear flow diagram
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor("#0E1117")
    ax.set_facecolor("#0E1117")

    # 5 steps in a clean line
    steps = [
        (0.5, 2, "#4FC3F7", "📷", "Store Camera", "Captures shelf\nevery 15 min"),
        (3.5, 2, "#81C784", "☁️", "Cloud Server", "Receives image\nvia HTTPS"),
        (6.5, 2, "#BA68C8", "🧠", "YOLO11n", "Detects gaps\nin 2ms"),
        (9.5, 2, "#FFD54F", "📊", "Dashboard", "Shows results\n+ alerts"),
        (12.5, 2, "#FF8A65", "🔔", "Action", "Associate\nrestocks shelf"),
    ]

    for x, y, color, icon, label, sublabel in steps:
        box = mpatches.FancyBboxPatch(
            (x, y), 2.5, 2.2,
            boxstyle="round,pad=0.2",
            facecolor=color + "25",
            edgecolor=color,
            linewidth=2,
        )
        ax.add_patch(box)
        ax.text(x + 1.25, y + 1.65, icon, ha="center", va="center", fontsize=20, color="white")
        ax.text(x + 1.25, y + 1.0, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        ax.text(x + 1.25, y + 0.4, sublabel, ha="center", va="center",
                fontsize=8, color="#BBB")

    # Arrows between steps
    for i in range(4):
        x_start = steps[i][0] + 2.5
        x_end = steps[i + 1][0]
        ax.annotate(
            "", xy=(x_end, 3.1), xytext=(x_start, 3.1),
            arrowprops=dict(arrowstyle="-|>", color="white", lw=2),
        )

    ax.text(8, 4.7, "Production Pipeline — End to End", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#FF6D28")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Pipeline details
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("### How It Scales")
        st.markdown(
            "- **Per store:** 20 cameras, 1 scan per aisle every 15 min\n"
            "- **Cloud deployment:** Docker container on any cloud (AWS, GCP, Azure)\n"
            "- **Auto-scaling:** 0 to N instances based on traffic\n"
            "- **Threshold per camera:** Dense aisles use higher threshold, sparse aisles lower\n"
            "- **Data warehouse:** Detection logs stored for trend analysis (Snowflake, BigQuery, Redshift)"
        )

    with col_p2:
        st.markdown("### Output Format")
        st.markdown(
            "Each detection returns a JSON response:\n"
            "```json\n"
            "{\n"
            '  "gaps": 3,\n'
            '  "alert_level": "WARNING",\n'
            '  "detections": [\n'
            '    {"confidence": 0.88, "bbox": [341, 473, 411, 578]}\n'
            "  ]\n"
            "}\n"
            "```\n"
            "Alerts route to Slack, email, or mobile app based on severity."
        )

    st.markdown("---")

    # --- Model Architecture (compact) ---
    st.markdown('<p class="section-header">Model Architecture</p>', unsafe_allow_html=True)

    col_a1, col_a2 = st.columns([1, 1])

    with col_a1:
        st.markdown(
            "**YOLO11n** processes images through three stages:\n\n"
            "1. **Backbone**: Extracts visual features from the image. "
            "Learns patterns like edges, textures, and shapes that distinguish "
            "empty shelf space from products.\n\n"
            "2. **Neck / FPN**: Combines features at three scales "
            "simultaneously — small gaps (between products), medium gaps, "
            "and large empty shelf sections are all detected in parallel.\n\n"
            "3. **Detect Head**: Outputs bounding boxes with confidence "
            "scores. Non-Maximum Suppression (NMS) removes duplicate detections."
        )

    with col_a2:
        st.markdown("### Model Card")
        st.markdown(
            "| Parameter | Value |\n"
            "|---|---|\n"
            "| Architecture | YOLO11n (nano) |\n"
            "| Parameters | 2.6M |\n"
            "| Input size | 640x640 |\n"
            "| Training epochs | 50 |\n"
            "| Optimizer | AdamW (lr=0.002) |\n"
            "| Training time | 34 min (RTX 4080) |\n"
            "| Inference | 2.1ms/image |\n"
        )

    st.markdown("---")

    # --- Honest Metrics ---
    st.markdown('<p class="section-header">Test Set Metrics — Honest Evaluation</p>', unsafe_allow_html=True)
    st.markdown("")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    col_m1.markdown(
        '<div class="metric-card">'
        '<p class="metric-label">Precision</p>'
        '<p class="metric-value">91.7%</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    col_m2.markdown(
        '<div class="metric-card">'
        '<p class="metric-label">Recall</p>'
        '<p class="metric-value">90.0%</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    col_m3.markdown(
        '<div class="metric-card">'
        '<p class="metric-label">mAP@50</p>'
        '<p class="metric-value">87.4%</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    col_m4.markdown(
        '<div class="metric-card">'
        '<p class="metric-label">Inference</p>'
        '<p class="metric-value">2.1ms</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.markdown("### What These Metrics Mean")
        st.markdown(
            "- **Precision 91.7%:** Of every 100 alerts, 92 are real gaps. "
            "8 are false alarms — an associate checks and finds no real gap.\n"
            "- **Recall 90.0%:** The model catches 90 of every 100 real gaps. "
            "10 are missed — these gaps stay undetected until the next manual walk.\n"
            "- **mAP@50 87.4%:** Overall detection quality across all confidence levels.\n"
            "- **Inference 2.1ms:** Processes ~475 images per second."
        )
    st.markdown("---")


# ================================================================
# FOOTER
# ================================================================
st.markdown(
    '<div class="footer">'
    'RetailVision AI | Built with YOLO11 + FastAPI + Streamlit | '
    '<a href="https://github.com/marianunez-data/retail-vision-ai">GitHub</a> | '
    'Compatible with AWS, GCP, Azure deployment'
    '</div>',
    unsafe_allow_html=True,
)
