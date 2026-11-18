
import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import io
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="URINE WISE KIT",
    page_icon="🧪",
    layout="wide"
)

# ==========================================
# 2. PASTEL THEME
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #FFF9F2;
    color: #76515B;
}

[data-testid="stSidebar"] {
    background-color: #F9DDE6;
}

h1, h2, h3 {
    color: #C45F7D !important;
}

p, label {
    color: #76515B;
}

.stButton > button {
    background-color: #F5C9D7;
    color: #78485D;
    border: 1px solid #EFC5D3;
    border-radius: 12px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #F8E5A7;
    color: #76515B;
    border-color: #E8D48F;
}

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #F4D6DF;
    border-radius: 16px;
    padding: 16px;
}

[data-testid="stFileUploader"] {
    background-color: #FFF0F4;
    border: 2px dashed #E9AFC2;
    border-radius: 14px;
    padding: 12px;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #A95776;
    border-bottom-color: #F0B7CB;
}

hr {
    border-color: #F2D5DE;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

if "calibration" not in st.session_state:
    st.session_state.calibration = []


PARAMETERS = [
    "Nitrit",
    "Leukocyte esterase (LE)"
]


# ==========================================
# 4. IMAGE PROCESSING FUNCTIONS
# ==========================================

def load_image(upload):

    return np.array(
        Image.open(
            io.BytesIO(upload.getvalue())
        ).convert("RGB")
    )


def crop_image(image, x0, y0, x1, y1):

    h, w = image.shape[:2]

    x0 = max(0, min(int(x0), w - 1))
    y0 = max(0, min(int(y0), h - 1))

    x1 = max(x0 + 1, min(int(x1), w))
    y1 = max(y0 + 1, min(int(y1), h))

    return image[y0:y1, x0:x1]


def analyze_color(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2LAB
    )

    spaces = {
        "RGB": (image, ["R", "G", "B"]),
        "HSV": (hsv, ["H", "S", "V"]),
        "CIELAB (OpenCV)": (lab, ["L", "a", "b"])
    }

    results = []

    for name, (data, channels) in spaces.items():

        pixels = data.reshape(-1, 3).astype(float)

        mean = pixels.mean(axis=0)
        median = np.median(pixels, axis=0)
        std = pixels.std(axis=0)

        for i, channel in enumerate(channels):

            results.append({
                "Color Space": name,
                "Channel": channel,
                "Mean": round(float(mean[i]), 2),
                "Median": round(float(median[i]), 2),
                "Std Dev": round(float(std[i]), 2)
            })

    rgb_mean = image.reshape(-1, 3).mean(axis=0)

    return pd.DataFrame(results), rgb_mean


# ==========================================
# 5. CALIBRATION FUNCTION
# ==========================================

def match_calibration(rgb, parameter):

    df = pd.DataFrame(
        st.session_state.calibration
    )

    if df.empty:
        return None

    df = df[
        df["Parameter"] == parameter
    ]

    if df.empty:
        return None

    reference = df[
        ["R", "G", "B"]
    ].to_numpy(dtype=float)

    distances = np.linalg.norm(
        reference - rgb.reshape(1, 3),
        axis=1
    )

    index = int(np.argmin(distances))

    return (
        str(df.iloc[index]["Label"]),
        float(distances[index])
    )


# ==========================================
# 6. SIDEBAR MENU
# ==========================================

st.sidebar.markdown(
    "<h1 style='text-align:center;'>🧪 URINE WISE</h1>",
    unsafe_allow_html=True
)

st.sidebar.markdown(
    "<p style='text-align:center;'>Smart Urine Analysis</p>",
    unsafe_allow_html=True
)

menu = st.sidebar.radio(
    "MAIN MENU",
    [
        "🏠 Dashboard",
        "📷 Scan Dipstick",
        "🎨 Color Calibration",
        "📊 Examination History",
        "ℹ️ About Urine Wise"
    ]
)


# ==========================================
# 7. DASHBOARD
# ==========================================

if menu == "🏠 Dashboard":

    st.title("🧪 URINE WISE KIT")

    st.subheader(
        "Smartphone Colorimetry for Urine Dipstick"
    )

    st.markdown("""
    Welcome to **URINE WISE KIT**!

    A research prototype for smartphone-based
    urine dipstick color analysis.
    """)

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Examinations",
        len(st.session_state.history)
    )

    col2.metric(
        "Test Parameters",
        "2"
    )

    col3.metric(
        "Analysis Method",
        "Colorimetry"
    )

    st.markdown("### Explore Urine Wise")

    c1, c2 = st.columns(2)

    with c1:
        st.info(
            "📷 **Scan Dipstick**\n\n"
            "Upload a photo of your urine dipstick."
        )

        st.warning(
            "🎨 **Color Calibration**\n\n"
            "Build your own color reference dataset."
        )

    with c2:
        st.success(
            "📊 **Examination History**\n\n"
            "Review and download previous analyses."
        )

        st.info(
            "🔬 **Color Analysis**\n\n"
            "Extract RGB, HSV, and CIELAB features."
        )


# ==========================================
# 8. SCAN DIPSTICK
# ==========================================

elif menu == "📷 Scan Dipstick":

    st.title("📷 Scan Urine Dipstick")

    st.write(
        "Upload one photo containing the complete urine dipstick."
    )

    upload = st.file_uploader(
        "Upload Dipstick Photo",
        type=["jpg", "jpeg", "png"]
    )

    if upload:

        image = load_image(upload)

        h, w = image.shape[:2]

        st.image(
            image,
            caption="Original Dipstick Image",
            use_container_width=True
        )

        st.subheader("1. Select Strip Area")

        st.caption(
            "Adjust the sliders to isolate the strip from its background."
        )

        c1, c2 = st.columns(2)

        with c1:
            sx0 = st.slider(
                "Left boundary (%)",
                0, 95, 5
            ) / 100

            sy0 = st.slider(
                "Top boundary (%)",
                0, 95, 35
            ) / 100

        with c2:
            sx1 = st.slider(
                "Right boundary (%)",
                5, 100, 95
            ) / 100

            sy1 = st.slider(
                "Bottom boundary (%)",
                5, 100, 65
            ) / 100

        if sx1 <= sx0 or sy1 <= sy0:

            st.error("Invalid strip area.")

            st.stop()

        strip = crop_image(
            image,
            sx0 * w,
            sy0 * h,
            sx1 * w,
            sy1 * h
        )

        st.image(
            strip,
            caption="Cropped Strip",
            use_container_width=True
        )

        st.subheader("2. Configure Test Pads")

        orientation = st.radio(
            "Pad arrangement",
            ["Horizontal", "Vertical"]
        )

        start = st.slider(
            "First pad area (%)",
            0, 40, 10
        ) / 100

        end = st.slider(
            "Last pad area (%)",
            60, 100, 90
        ) / 100

        pad_size = st.slider(
            "Pad size (%)",
            30, 100, 65
        ) / 100

        center = st.slider(
            "Pad center position (%)",
            0, 100, 50
        ) / 100

        if end <= start:

            st.error("Invalid pad arrangement.")

            st.stop()

        sh, sw = strip.shape[:2]

        boxes = []

        if orientation == "Horizontal":

            step = sw * (end - start) / 2

            for i in range(2):

                cx = int(
                    sw * start + (i + 0.5) * step
                )

                pw = max(2, int(step * pad_size))
                ph = max(2, int(sh * pad_size))

                boxes.append((
                    max(0, cx - pw // 2),
                    max(0, int(sh * center - ph // 2)),
                    min(sw, cx + pw // 2),
                    min(sh, int(sh * center + ph // 2))
                ))

        else:

            step = sh * (end - start) / 2

            for i in range(2):

                cy = int(
                    sh * start + (i + 0.5) * step
                )

                ph = max(2, int(step * pad_size))
                pw = max(2, int(sw * pad_size))

                boxes.append((
                    max(0, int(sw * center - pw // 2)),
                    max(0, cy - ph // 2),
                    min(sw, int(sw * center + pw // 2)),
                    min(sh, cy + ph // 2)
                ))

        preview = strip.copy()

        for i, box in enumerate(boxes):

            x0, y0, x1, y1 = box

            cv2.rectangle(
                preview,
                (x0, y0),
                (x1, y1),
                (0, 255, 0),
                2
            )

            cv2.putText(
                preview,
                PARAMETERS[i],
                (x0, max(15, y0 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1
            )

        st.image(
            preview,
            caption="Pad ROI Preview",
            use_container_width=True
        )

        st.caption(
            "Check that both boxes cover the correct reagent pads. "
            "This prototype assumes a fixed pad layout."
        )

        st.subheader("3. Analyze Pad Colors")

        results = []

        for i, box in enumerate(boxes):

            x0, y0, x1, y1 = box

            pad = strip[y0:y1, x0:x1]

            if pad.size == 0:
                st.error("ROI is empty. Adjust the strip settings.")
                continue

            parameter = PARAMETERS[i]

            st.markdown(f"#### {parameter}")

            st.image(
                pad,
                width=220
            )

            table, rgb = analyze_color(pad)

            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True
            )

            st.write(
                f"RGB Mean: "
                f"R={rgb[0]:.2f}, "
                f"G={rgb[1]:.2f}, "
                f"B={rgb[2]:.2f}"
            )

            match = match_calibration(
                rgb,
                parameter
            )

            if match:

                label, distance = match

                st.metric(
                    "Nearest calibration reference",
                    label
                )

                st.caption(
                    f"RGB distance: {distance:.2f}. "
                    "Experimental color matching only."
                )

            else:

                label = "Not calibrated"

                st.info(
                    "No calibration data for this parameter yet."
                )

            results.append({
                "Date": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Parameter": parameter,
                "R": round(float(rgb[0]), 2),
                "G": round(float(rgb[1]), 2),
                "B": round(float(rgb[2]), 2),
                "Reference": label
            })

        if results:

            result_df = pd.DataFrame(results)

            st.subheader("Examination Summary")

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Scan Results",
                result_df.to_csv(index=False).encode("utf-8"),
                "urine_wise_results.csv",
                "text/csv"
            )

            if st.button(
                "Save Examination",
                type="primary"
            ):

                st.session_state.history.extend(
                    results
                )

                st.success(
                    "Examination saved successfully."
                )


# ==========================================
# 9. COLOR CALIBRATION
# ==========================================

elif menu == "🎨 Color Calibration":

    st.title("🎨 Color Calibration")

    st.write(
        "Collect color reference data from urine dipstick reagent pads."
    )

    parameter = st.selectbox(
        "Select Parameter",
        PARAMETERS
    )

    label = st.text_input(
        "Reference Label",
        placeholder="Example: Negative, Positive, 1+, 2+"
    )

    upload = st.file_uploader(
        "Upload One Reference Pad Photo",
        type=["jpg", "jpeg", "png"],
        key="calibration_upload"
    )

    if upload:

        image = load_image(upload)

        h, w = image.shape[:2]

        st.image(
            image,
            caption="Reference Pad Image",
            use_container_width=True
        )

        st.subheader("Select Pad ROI")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            x0 = st.number_input(
                "X Left", 0, w-1, 0
            )

        with c2:
            y0 = st.number_input(
                "Y Top", 0, h-1, 0
            )

        with c3:
            x1 = st.number_input(
                "X Right", 1, w, w
            )

        with c4:
            y1 = st.number_input(
                "Y Bottom", 1, h, h
            )

        if x1 > x0 and y1 > y0:

            roi = crop_image(
                image,
                x0, y0, x1, y1
            )

            st.image(
                roi,
                caption="Selected Reference Area",
                width=250
            )

            _, rgb = analyze_color(roi)

            st.write(
                f"RGB Mean: {rgb.round(2)}"
            )

            if st.button(
                "Add Calibration Data",
                disabled=not label.strip()
            ):

                st.session_state.calibration.append({
                    "Parameter": parameter,
                    "R": float(rgb[0]),
                    "G": float(rgb[1]),
                    "B": float(rgb[2]),
                    "Label": label.strip()
                })

                st.success(
                    "Calibration data added."
                )

        else:

            st.error("Invalid ROI coordinates.")

    if st.session_state.calibration:

        df = pd.DataFrame(
            st.session_state.calibration
        )

        st.subheader("Calibration Dataset")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Calibration CSV",
            df.to_csv(index=False).encode("utf-8"),
            "urine_wise_calibration.csv",
            "text/csv"
        )

        if st.button("Clear Calibration Session"):

            st.session_state.calibration = []

            st.rerun()

    else:

        st.info(
            "No calibration data yet. "
            "You can add reference photos when your dipstick samples are available."
        )


# ==========================================
# 10. EXAMINATION HISTORY
# ==========================================

elif menu == "📊 Examination History":

    st.title("📊 Examination History")

    if st.session_state.history:

        df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download History CSV",
            df.to_csv(index=False).encode("utf-8"),
            "urine_wise_history.csv",
            "text/csv"
        )

        if st.button("Clear History"):

            st.session_state.history = []

            st.rerun()

    else:

        st.info(
            "No examinations recorded yet."
        )


# ==========================================
# 11. ABOUT
# ==========================================

elif menu == "ℹ️ About Urine Wise":

    st.title("ℹ️ About Urine Wise")

    st.markdown("""
    ### URINE WISE KIT

    **Smartphone Colorimetry Innovation for Urine Analysis**

    Urine Wise is a research prototype designed to
    analyze color changes in urine dipstick reagent pads
    using smartphone images and computer vision.

    ### Target Parameters

    - Nitrite
    - Leukocyte esterase (LE)

    ### Technologies

    - Python
    - Streamlit
    - OpenCV
    - NumPy
    - Pandas

    ### Research Notice

    This application is an experimental prototype.
    It is not a medical diagnostic device.

    Analytical and clinical validation is required
    before any health-related use.
    """)


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.markdown(
    "<p style='text-align:center;color:#C45F7D;'>"
    "🧪 URINE WISE KIT | Smartphone Colorimetry"
    "</p>",
    unsafe_allow_html=True
)

st.caption(
    "Research prototype — not a medical diagnostic device."
)
