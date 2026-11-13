
from pathlib import Path
from datetime import datetime
from io import BytesIO
from xml.sax.saxutils import escape

import streamlit as st
import numpy as np
from PIL import Image as PILImage
from ultralytics import YOLO
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage
)


# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PRJ_493 - Smart Malnutrition Detection",
    page_icon="🥣",
    layout="wide"
)

APP_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    APP_DIR / "models" / "best_malnutrition_yolov8n.pt"
)

# ==========================================
# 2. SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:

    st.title("🥣 PRJ_493")

    st.caption("Smart Malnutrition Detection System")

    st.divider()

    selected_page = st.radio(
        "Navigation",
        [
            "Home",
            "AI Screening",
            "Nutrition Awareness",
            "Download Report",
            "About Project"
        ]
    )

    st.divider()

    st.caption("AI-Based Child Health Screening")


# ==========================================
# 3. OTHER APPLICATION PAGES
# ==========================================


if selected_page == "Home":

    # Custom styling for Home page
    st.markdown("""
        <style>
        .hero-section {
            background: linear-gradient(120deg, #123B5D, #247BA0);
            padding: 40px 35px;
            border-radius: 15px;
            color: white;
            margin-bottom: 25px;
        }

        .hero-section h1 {
            color: white;
            font-size: 34px;
        }

        .hero-section p {
            color: #EAF6FF;
            font-size: 16px;
        }

        .info-card {
            background-color: #F0F6FA;
            padding: 22px;
            border-radius: 12px;
            border: 1px solid #D8E6EF;
            min-height: 145px;
        }

        .info-card h3 {
            color: #183B56;
            font-size: 19px;
        }

        .info-card p {
            color: #52616B;
            font-size: 14px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Hero section
    st.markdown("""
        <div class="hero-section">
            <h1>Smart Malnutrition Detection System</h1>
            <p>
                An AI-powered experimental image screening platform
                using Computer Vision and Machine Learning.
            </p>
            <p>
                Supporting awareness and further assessment of
                child nutritional health.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Welcome message
    st.subheader("Welcome!")

    st.write(
        "This platform uses a trained YOLOv8n model to analyze "
        "uploaded images and provide an experimental screening output."
    )

    st.divider()

    # Feature cards
    st.subheader("Platform Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class="info-card">
                <h3>AI Image Screening</h3>
                <p>
                    Upload an image and analyze it using our
                    trained Computer Vision model.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="info-card">
                <h3>Screening Report</h3>
                <p>
                    Generate and download a PDF report containing
                    screening details and annotated images.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.write("")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
            <div class="info-card">
                <h3>Nutrition Awareness</h3>
                <p>
                    Explore basic information about child nutrition
                    and healthy growth practices.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
            <div class="info-card">
                <h3>Computer Vision</h3>
                <p>
                    Explore how Artificial Intelligence and
                    image processing support this project.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.divider()

    # How it works
    st.subheader("How Does It Work?")

    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("### 01")
        st.write("**Upload Image**")
        st.caption("Provide a suitable image for screening.")

    with step2:
        st.markdown("### 02")
        st.write("**AI Analysis**")
        st.caption("The trained YOLO model processes the image.")

    with step3:
        st.markdown("### 03")
        st.write("**View Results**")
        st.caption("Review the model output and download the PDF report.")

    st.divider()

    # Disclaimer
    st.warning(
        "Medical Disclaimer: This application is an experimental "
        "AI screening project, not a medical diagnostic tool. "
        "Its output must not replace professional healthcare assessment."
    )

    st.caption(
        "PRJ_493 | Smart Malnutrition Detection System"
    )

elif selected_page == "Nutrition Awareness":

    st.title("🥗 Nutrition Awareness")

    st.write(
        "Explore nutrition information based on the latest "
        "AI screening result."
    )

    st.divider()

    if "latest_report" not in st.session_state:

        st.info(
            "Please complete an AI screening first to view "
            "nutrition awareness information."
        )

    else:

        report = st.session_state["latest_report"]

        prediction = report["prediction"].strip().lower()

        st.subheader("Latest Screening Information")

        st.write(f"**Child Name:** {report['name']}")

        st.write(f"**AI Prediction:** {report['prediction']}")

        st.divider()

        # HEALTHY PREDICTION

        if prediction == "healthy":

            st.success("Healthy Class Prediction")

            st.markdown("""
            ## 🌱 Supporting Healthy Growth

            The AI model assigned the uploaded image to the
            **Healthy** class.

            ### 1. Healthy Eating Habits

            - Encourage a varied and age-appropriate diet.
            - Include different food groups in meals.
            - Offer fruits and vegetables appropriate for the child's age.
            - Follow recommended infant and young-child feeding practices.

            ### 2. Important Nutrients

            - **Protein:** Supports growth and tissue development.
            - **Iron:** Important for healthy blood and development.
            - **Calcium:** Supports bone and teeth development.
            - **Vitamin A:** Supports normal vision and immune function.

            ### 3. Growth Monitoring

            - Monitor the child's growth regularly.
            - Attend routine healthcare check-ups.
            - Follow age-appropriate nutrition guidance.
            - Consult a healthcare professional if growth concerns arise.

            ### 4. Healthy Lifestyle

            - Maintain proper hygiene.
            - Provide safe drinking water.
            - Encourage age-appropriate physical activity and play.
            - Maintain healthy sleep routines.
            """)

            st.info(
                "Remember: A Healthy model prediction does not "
                "confirm that a child is nutritionally healthy. "
                "Continue routine growth monitoring and healthcare."
            )

        # MALNOURISHED PREDICTION

        elif prediction == "malnourished":

            st.warning("Malnourished Class Prediction")

            st.markdown("""
            ## ❤️ Understanding Nutritional Risk

            The AI model assigned the uploaded image to the
            **Malnourished** class. This is an experimental
            model prediction and requires professional assessment.

            ### 1. Seek Professional Assessment

            - Consult a qualified pediatrician or healthcare professional.
            - Request an appropriate nutritional and growth assessment.
            - Follow professional advice regarding further evaluation.
            - Do not rely on the AI prediction alone.

            ### 2. Understanding Important Nutrients

            - **Protein:** Supports normal growth and tissue development.
            - **Iron:** Helps support healthy blood production.
            - **Calcium:** Supports bone development.
            - **Vitamin A:** Supports normal vision and immune function.
            - **Zinc:** Supports normal growth and immune function.

            ### 3. Nutrition Support Awareness

            - Follow an age-appropriate feeding plan recommended by a professional.
            - Provide safe and hygienically prepared food.
            - Follow prescribed nutritional supplements only when advised.
            - Avoid making sudden dietary changes without professional guidance.

            ### 4. Regular Monitoring

            - Attend recommended follow-up appointments.
            - Monitor growth using appropriate clinical measurements.
            - Discuss feeding difficulties or changes in appetite with a professional.
            - Follow the healthcare professional's recommendations.
            """)

            st.warning(
                "Important: This AI output is not a diagnosis. "
                "A qualified healthcare professional must assess "
                "the child's nutritional status."
            )

        # NO DETECTION OR UNKNOWN CLASS

        else:

            st.info("Nutrition Information")

            st.write(
                "The AI model did not produce a recognized "
                "Healthy or Malnourished class prediction."
            )

            st.markdown("""
            - Ensure the uploaded image is clear.
            - Consult a healthcare professional if there are concerns.
            - Use appropriate clinical measurements for nutritional assessment.
            """)

        st.divider()

        st.caption(
            "Nutrition awareness information is educational only. "
            "The AI screening result must not be used as a substitute "
            "for professional medical assessment."
        )

    

elif selected_page == "Download Report":

    st.title("📄 Download Screening Report")

    st.write(
        "View your latest AI screening report "
        "and download it in PDF format."
    )

    st.divider()

    if "latest_pdf" in st.session_state:

        report = st.session_state["latest_report"]

        st.subheader("Latest Screening Report")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Child Name", report["name"])
            st.metric("Age", report["age"])

        with col2:
            st.metric("Predicted Class", report["prediction"])
            st.metric("Confidence Score", report["confidence"])

        st.write(f"**Gender:** {report['gender']}")
        st.write(f"**Screening Date:** {report['date']}")
        st.write(f"**Detection Status:** {report['status']}")

        st.divider()

        st.subheader("Download Your Report")

        safe_name = "".join(
            character if character.isalnum() or character in "-_"
            else "_"
            for character in report["name"]
        )

        st.download_button(
            label="⬇️ Download Report Card (PDF)",
            data=st.session_state["latest_pdf"],
            file_name=f"{safe_name}_Health_Report.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

        st.success("Your screening report is ready.")

        st.caption(
            "The PDF contains the screening details, "
            "uploaded image, annotated image and recommendations."
        )

    else:

        st.info(
            "No screening report is available yet. "
            "Please complete an AI screening first."
        )

    st.warning(
        "This report contains experimental AI screening results "
        "and is not a medical diagnosis."
    )


elif selected_page == "About Project":

    st.title("About Our Project")

    st.markdown("""
    ### 🥣 Smart Malnutrition Detection System

    PRJ_493 is an AI-based project that explores the use
    of Computer Vision and Machine Learning for
    experimental image-based malnutrition screening.
    """)

    st.divider()

    st.subheader("🎯 Project Objectives")

    st.markdown("""
    - Develop an AI-based image screening application.
    - Apply Computer Vision for image analysis.
    - Use a trained YOLOv8n model for object detection.
    - Display prediction results and annotated images.
    - Generate downloadable PDF screening reports.
    """)

    st.divider()

    st.subheader("⚙️ Technologies Used")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        - Python
        - YOLOv8n
        - Computer Vision
        - NumPy
        """)

    with col2:
        st.markdown("""
        - Streamlit
        - Pillow
        - ReportLab
        - OpenCV
        """)

    st.divider()

    st.subheader("🔄 System Workflow")

    st.markdown("""
    1. Enter child information.
    2. Upload a child image.
    3. Process the image using the trained YOLO model.
    4. Display the predicted class and confidence score.
    5. Generate an annotated image.
    6. Create and download the PDF report.
    """)

    st.divider()

    st.subheader("👥 Project Information")

    st.write("**Project ID:** PRJ_493")

    st.write(
        "**Project Domain:** Machine Learning "
        "and Computer Vision"
    )


    st.write(
        "**Deployment:** Streamlit Web Application"
    )

    st.divider()

    st.subheader("⚠️ Disclaimer")

    st.warning(
        "This project is an experimental AI screening system. "
        "It is not a medical diagnostic tool. All results "
        "require appropriate professional healthcare assessment."
    )


# Stop rendering here for pages other than AI Screening.
if selected_page != "AI Screening":
    st.stop()


# ==========================================
# 2. LOAD TRAINED YOLO MODEL
# ==========================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    return model


# ==========================================
# 3. GENERATE PDF REPORT
# ==========================================


def generate_pdf(report):

    pdf_buffer = BytesIO()

    page_width, page_height = A4

    c = pdf_canvas.Canvas(
        pdf_buffer,
        pagesize=A4
    )

    # ----------------------------------
    # COLOURS
    # ----------------------------------

    navy = colors.HexColor("#123B70")
    blue = colors.HexColor("#1765A8")
    light_blue = colors.HexColor("#EAF3FC")
    border = colors.HexColor("#C7D8E8")
    dark = colors.HexColor("#183047")
    grey = colors.HexColor("#5C6B78")
    green = colors.HexColor("#198754")
    red = colors.HexColor("#C83232")
    orange = colors.HexColor("#D98220")

    # ----------------------------------
    # HELPER FUNCTIONS
    # ----------------------------------

    def draw_card(x, y, w, h, fill=colors.white):

        c.setFillColor(fill)
        c.setStrokeColor(border)
        c.roundRect(
            x, y, w, h,
            9, fill=1, stroke=1
        )

    def draw_text(x, y, text, size=9, color=dark,
                  bold=False):

        c.setFillColor(color)

        c.setFont(
            "Helvetica-Bold" if bold else "Helvetica",
            size
        )

        c.drawString(x, y, str(text))

    def draw_image(image_bytes, x, y, w, h):

        try:

            image = PILImage.open(
                BytesIO(image_bytes)
            ).convert("RGB")

            image.thumbnail((900, 900))

            c.drawImage(
                ImageReader(image),
                x, y,
                width=w,
                height=h,
                preserveAspectRatio=True,
                anchor="c",
                mask="auto"
            )

        except Exception:

            draw_text(
                x + 10, y + h / 2,
                "Image unavailable",
                9, grey
            )

    def draw_paragraph(text, x, y, width,
                       font_size=8, color=dark):

        style = ParagraphStyle(
            "ReportText",
            fontName="Helvetica",
            fontSize=font_size,
            leading=font_size + 3,
            textColor=color
        )

        paragraph = Paragraph(
            escape(str(text)),
            style
        )

        _, height = paragraph.wrap(width, 200)

        paragraph.drawOn(
            c, x, y - height
        )

        return height

    # ----------------------------------
    # 1. TOP HEADER
    # ----------------------------------

    c.setFillColor(navy)

    c.roundRect(
        25,
        page_height - 90,
        page_width - 50,
        65,
        12,
        fill=1,
        stroke=0
    )

    c.setFillColor(colors.white)

    c.setFont("Helvetica-Bold", 17)

    c.drawCentredString(
        page_width / 2,
        page_height - 53,
        "SMART MALNUTRITION DETECTION SYSTEM"
    )

    c.setFont("Helvetica", 10)

    c.drawCentredString(
        page_width / 2,
        page_height - 73,
        "AI SCREENING REPORT CARD | PRJ_493"
    )

    c.setFillColor(grey)

    c.setFont("Helvetica", 8)

    c.drawCentredString(
        page_width / 2,
        page_height - 105,
        "Powered by Machine Learning and Computer Vision"
    )

    # ----------------------------------
    # 2. CHILD DETAILS
    # ----------------------------------

    section_y = page_height - 245

    draw_card(
        25, section_y,
        page_width - 50, 120
    )

    draw_text(
        40, section_y + 98,
        "1. CHILD DETAILS",
        13, navy, True
    )

    details = [
        ("Name", report["name"]),
        ("Age", report["age"]),
        ("Gender", report["gender"]),
        ("Date", report["date"])
    ]

    y = section_y + 75

    for label, value in details:

        draw_text(
            42, y,
            label + ":",
            9, dark, True
        )

        draw_text(
            115, y,
            value,
            9, dark
        )

        y -= 19

    # Child uploaded image

    draw_image(
        report["original_image"],
        page_width - 145,
        section_y + 20,
        95,
        85
    )

    draw_text(
        page_width - 142,
        section_y + 9,
        "Uploaded Child Image",
        7, grey
    )

    # ----------------------------------
    # 3. COMPUTER VISION ANALYSIS
    # ----------------------------------

    analysis_y = page_height - 445

    draw_card(
        25, analysis_y,
        page_width - 50, 180
    )

    draw_text(
        40, analysis_y + 157,
        "2. COMPUTER VISION ANALYSIS",
        13, navy, True
    )

    draw_text(
        42, analysis_y + 130,
        "AI Annotated Image",
        10, dark, True
    )

    draw_image(
        report["annotated_image"],
        42,
        analysis_y + 20,
        220,
        100
    )

    # Model information

    draw_card(
        285,
        analysis_y + 85,
        245,
        55,
        light_blue
    )

    draw_text(
        298, analysis_y + 120,
        "MODEL USED",
        8, grey, True
    )

    draw_text(
        298, analysis_y + 101,
        "YOLOv8n",
        11, navy, True
    )

    draw_card(
        285,
        analysis_y + 20,
        245,
        55,
        light_blue
    )

    draw_text(
        298, analysis_y + 55,
        "CONFIDENCE THRESHOLD",
        8, grey, True
    )

    draw_text(
        298, analysis_y + 36,
        report["threshold"],
        11, navy, True
    )

    # ----------------------------------
    # 4. SCREENING RESULT
    # ----------------------------------

    result_y = page_height - 555

    prediction = report["prediction"]

    if prediction.lower() == "malnourished":

        result_color = red

    elif prediction.lower() == "healthy":

        result_color = green

    else:

        result_color = orange

    c.setFillColor(result_color)

    c.roundRect(
        25,
        result_y,
        page_width - 50,
        80,
        10,
        fill=1,
        stroke=0
    )

    c.setFillColor(colors.white)

    c.setFont("Helvetica-Bold", 17)

    c.drawCentredString(
        page_width / 2,
        result_y + 49,
        "AI SCREENING RESULT"
    )

    c.setFont("Helvetica-Bold", 15)

    c.drawCentredString(
        page_width / 2,
        result_y + 27,
        prediction.upper()
    )

    c.setFont("Helvetica", 9)

    c.drawCentredString(
        page_width / 2,
        result_y + 11,
        "Confidence: " + report["confidence"]
    )

    # ----------------------------------
    # 5. CONFIDENCE INDICATOR
    # ----------------------------------

    confidence_y = page_height - 620

    draw_card(
        25,
        confidence_y,
        page_width - 50,
        48,
        light_blue
    )

    draw_text(
        40,
        confidence_y + 29,
        "MODEL CONFIDENCE SCORE",
        9, navy, True
    )

    try:

        confidence_value = float(
            report["confidence"].replace("%", "")
        ) / 100

    except Exception:

        confidence_value = 0

    bar_x = 40
    bar_y = confidence_y + 10
    bar_width = 400
    bar_height = 9

    c.setFillColor(colors.HexColor("#D5DEE7"))

    c.roundRect(
        bar_x, bar_y,
        bar_width, bar_height,
        5, fill=1, stroke=0
    )

    c.setFillColor(blue)

    c.roundRect(
        bar_x, bar_y,
        bar_width * max(0, min(confidence_value, 1)),
        bar_height,
        5, fill=1, stroke=0
    )

    draw_text(
        455,
        confidence_y + 10,
        report["confidence"],
        9, navy, True
    )

    # ----------------------------------
    # 6. RECOMMENDATIONS
    # ----------------------------------

    recommendations_y = page_height - 775

    draw_card(
        25,
        recommendations_y,
        page_width - 50,
        140
    )

    draw_text(
        40,
        recommendations_y + 118,
        "3. RECOMMENDATIONS",
        13, navy, True
    )

    y = recommendations_y + 95

    for item in report["recommendations"]:

        c.setFillColor(green)

        c.circle(
            47, y + 2,
            3,
            fill=1,
            stroke=0
        )

        height = draw_paragraph(
            item,
            58,
            y + 7,
            page_width - 105,
            8,
            dark
        )

        y -= max(height + 9, 18)

    # ----------------------------------
    # 7. MEDICAL DISCLAIMER
    # ----------------------------------

    disclaimer = (
        "This application provides an experimental, image-based "
        "AI screening indication. It does not diagnose malnutrition "
        "or determine a child's overall nutritional status. The "
        "confidence score is a model output, not the probability "
        "that the child has a nutritional disorder. Consult a "
        "qualified healthcare professional for assessment using "
        "appropriate growth measurements, medical history and "
        "clinical evaluation."
    )

    draw_text(
        30, 52,
        "MEDICAL DISCLAIMER",
        8, navy, True
    )

    draw_paragraph(
        disclaimer,
        30, 43,
        page_width - 60,
        6.5,
        grey
    )

    # ----------------------------------
    # FOOTER
    # ----------------------------------

    c.setStrokeColor(border)

    c.line(
        25, 25,
        page_width - 25, 25
    )

    c.setFont("Helvetica", 7)

    c.setFillColor(grey)

    c.drawCentredString(
        page_width / 2,
        14,
        "Generated by PRJ_493 Smart Malnutrition Detection System"
    )

    c.save()

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue()


# ==========================================
# 4. PROFESSIONAL AI SCREENING HEADER
# ==========================================

st.markdown("""
    <div style="
        background: linear-gradient(120deg, #183B56, #2878A5);
        padding: 28px;
        border-radius: 14px;
        color: white;
        margin-bottom: 22px;
    ">
        <h1 style="color:white; margin-bottom:8px;">
            AI Child Screening
        </h1>
        <p style="font-size:16px; margin-bottom:0;">
            Upload an image to perform experimental
            AI-based image screening.
        </p>
    </div>
""", unsafe_allow_html=True)


# ==========================================
# 5. CHILD INFORMATION AND IMAGE UPLOAD
# ==========================================

st.markdown("## Screening Information")

st.caption(
    "Enter the child's details and upload an image "
    "to begin the screening process."
)

with st.form("child_screening_form"):

    # CHILD INFORMATION

    st.markdown("### 👤 Child Information")

    col1, col2 = st.columns(2)

    with col1:

        child_name = st.text_input(
            "Child Name",
            placeholder="Enter child's name"
        )

        age_years = st.number_input(
            "Age (Years)",
            min_value=0,
            max_value=18,
            value=2
        )

    with col2:

        age_months = st.number_input(
            "Additional Months",
            min_value=0,
            max_value=11,
            value=0
        )

        gender = st.selectbox(
            "Gender",
            ["Select", "Male", "Female", "Other"]
        )

    st.divider()

    # IMAGE UPLOAD

    st.markdown("### 📷 Upload Child Image")

    uploaded_file = st.file_uploader(
        "Choose a clear JPG, JPEG or PNG image",
        type=["jpg", "jpeg", "png"],
        help="Upload an image suitable for your project's model."
    )

    st.caption(
        "For better image processing, use a clear image "
        "with adequate lighting."
    )

    st.divider()

    # MODEL SETTINGS

    st.markdown("### ⚙️ Model Settings")

    confidence_threshold = st.slider(
        "Minimum Detection Confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05
    )

    st.caption(
        "This setting controls the minimum model confidence "
        "required to display a detection."
    )

    st.divider()

    # SUBMIT BUTTON

    submit = st.form_submit_button(
        "🔍 Run AI Screening",
        type="primary",
        use_container_width=True
    )


# MEDICAL DISCLAIMER

st.info(
    "Important: This application provides experimental "
    "AI-based image screening only. It is not a medical "
    "diagnostic tool. Consult a qualified healthcare "
    "professional for nutritional assessment."
)


# ==========================================
# 5. RUN AI MODEL
# ==========================================

if submit:

    if not child_name.strip():

        st.error("Please enter the child's name.")

    elif gender == "Select":

        st.error("Please select the child's gender.")

    elif uploaded_file is None:

        st.error("Please upload a child image.")

    else:

        try:

            # Read uploaded image
            image_bytes = uploaded_file.getvalue()

            original_image = PILImage.open(
                BytesIO(image_bytes)
            ).convert("RGB")

            # Load trained model
            with st.spinner("Loading trained YOLO model..."):

                model = load_model()

            # Run prediction
            with st.spinner("Analyzing uploaded image..."):

                results = model.predict(
                    source=original_image,
                    imgsz=640,
                    conf=confidence_threshold,
                    verbose=False
                )

            result = results[0]

            # Generate annotated image
            annotated_array = result.plot()

            annotated_array = annotated_array[:, :, ::-1]

            annotated_image = PILImage.fromarray(
                annotated_array
            )

            # Check detections
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:

                confidence_values = (
                    boxes.conf.cpu().numpy()
                )

                class_values = (
                    boxes.cls.cpu().numpy()
                )

                best_index = int(
                    np.argmax(confidence_values)
                )

                best_class_id = int(
                    class_values[best_index]
                )

                prediction = str(
                    model.names[best_class_id]
                )

                confidence = float(
                    confidence_values[best_index]
                )

                status = "Detection Found"

                if prediction.lower() == "malnourished":

                    recommendations = [
                        "Consult a qualified pediatrician or healthcare professional for nutritional assessment.",
                        "The child's growth should be evaluated using appropriate clinical measurements.",
                        "Do not make treatment or dietary decisions based only on this AI prediction."
                    ]

                elif prediction.lower() == "healthy":

                    recommendations = [
                        "The model assigned the image to the Healthy class.",
                        "Continue routine growth monitoring and age-appropriate nutrition.",
                        "Consult a healthcare professional if there are concerns about the child's growth or health."
                    ]

                else:

                    recommendations = [
                        "The model returned a class that requires further interpretation.",
                        "Consult a qualified healthcare professional for assessment.",
                        "Do not use this prediction alone to make healthcare decisions."
                    ]

            else:

                prediction = "No Detection"

                confidence = None

                status = "No object detected above the selected threshold"

                recommendations = [
                    "No reliable class prediction was produced for this image.",
                    "Try a clear, well-lit image with the child fully visible.",
                    "Seek professional healthcare advice if there are concerns about the child's health."
                ]

            # Prepare report data
            confidence_text = (
                f"{confidence * 100:.2f}%"
                if confidence is not None
                else "Not Available"
            )

            age_text = (
                f"{int(age_years)} years, "
                f"{int(age_months)} months"
            )

            screening_date = datetime.now().astimezone().strftime(
                "%d-%m-%Y %I:%M %p"
            )

            # Convert annotated image to bytes
            annotated_buffer = BytesIO()

            annotated_image.save(
                annotated_buffer,
                format="JPEG",
                quality=90
            )

            annotated_bytes = annotated_buffer.getvalue()

            report = {

                "name": child_name.strip(),

                "age": age_text,

                "gender": gender,

                "date": screening_date,

                "prediction": prediction,

                "status": status,

                "confidence": confidence_text,

                "threshold": f"{confidence_threshold:.2f}",

                "original_image": image_bytes,

                "annotated_image": annotated_bytes,

                "recommendations": recommendations
            }

            # Generate PDF
            pdf_data = generate_pdf(report)

            # Save report in session
            st.session_state["latest_report"] = report

            st.session_state["latest_pdf"] = pdf_data

        except Exception as error:

            st.error(
                f"Screening failed: {error}"
            )



# ==========================================
# 7. PROFESSIONAL AI SCREENING RESULTS
# ==========================================

if "latest_report" in st.session_state:

    report = st.session_state["latest_report"]

    pdf_data = st.session_state["latest_pdf"]

    st.divider()

    # RESULTS HEADER

    st.markdown("""
        <div style="
            background-color:#EAF3F8;
            padding:20px;
            border-radius:12px;
            margin-bottom:20px;
        ">
            <h2 style="color:#183B56; margin-bottom:5px;">
                AI Screening Results
            </h2>
            <p style="color:#52616B; margin-bottom:0;">
                Summary of the image-based model output.
            </p>
        </div>
    """, unsafe_allow_html=True)


    # CHILD DETAILS

    st.subheader("👤 Child Information")

    with st.container(border=True):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Child Name**")
            st.write(report["name"])

        with col2:
            st.write("**Age**")
            st.write(report["age"])

        with col3:
            st.write("**Gender**")
            st.write(report["gender"])

        st.caption(
            f"Screening Date: {report['date']}"
        )


    # PREDICTION SUMMARY

    st.subheader("📊 Prediction Summary")

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.metric(
                label="Predicted Class",
                value=report["prediction"]
            )

    with col2:

        with st.container(border=True):

            st.metric(
                label="Confidence Score",
                value=report["confidence"]
            )


    # DETECTION STATUS

    st.markdown("### Detection Status")

    st.info(report["status"])

    st.caption(
        "The confidence score represents the model's output "
        "for its selected detection. It is not the probability "
        "that a child has malnutrition."
    )


    # IMAGE COMPARISON

    st.divider()

    st.subheader("🖼️ Image Analysis")

    st.write(
        "Compare the uploaded image with the image "
        "annotated by the YOLO model."
    )

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.markdown("**Original Uploaded Image**")

            st.image(
                report["original_image"],
                use_container_width=True
            )

    with col2:

        with st.container(border=True):

            st.markdown("**AI Annotated Image**")

            st.image(
                report["annotated_image"],
                use_container_width=True
            )


    # RECOMMENDATIONS

    st.divider()

    st.subheader("📋 Recommendations")

    with st.container(border=True):

        for index, item in enumerate(
            report["recommendations"], start=1
        ):

            st.markdown(
                f"**{index}.** {item}"
            )


    # DOWNLOAD PDF

    st.divider()

    st.subheader("📄 Download Screening Report")

    st.write(
        "Download the generated PDF containing child information, "
        "AI screening output, images and recommendations."
    )

    safe_name = "".join(
        character if character.isalnum() or character in "-_"
        else "_"
        for character in report["name"]
    )

    st.download_button(
        label="⬇️ Download Report Card (PDF)",
        data=pdf_data,
        file_name=f"{safe_name}_Health_Report.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True
    )


    # MEDICAL DISCLAIMER

    st.warning(
        "Medical Disclaimer: This is an experimental AI-based "
        "image screening application, not a diagnostic tool. "
        "The model output must not be used alone to determine "
        "a child's nutritional status. Consult a qualified "
        "healthcare professional for proper assessment."
    )
