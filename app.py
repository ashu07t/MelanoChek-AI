import os
import time
import logging
import threading
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import tensorflow.keras.backend as K
import cv2

# ====================  ====================
from flask import send_file
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.units import inch
# ==============================================================
# =========================
# Basic configuration
# =========================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logs
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# =========================
# Labels & risk levels
# =========================
CLASS_LABELS = {
    0: 'Actinic keratoses (akiec)',
    1: 'Basal cell carcinoma (bcc)',
    2: 'Benign keratosis-like lesions (bkl)',
    3: 'Dermatofibroma (df)',
    4: 'Melanoma (mel)' ,
    5: 'Melanocytic nevi (nv)',
    6: 'Vascular lesions (vasc)'

}

RISK_LEVELS = {
    'Actinic keratoses (akiec)': 'High Risk',
   'Basal cell carcinoma (bcc)': 'High Risk ',
    'Benign keratosis-like lesions (bkl)': 'Low Risk',
    'Dermatofibroma (df)': 'Low Risk',
    'Melanoma (mel)': 'High Risk ',
    'Melanocytic nevi (nv)': 'Low Risk',
    'Vascular lesions (vasc)': 'Low Risk'

}

# =========================
# Inference configuration
# =========================
MODEL_PATH = 'templates/resnet_vgg_ensemble.h5'
INPUT_SIZE = (224, 224)

# Decision thresholds
DECISION_THRESHOLD = 60.0        # Below this => low-confidence cancer prediction
OOD_CONF_THRESHOLD = 80.0        # For non-skin/uncertainty gate
OOD_ENTROPY_THRESHOLD = 1.5      # Higher entropy => more uncertain
TOPK = 3

# Options
ENABLE_HAIR_REMOVAL = False      # Set True to inpaint hair
ENABLE_TTA = False               # Test-time augmentation (flips)

# Preprocessing must match training: 'resnet', 'vgg', or 'scale255'
BACKBONE_PREPROCESS = 'resnet'   # change to 'vgg' or 'scale255' if needed

# =========================
# Model globals
# =========================
model = None
model_loaded = False
model_loading = False
model_error = None

# If your Lambda used named helpers, define/register them here
CUSTOM_HELPERS = {
    # 'swish': lambda x: tf.nn.swish(x),
    # 'l2_norm': lambda x: tf.nn.l2_normalize(x, axis=-1),
    # 'preprocess_input': tf.keras.applications.vgg16.preprocess_input,
}

# =========================
# Utility functions
# =========================
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def remove_hair(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(rgb, thresh, 3, cv2.INPAINT_TELEA)
    return inpainted

def apply_backbone_preprocess(img_float):
    if BACKBONE_PREPROCESS == 'resnet':
        return tf.keras.applications.resnet50.preprocess_input(img_float)
    elif BACKBONE_PREPROCESS == 'vgg':
        return tf.keras.applications.vgg16.preprocess_input(img_float)
    else:  # 'scale255'
        return img_float / 255.0

def softmax_entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-8, 1.0)
    return float(-(p * np.log(p)).sum())

def is_likely_skin_rgb(img_rgb: np.ndarray) -> bool:
    """
    Heuristic gate: checks color distribution and edge density
    to quickly reject obvious non-skin images.
    """
    h, w, _ = img_rgb.shape
    ch0 = h // 4; ch1 = 3 * h // 4
    cw0 = w // 4; cw1 = 3 * w // 4
    crop = img_rgb[ch0:ch1, cw0:cw1]

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    hch, sch, vch = cv2.split(hsv)
    mask1 = (hch < 25) | (hch > 160)         # reds/browns
    mask2 = (sch > 20) & (vch > 40)
    skinish = (mask1 & mask2).mean()

    edges = cv2.Canny(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), 30, 100)
    edge_frac = (edges > 0).mean()

    return (skinish > 0.25) and (edge_frac > 0.02)

def preprocess_image(image_path, target_size=INPUT_SIZE):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_rgb = img.copy()

    if ENABLE_HAIR_REMOVAL:
        img = remove_hair(img)

    img = cv2.resize(img, target_size)
    img = img.astype(np.float32)
    img = apply_backbone_preprocess(img)
    img = np.expand_dims(img, axis=0)
    return img, original_rgb

def predict_with_tta(x):
    if not ENABLE_TTA:
        return model.predict(x, verbose=0)
    variants = [x, x[:, :, ::-1, :], x[:, ::-1, :, :]]
    pred = None
    for v in variants:
        p = model.predict(v, verbose=0)
        pred = p if pred is None else pred + p
    return pred / len(variants)

# =========================
# Model loading (robust)
# =========================
def load_model_worker():
    global model, model_loaded, model_loading, model_error

    try:
        model_loading = True
        logger.info("Background: Starting model load")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
        logger.info(f"Model file size: {size_mb:.1f} MB")

        tf.keras.config.enable_unsafe_deserialization()

        custom_objects = {'tf': tf, 'K': K}
        custom_objects.update(CUSTOM_HELPERS)

        from tensorflow.keras.utils import custom_object_scope
        with custom_object_scope(custom_objects):
            mdl = tf.keras.models.load_model(
                MODEL_PATH,
                safe_mode=False,
                compile=False,
                custom_objects=custom_objects
            )

        # Ensure Lambda.function has tf/K in its globals
        try:
            from tensorflow.keras.layers import Lambda
            patched = 0
            for layer in mdl.layers:
                if isinstance(layer, Lambda):
                    fn = layer.function
                    if callable(fn) and hasattr(fn, "__globals__") and isinstance(fn.__globals__, dict):
                        fn.__globals__.setdefault('tf', tf)
                        fn.__globals__.setdefault('K', K)
                        for name, obj in CUSTOM_HELPERS.items():
                            fn.__globals__.setdefault(name, obj)
                        patched += 1
            logger.info(f"Patched {patched} Lambda functions' globals.")
        except Exception as patch_err:
            logger.warning(f"Lambda globals patch warning: {patch_err}")

        # Soft wrap (secondary)
        try:
            from tensorflow.keras.layers import Lambda
            wrapped = 0
            for layer in mdl.layers:
                if isinstance(layer, Lambda):
                    fn = layer.function
                    if callable(fn):
                        def make_wrapped(f):
                            def wrapped_f(x, *args, **kwargs):
                                return f(x, *args, **kwargs)
                            wrapped_f.tf = tf
                            wrapped_f.K = K
                            for name, obj in CUSTOM_HELPERS.items():
                                setattr(wrapped_f, name, obj)
                            return wrapped_f
                        layer.function = make_wrapped(fn)
                        wrapped += 1
            logger.info(f"Wrapped {wrapped} Lambda functions.")
        except Exception as wrap_err:
            logger.warning(f"Lambda wrapper warning: {wrap_err}")

        mdl.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        logger.info("Model recompiled")

        # Dry‑run validation
        dummy = tf.zeros((1, INPUT_SIZE[0], INPUT_SIZE[1], 3), dtype=tf.float32)
        dummy_img = (dummy.numpy()[0] * 255.0).astype(np.uint8)
        dummy_img = apply_backbone_preprocess(dummy_img.astype(np.float32))
        dummy_batch = np.expand_dims(dummy_img, 0)
        _ = mdl(dummy_batch, training=False)
        logger.info("Dry-run forward pass OK")

        model = mdl
        model_loaded = True
        model_loading = False
        logger.info("Model loaded successfully")

    except Exception as e:
        model_error = f"{e}"
        model_loaded = False
        model_loading = False
        logger.error(f"Model loading failed: {e}", exc_info=True)

def start_model_loading():
    th = threading.Thread(target=load_model_worker, daemon=True)
    th.start()
    return th

# =========================
# Prediction with OOD gates
# =========================
def predict_skin_cancer(image_path: str):
    if not model_loaded or model is None:
        return None, "Model not loaded yet. Please wait for model to finish loading."
    try:
        x, original_rgb = preprocess_image(image_path)

        # Non‑skin prefilter
        if not is_likely_skin_rgb(original_rgb):
            return {
                'predicted_class': 'Not a skin lesion image',
                'confidence': 0.0,
                'risk_level': 'N/A',
                'topk': [],
                'all_probabilities': {}
            }, None

        preds = predict_with_tta(x)[0]
        order = np.argsort(preds)[::-1]
        top_idx = int(order[0])
        top_conf = float(preds[top_idx] * 100.0)
        entropy = softmax_entropy(preds)

        # OOD / uncertainty gates
        if (top_conf < OOD_CONF_THRESHOLD) or (entropy > OOD_ENTROPY_THRESHOLD):
            return {
                'predicted_class': 'Uncertain / Possibly not a skin lesion image',
                'confidence': 0.0,
                'risk_level': 'N/A',
                'topk': [],
                'all_probabilities': {}
            }, None

        label = CLASS_LABELS.get(top_idx, 'Unknown')
        risk = RISK_LEVELS.get(label, 'Unknown Risk')

        topk = []
        for i in order[:TOPK]:
            topk.append({'class': CLASS_LABELS[i], 'confidence': float(preds[i] * 100.0)})

        decision = label if top_conf >= DECISION_THRESHOLD else "Uncertain (low confidence)"

        result = {
            'predicted_class': decision,
            'confidence': top_conf,
            'risk_level': risk if decision != "Uncertain (low confidence)" else "Needs review",
            'topk': topk,
            'all_probabilities': {CLASS_LABELS[i]: float(p * 100.0) for i, p in enumerate(preds)}
        }
        return result, None

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return None, f"Prediction error: {e}"

# =========================
# Routes
# =========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model_loading:
            return jsonify({'error': 'Model is still loading. Please wait.'})
        if not model_loaded:
            return jsonify({'error': f'Model not loaded. Error: {model_error or "Unknown error"}'})

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'No file selected'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Use PNG, JPG, JPEG, BMP, or TIFF.'})

        fname = secure_filename(file.filename)
        fname = f"{int(time.time())}_{fname}"
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(fpath)

        result, err = predict_skin_cancer(fpath)
        if err:
            return jsonify({'error': err})

        result['image_url'] = url_for('static', filename=f'uploads/{fname}')
        result['preprocess'] = BACKBONE_PREPROCESS
        result['threshold'] = DECISION_THRESHOLD
        result['tta'] = ENABLE_TTA
        result['hair_removal'] = ENABLE_HAIR_REMOVAL
        result['ood_conf_thresh'] = OOD_CONF_THRESHOLD
        result['ood_entropy_thresh'] = OOD_ENTROPY_THRESHOLD
        return jsonify(result)

    except Exception as e:
        logger.error(f"Predict route error: {e}", exc_info=True)
        return jsonify({'error': 'Server error during prediction'})

@app.route('/model-status')
def model_status():
    return jsonify({
        'loaded': model_loaded,
        'loading': model_loading,
        'error': model_error,
        'model_exists': os.path.exists(MODEL_PATH),
        'preprocess': BACKBONE_PREPROCESS
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_status': 'loaded' if model_loaded else ('loading' if model_loading else 'error'),
        'tensorflow_version': tf.__version__,
        'model_exists': os.path.exists(MODEL_PATH)
    })
@app.route('/generate-report', methods=['POST'])
def generate_report():
    try:
        data = request.get_json()
        patient_info = data.get('patient', {})
        results = data.get('results', {})
        
        # Create PDF in memory
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c5aa0'),
            spaceAfter=30,
            alignment=1  # Center
        )
        
        # Title
        story.append(Paragraph("Skin Lesion Detection Report", title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Report Date
        report_date = datetime.now().strftime("%B %d, %Y %I:%M %p")
        story.append(Paragraph(f"<b>Report Generated:</b> {report_date}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Patient Information Table
        patient_data = [
            ['Patient Information', ''],
            ['Name:', patient_info.get('name', 'N/A')],
            ['Age:', str(patient_info.get('age', 'N/A'))],
            ['Gender:', patient_info.get('gender', 'N/A')]
        ]
        
        patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(patient_table)
        story.append(Spacer(1, 0.5*inch))
        
        # Detection Results
        story.append(Paragraph("<b>Detection Results</b>", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        
        # Primary Detection - Extract from results
        predicted_class = results.get('predicted_class', 'N/A')
        confidence = results.get('confidence', 0)
        risk_level = results.get('risk_level', 'N/A')
        
        # Determine risk color
        risk_color = colors.grey
        if 'High' in risk_level:
            risk_color = colors.red
        elif 'Medium' in risk_level:
            risk_color = colors.orange
        elif 'Low' in risk_level:
            risk_color = colors.green
        
        detection_text = f"""
        <b>Primary Detection:</b> {predicted_class}<br/>
        <b>Confidence:</b> {confidence:.2f}%<br/>
        <b>Risk Level:</b> <font color="{risk_color.hexval()}">{risk_level}</font>
        """
        story.append(Paragraph(detection_text, styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Top Predictions Table
        topk_predictions = results.get('topk', [])
        pred_data = [['Rank', 'Lesion Type', 'Confidence (%)', 'Risk Level']]
        
        for idx, pred in enumerate(topk_predictions, 1):
            lesion_class = pred.get('class', 'N/A')
            lesion_conf = pred.get('confidence', 0)
            lesion_risk = RISK_LEVELS.get(lesion_class, 'N/A')
            
            pred_data.append([
                str(idx),
                lesion_class,
                f"{lesion_conf:.2f}",
                lesion_risk
            ])
        
        pred_table = Table(pred_data, colWidths=[0.75*inch, 2.5*inch, 1.5*inch, 1.25*inch])
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5aa0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(pred_table)
        story.append(Spacer(1, 0.5*inch))
        
        # Disclaimer
        disclaimer = """
        <b>IMPORTANT DISCLAIMER:</b><br/>
        This report is generated by an AI-powered diagnostic tool and should be used for 
        screening purposes only. This is NOT a medical diagnosis. Please consult a qualified 
        dermatologist or healthcare professional for proper medical evaluation and treatment. 
        Early detection and professional assessment are crucial for skin health.
        """
        story.append(Paragraph(disclaimer, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name=f'Skin_Cancer_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return jsonify({'error': str(e)}), 500
# ==================================================================

# =========================
# Entry point
# =========================
if __name__ == '__main__':
    print("🚀 Starting Skin Cancer Detection Application")
    print(f"🔧 TensorFlow: {tf.__version__}")
    print(f"📁 CWD: {os.getcwd()}")
    print(f"📄 Model path: {MODEL_PATH} | exists: {os.path.exists(MODEL_PATH)}")
    if os.path.exists(MODEL_PATH):
        print(f"📊 Model size: {os.path.getsize(MODEL_PATH)/1024/1024:.1f} MB")
    else:
        print("❌ Model file not found. Exiting.")
        raise SystemExit(1)

    print("🔄 Loading model in background...")
    start_model_loading()
    print("🌐 Server: http://localhost:5000  |  Status: http://localhost:5000/model-status")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
