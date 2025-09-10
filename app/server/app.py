from flask import Flask, request, jsonify, send_from_directory
from bson import ObjectId
from bson.errors import InvalidId
from torchvision.models import ResNet50_Weights
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from flask_cors import CORS
import os
import uuid
import matplotlib.cm as cm
import traceback

# Rate limiting
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# JWT (custom)
import jwt
from functools import wraps
from datetime import datetime, timedelta

# ML deps
import joblib
import numpy as np
from scipy.stats import norm

# Torch / vision
import h5py
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from pymongo import ASCENDING, DESCENDING

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Limiter (Redis by default; in-memory if not reachable)
limiter_storage = os.environ.get("REDIS_URL") or "memory://"
limiter = Limiter(
    key_func=get_remote_address,
    app=None,
    default_limits=["200 per day", "50 per hour"],
    storage_uri=limiter_storage
)

# Flask app
app = Flask(__name__)
CORS(app)  # tighten origins in prod
limiter.init_app(app)

# Mongo
app.config["MONGO_URI"] = os.environ.get("MONGO_URI", "mongodb://localhost:27017/alzheimers_db")
mongo = PyMongo(app)
bcrypt = Bcrypt(app)

# Predictions collection and indexes
predictions_col = mongo.db.predictions
try:
    predictions_col.create_index([("user_id", ASCENDING), ("date", DESCENDING)])
    predictions_col.create_index([("user_id", DESCENDING), ("_id", DESCENDING)])
    # Ensure uniqueness only for documents where predictionId is a non-null string
    predictions_col.create_index(
        [("predictionId", ASCENDING)],
        unique=True,
        partialFilterExpression={"predictionId": {"$type": "string"}}
    )
except Exception as e:
    logger.warning(f"[WARN] Index creation skipped/failed: {e}")

# ===== JWT helpers =====
JWT_SECRET = os.environ.get('JWT_SECRET', 'supersecretkey')
JWT_EXP_DELTA_SECONDS = 18000

def generate_jwt(user_id):
    payload = {
        'user_id': str(user_id),
        'exp': datetime.utcnow() + timedelta(seconds=JWT_EXP_DELTA_SECONDS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm='HS256')

def jwt_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', None)
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'message': 'Token is missing or invalid'}), 401
        token = auth_header.split(' ')[1]
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            request.user_id = payload['user_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token is invalid'}), 401
        return f(*args, **kwargs)
    return decorated

# ===== Models: Logistic (numeric) =====
logistic_model = joblib.load(os.path.join(BASE_DIR, 'logistic_model_20250906_192440.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler_20250906_192440.pkl'))


FEATURE_NAMES = ['MR Delay', 'CDR', 'M/F', 'Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']

# Constrained counterfactuals (adjust to your data)
FEATURE_CONSTRAINTS = {
    'CDR': {'type': 'categorical', 'values': [0.0, 0.5, 1.0, 2.0, 3.0]},
    'M/F': {'type': 'categorical', 'values': [0.0, 1.0]},  
    'Age': {'type': 'range', 'min': 50.0, 'max': 100.0},
    'MR Delay': {'type': 'range', 'min': 0.0, 'max': 5000.0},  
    'EDUC': {'type': 'range', 'min': 0.0, 'max': 25.0},
    'SES': {'type': 'range', 'min': 1.0, 'max': 5.0},
    'MMSE': {'type': 'range', 'min': 0.0, 'max': 30.0},
    'eTIV': {'type': 'range', 'min': 1000.0, 'max': 2500.0},  
    'nWBV': {'type': 'range', 'min': 0.6, 'max': 0.9},
    'ASF': {'type': 'range', 'min': 0.7, 'max': 1.4},
}

# ===== PyTorch MRI model =====
class AlzheimersResNet50(torch.nn.Module):
    def __init__(self, num_classes=4, dropout_rate=0.5):
        super(AlzheimersResNet50, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Sequential(
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(num_features, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.resnet(x)

# Load .h5 metadata & weights into PyTorch model
h5_model_path = os.path.join(BASE_DIR, 'alzheimers_BEST_model_20250814_161222.h5')
with h5py.File(h5_model_path, 'r') as h5f:
    # Metadata
    try:
        val_acc = float(h5f['metadata'].attrs.get('validation_accuracy', np.nan))
        test_acc = float(h5f['metadata'].attrs.get('test_accuracy', np.nan))
    except Exception:
        val_acc, test_acc = np.nan, np.nan

    # Classes
    classes = []
    if 'metadata' in h5f and 'classes' in h5f['metadata']:
        classes_bytes = h5f['metadata']['classes'][:]
        classes = [cls.decode('utf-8') if isinstance(cls, (bytes, bytearray)) else str(cls) for cls in classes_bytes]
    if not classes:
        classes = ['class_0', 'class_1', 'class_2', 'class_3']

    logger.info(f"Loaded classes: {classes}")

    # Preprocessing
    mean = h5f['preprocessing'].attrs.get('mean', [0.485, 0.456, 0.406])
    std = h5f['preprocessing'].attrs.get('std', [0.229, 0.224, 0.225])
    input_size_val = h5f['preprocessing'].attrs.get('input_size', 224)
    if isinstance(input_size_val, (np.ndarray, list, tuple)):
        input_size = int(input_size_val[0])
    else:
        input_size = int(input_size_val)
        
# Normalize mean/std to length-3 list
def _to_rgb_triplet(x, default):
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = list(map(float, x))
        if len(arr) == 3:
            return arr
        if len(arr) == 1:
            return [arr[0]] * 3
    try:
        val = float(x)
        return [val, val, val]
    except Exception:
        return default

mean = _to_rgb_triplet(mean, [0.485, 0.456, 0.406])
std = _to_rgb_triplet(std, [0.229, 0.224, 0.225])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pt_model = AlzheimersResNet50(num_classes=len(classes), dropout_rate=0.5).to(device)

# Load parameters from h5 (expects compatible keys/shapes)
with h5py.File(h5_model_path, 'r') as h5f:
    state_dict = {}
    if 'model_parameters' in h5f:
        for param_name in h5f['model_parameters']:
            dataset = h5f['model_parameters'][param_name]
            param_data = dataset[()] if dataset.shape == () else dataset[:]
            state_dict[param_name] = torch.from_numpy(np.array(param_data))
    else:
        logger.warning("HDF5 missing 'model_parameters' group; using ImageNet-initialized head.")

missing, unexpected = pt_model.load_state_dict(state_dict, strict=False)
if missing or unexpected:
    logger.warning(f"State dict mismatch. Missing: {missing}, Unexpected: {unexpected}")
pt_model.eval()

test_transform = transforms.Compose([
    transforms.Resize((int(input_size), int(input_size))),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# ===== Marshmallow Schemas =====
from marshmallow import Schema, fields, ValidationError

class RegisterSchema(Schema):
    username = fields.Str(required=True)
    password = fields.Str(required=True)

class LoginSchema(Schema):
    username = fields.Str(required=True)
    password = fields.Str(required=True)

class NumericPredictSchema(Schema):
    input = fields.List(fields.Float(), required=True, validate=lambda l: len(l) == 10)

register_schema = RegisterSchema()
login_schema = LoginSchema()
numeric_predict_schema = NumericPredictSchema()

# ===== Auth routes =====
@app.route('/register', methods=['POST'])
@limiter.limit("5 per minute")
def register():
    try:
        data = request.get_json()
        try:
            data = register_schema.load(data)
        except ValidationError as ve:
            return jsonify({'message': 'Invalid input', 'errors': ve.messages}), 400

        username = data['username']
        password = data['password']
        if mongo.db.users.find_one({'username': username}):
            logger.info(f"Registration attempt for existing user: {username}")
            return jsonify({'success': False, 'message': 'User already exists'}), 409

        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        user_id = mongo.db.users.insert_one({'username': username, 'password': hashed_pw}).inserted_id
        logger.info(f"User registered: {username}")
        return jsonify({'success': True, 'message': 'Registration successful', 'data': {'user_id': str(user_id)}}), 201
    except Exception as e:
        logger.error(f"[ERROR] /register: {e}")
        return jsonify({'success': False, 'message': 'Server error'}), 500

@app.route('/login', methods=['POST'])
@limiter.limit("10 per minute")
def login():
    try:
        data = request.get_json()
        try:
            data = login_schema.load(data)
        except ValidationError as ve:
            logger.info(f"Login validation error: {ve.messages}")
            return jsonify({'success': False, 'message': 'Invalid input', 'errors': ve.messages}), 400

        username = data['username']
        password = data['password']
        user = mongo.db.users.find_one({'username': username})
        if user and bcrypt.check_password_hash(user['password'], password):
            token = generate_jwt(user['_id'])
            logger.info(f"User logged in: {username}")
            return jsonify({'success': True, 'message': 'Login successful', 'data': {'token': token, 'user_id': str(user['_id'])}}), 200
        else:
            logger.info(f"Failed login attempt for user: {username}")
            return jsonify({'success': False, 'message': 'Invalid username or password'}), 401
    except Exception as e:
        logger.error(f"[ERROR] /login: {e}")
        return jsonify({'success': False, 'message': 'Server error'}), 500

# ===== Numeric prediction =====
@app.route('/predict/numeric', methods=['POST'])
@limiter.limit("20 per minute")
def predict_numeric():
    try:
        data = request.get_json()
        try:
            data = numeric_predict_schema.load(data)
        except ValidationError as ve:
            logger.info(f"Numeric prediction validation error: {ve.messages}")
            return jsonify({'success': False, 'message': 'Invalid input', 'errors': ve.messages}), 400

        input_data = data.get('input')
        if not input_data:
            return jsonify({'success': False, 'message': 'Input is missing'}), 400

        if len(input_data) != len(FEATURE_NAMES):
            return jsonify({'success': False, 'message': f'Expected {len(FEATURE_NAMES)} features'}), 400

        input_array = np.array(input_data, dtype=float).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        # Prediction
        prediction = int(logistic_model.predict(scaled_input)[0])
        probabilities = logistic_model.predict_proba(scaled_input)[0].tolist()

        # Coefs / importance
        coefs = logistic_model.coef_[0]
        feature_importance = [
            {'name': name, 'weight': float(w)}
            for name, w in zip(FEATURE_NAMES, coefs)
        ]
        feature_importance = sorted(feature_importance, key=lambda x: abs(x['weight']), reverse=True)

        # Internals
        means = scaler.mean_
        stds = scaler.scale_
        z_scores_np = (input_array[0] - means) / stds
        z_scores_list = [{'name': n, 'z': float(z)} for n, z in zip(FEATURE_NAMES, z_scores_np)]
        feature_coefficients = [
            {'name': n, 'coef': float(c), 'sign': 'positive' if c > 0 else 'negative' if c < 0 else 'neutral'}
            for n, c in zip(FEATURE_NAMES, coefs)
        ]

        # Decision margin (distance from boundary)
        logit_margin = float(logistic_model.decision_function(scaled_input)[0])

        # Contributions
        contributions = (z_scores_np * coefs).astype(float)
        total_contribution = float(np.sum(np.abs(contributions))) or 1.0
        feature_contributions = [
            {'name': n, 'contribution': float(c), 'percent_share': float(abs(c)) / total_contribution * 100.0}
            for n, c in zip(FEATURE_NAMES, contributions)
        ]

        # Debug logs
        logger.info(f"Contributions: {contributions}")
        logger.info(f"Feature Contributions: {feature_contributions}")

       
        feature_percentiles = [
            {'name': n, 'percentile': float(norm.cdf(val, loc=mu, scale=sd) * 100.0) if sd > 0 else 50.0}
            for n, val, mu, sd in zip(FEATURE_NAMES, input_array[0], means, stds)
        ]

        # SHAP placeholder

        # Wrap the logistic_model's predict_proba method for SHAP compatibility
        def model_predict(X):
            return logistic_model.predict_proba(X)

    

        probability_threshold = 0.5
        calibration_note = "Model is well-calibrated in validation."
        low_confidence_flag = abs(logit_margin) < 1.0  # tune as needed

        # Commentary
        feature_commentary = []
        for i, name in enumerate(FEATURE_NAMES):
            z = z_scores_np[i]
            if abs(z) < 0.5:
                comment = f"{name} is close to average."
            elif z > 0.5:
                comment = f"{name} is higher than average."
            else:
                comment = f"{name} is lower than average."
            feature_commentary.append({'name': name, 'comment': comment})

        # Add timestamp
        now = datetime.utcnow()

        # Optional: tie to user (JWT in Authorization header)
        user_id = None
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header.split(' ', 1)[1]
            try:
                payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
                user_id = payload.get('user_id')
            except Exception:
                user_id = None

        # Determine class names from model classes (assume binary: 0=Normal, 1=Dementia if applicable)
        try:
            classes_np = getattr(logistic_model, 'classes_', np.array([0, 1]))
            classes_list = classes_np.tolist() if hasattr(classes_np, 'tolist') else list(classes_np)
        except Exception:
            classes_list = [0, 1]
        name_map = {0: 'Normal', 1: 'Dementia'}
        classNames = [name_map.get(int(c), f'class_{c}') for c in classes_list]

        # Map probs to names when lengths match; else leave empty
        prob_map = None
        try:
            if len(classNames) == len(probabilities):
                prob_map = {classNames[i]: float(probabilities[i]) for i in range(len(classNames))}
        except Exception:
            prob_map = None

        # Predicted label string and confidence
        try:
            pred_idx = int(prediction) if isinstance(prediction, (int, np.integer)) else int(np.argmax(probabilities))
        except Exception:
            pred_idx = 0
        predicted_label = classNames[pred_idx] if pred_idx < len(classNames) else str(prediction)
        try:
            conf = float(max(probabilities)) if probabilities else None
        except Exception:
            conf = None

        # Persist to Mongo if user known
        try:
            doc = {
                'user_id': str(user_id) if user_id else None,
                'input_type': 'numeric',
                'predictionId': uuid.uuid4().hex,
                'date': now,
                'prediction': predicted_label,
                'confidence': conf,
                'probabilities': prob_map,
                'probabilitiesList': probabilities,
                'classNames': classNames,
                'features': input_data,
                'result': predicted_label,
                'source': 'numeric',
                'feature_importance': feature_importance,
                'feature_coefficients': feature_coefficients,
                'feature_means': means.tolist(),
                'feature_stds': stds.tolist(),
                'z_scores': z_scores_list,
                'logit_margin': logit_margin,
                'feature_contributions': feature_contributions,
                'feature_percentiles': feature_percentiles,
                'probability_threshold': probability_threshold,
                'calibration_note': calibration_note,
                'low_confidence_flag': low_confidence_flag,
                'feature_commentary': feature_commentary
            }
            # Only insert if we have a user id; keep endpoint usable without auth
            if user_id:
                predictions_col.insert_one(doc)
        except Exception as e:
            logger.warning(f"[WARN] Numeric prediction not persisted: {e}")

        # Response
        return jsonify({'success': True, 'message': 'Prediction successful', 'data': {
            'prediction': predicted_label,
            'probabilities': probabilities,
            'probabilitiesMap': prob_map,
            'classNames': classNames,
            'timestamp': now.isoformat() + 'Z',
            'features': input_data,

            'feature_importance': feature_importance,
            'feature_coefficients': feature_coefficients,
            'feature_means': means.tolist(),
            'feature_stds': stds.tolist(),
            'z_scores': z_scores_list,
            'logit_margin': logit_margin,
            'feature_contributions': feature_contributions,
            'feature_percentiles': feature_percentiles,
            'probability_threshold': probability_threshold,
            'calibration_note': calibration_note,
            'low_confidence_flag': low_confidence_flag,
            'feature_commentary': feature_commentary
        }}), 200

    except Exception as e:
        logger.error(f"[ERROR] /predict/numeric: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'message': 'Server error'}), 500

# ===== Image prediction =====
from werkzeug.utils import secure_filename
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/predict/image', methods=['POST'])
@limiter.limit("10 per minute")
def predict_image():
    file = request.files.get('image')
    token = request.form.get('token')
    # Fallback to Authorization header if form token not provided
    if not token:
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header.split(' ', 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        user_id = payload.get('user_id')  # stored as string in JWT
    except jwt.ExpiredSignatureError:
        return jsonify({'success': False, 'message': 'Token has expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'success': False, 'message': 'Token is invalid'}), 401
    except Exception:
        return jsonify({'success': False, 'message': 'Token is missing or invalid'}), 401

    if not file:
        logger.info("Image prediction missing file.")
        return jsonify({'success': False, 'message': 'Image is missing'}), 400

    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        return jsonify({'success': False, 'message': 'Only images are allowed: jpg, jpeg, png'}), 400

    total_len = request.content_length
    if total_len and total_len > 5 * 1024 * 1024:
        return jsonify({'success': False, 'message': 'Image is too large (max 5MB)'}), 400

    safe_name = secure_filename(file.filename)
    filename = f"{uuid.uuid4().hex}_{safe_name}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        file.save(filepath)
        image = Image.open(filepath).convert('RGB')
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        logger.error(f"[ERROR] /predict/image open/save: {e}")
        return jsonify({'success': False, 'message': f'Error opening image: {str(e)}'}), 400

    try:
        input_tensor = test_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = pt_model(input_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_idx = int(np.argmax(probs))
            confidence = float(probs[predicted_idx])
            predicted_class = classes[predicted_idx]

        # Entropy (uncertainty)
        entropy = float(-np.sum(probs * np.log(probs + 1e-8)))

        # Reliability band
        if confidence >= 0.85:
            reliability = 'High'
        elif confidence >= 0.65:
            reliability = 'Medium'
        else:
            reliability = 'Low'

        # Timestamp and prediction_id
        now = datetime.utcnow()
        prediction_id = str(uuid.uuid4())

        # Model info (from loaded metadata)
        model_info = {
            'architecture': 'ResNet50',
            'version': os.path.basename(h5_model_path),
            'validation_accuracy': val_acc,
            'test_accuracy': test_acc
        }

        # Input details
        input_details = {
            'original_filename': file.filename,
            'input_size': image.size,  # (width, height)
            'preprocessing': {
                'resize': input_size,
                'normalize_mean': mean,
                'normalize_std': std
            }
        }

        # Grad-CAM visualization
        grad_cam_url = None
        orig_url = None
        try:
            target_layer = pt_model.resnet.layer4[-1].conv3
            grad_cam = GradCAM(pt_model, target_layer)
            cam_np = grad_cam(input_tensor, class_idx=predicted_idx)
            grad_cam.remove_hooks()
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() + 1e-8)
            from PIL import Image as PILImage
            cam_img = PILImage.fromarray((cam_np * 255).astype(np.uint8)).resize(image.size, resample=PILImage.BILINEAR)
            cam_img_np = np.array(cam_img)
            orig_img_np = np.array(image.convert('RGB'))
            heatmap = cm.get_cmap('jet')(cam_img_np / 255.0)[..., :3]
            heatmap = (heatmap * 255).astype(np.uint8)
            alpha = 0.35
            overlay = (alpha * heatmap + (1 - alpha) * orig_img_np).astype(np.uint8)
            grad_cam_filename = f"gradcam_{prediction_id}.png"
            orig_filename = f"orig_{prediction_id}.png"
            PILImage.fromarray(overlay).save(os.path.join(app.config['UPLOAD_FOLDER'], grad_cam_filename))
            image.convert('RGB').save(os.path.join(app.config['UPLOAD_FOLDER'], orig_filename))
            grad_cam_url = f"/uploads/{grad_cam_filename}"
            orig_url = f"/uploads/{orig_filename}"
        except Exception as e:
            logger.error(f"[ERROR] Grad-CAM save: {e}")

        download_links = {
            'original_image': orig_url,
            'grad_cam': grad_cam_url
        }

        logger.info(f"Image prediction: {predicted_class} ({confidence:.2f}) for user {user_id}")

        # Build probabilities map and list
        prob_dict = {cls: float(probs[i]) for i, cls in enumerate(classes)}
        prob_list = [float(x) for x in probs.tolist()] if hasattr(probs, 'tolist') else [float(x) for x in probs]

        # Persist document
        doc = {
            'user_id': str(user_id),
            'input_type': 'image',
            'predictionId': prediction_id,
            'date': now,
            'prediction': predicted_class,
            'confidence': confidence,
            'entropy': entropy,
            'reliability': reliability,
            'probabilities': prob_dict,           # map for History/UI readability
            'probabilitiesList': prob_list,       # keep list for potential analytics
            'classNames': classes,
            'modelInfo': model_info,
            'inputDetails': input_details,
            'downloadLinks': download_links,
            'gradCamUrl': grad_cam_url,
            'origUrl': orig_url,
            'source': 'image'
        }
        try:
            insert_res = predictions_col.insert_one(doc)
            mongo_id = str(insert_res.inserted_id)
        except Exception as e:
            logger.error(f"[ERROR] Mongo insert failed: {e}")
            mongo_id = None

        data = {
            'mongoId': mongo_id,
            'prediction': predicted_class,
            'confidence': confidence,
            'entropy': entropy,
            'reliability': reliability,
            'probabilities': prob_list,           # list for existing result card UI
            'probabilitiesMap': prob_dict,        # extra map for convenience
            'classNames': classes,
            'timestamp': now.isoformat() + 'Z',
            'predictionId': prediction_id,
            'modelInfo': model_info,
            'inputDetails': input_details,
            'downloadLinks': download_links,
            'gradCamUrl': grad_cam_url,
            'origUrl': orig_url
        }
        
        return jsonify({
            'success': True,
            'message': 'Prediction successful',
            'data': data
        }), 200

    except Exception as e:
        logger.error(f"[ERROR] /predict/image: {e}\n{traceback.format_exc()}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'success': False, 'message': f'Error during prediction: {str(e)}'}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

# ===== GET single numeric prediction by ID =====
@app.route('/api/prediction/numeric/<prediction_id>', methods=['GET'])
@jwt_required
def get_numeric_prediction_by_id(prediction_id):
    try:
        user_id = request.user_id
        try:
            oid = ObjectId(prediction_id)
        except InvalidId:
            return jsonify({'success': False, 'message': 'Invalid ID'}), 400

        logger.info(f"[FETCH] Looking for prediction: _id={prediction_id}, user_id={user_id}, input_type=numeric")
        record = mongo.db.predictions.find_one({'_id': oid, 'user_id': str(user_id), 'input_type': 'numeric'})
        if not record:
            logger.warning(f"[FETCH] Prediction not found: _id={prediction_id}, user_id={user_id}, input_type=numeric")
            return jsonify({'success': False, 'message': 'Prediction not found'}), 404

        result = {
            'prediction': record.get('result'),
            'probabilities': record.get('probabilities'),
            'features': record.get('features'),
            'date': record.get('date').strftime('%Y-%m-%d') if record.get('date') else None,
            'feature_importance': record.get('feature_importance'),
            'modelInternals': {
                'coefficients': record.get('feature_coefficients'),
                'zScores': record.get('z_scores'),
                'logitMargin': record.get('logit_margin'),
            },
            'feature_means': record.get('feature_means'),
            'feature_stds': record.get('feature_stds'),
            'feature_contributions': record.get('feature_contributions'),
            'feature_percentiles': record.get('feature_percentiles'),
            'shapValues': record.get('shap_values'),
            'modelCalibration': {
                'probabilityThreshold': record.get('probability_threshold'),
                'calibrationNote': record.get('calibration_note'),
            },
            'lowConfidenceFlag': record.get('low_confidence_flag'),
            'feature_commentary': record.get('feature_commentary'),
            'counterfactuals': record.get('counterfactuals'),
        }
        return jsonify({'success': True, 'data': result}), 200
    except Exception as e:
        logger.error(f"[ERROR] /api/prediction/numeric/<id>: {e}")
        return jsonify({'success': False, 'message': 'Server error'}), 500

# ===== History =====
@app.route('/history', methods=['GET'])
@jwt_required
@limiter.exempt
def get_history():
    user_id = request.user_id

    # Pagination params (defensive parsing)
    limit_str = request.args.get('limit', '20')
    try:
        limit = int(limit_str)
    except (ValueError, TypeError):
        limit = 20
    limit = max(1, min(limit, 50))

    cursor = request.args.get('cursor')  # last _id seen
    filter_type = request.args.get('type')  # 'image' or 'numeric'

    # Build query
    q = {'user_id': str(user_id)}
    if filter_type in ('image', 'numeric'):
        q['input_type'] = filter_type
    if cursor:
        try:
            q['_id'] = {'$lt': ObjectId(cursor)}
        except Exception:
            # Ignore bad cursor values
            pass

    try:
        docs = list(predictions_col.find(q).sort('_id', DESCENDING).limit(limit))
        history = []
        for r in docs:
            item = {
                'input_type': r.get('input_type'),
                'result': r.get('result') or r.get('prediction'),
                'confidence': r.get('confidence'),
                'date': r.get('date').strftime('%Y-%m-%d') if r.get('date') else None,
                'gradCamUrl': r.get('gradCamUrl'),
                'origUrl': r.get('origUrl'),
                'probabilities': r.get('probabilities'),
                'classNames': r.get('classNames'),
                'downloadLinks': r.get('downloadLinks'),
                'predictionId': r.get('predictionId'),
                'features': r.get('features'),
            }
            if r.get('_id'):
                item['id'] = str(r['_id'])
            history.append(item)

        next_cursor = str(docs[-1]['_id']) if docs else None

        logger.info(f"History fetched for user {user_id}. count={len(history)}")
        return jsonify({
            'success': True,
            'message': 'History fetched',
            'data': {
                'history': history,
                'nextCursor': next_cursor
            }
        }), 200
    except Exception as e:
        logger.error(f"[ERROR] /history: {e}\n{traceback.format_exc()}")
        return jsonify({'success': False, 'message': 'Server error'}), 500

# ===== Grad-CAM class =====
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_backward_hook(backward_hook))

    def __call__(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        loss = output[0, class_idx]
        self.model.zero_grad()
        loss.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().detach().numpy()
        return cam

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

# ===== File serving =====
@app.route('/uploads/<path:filename>')
@limiter.exempt
def serve_uploads(filename):
    return send_from_directory('uploads', filename)

# ===== Entrypoint =====
if __name__ == "__main__":
    app.run(debug=True)
