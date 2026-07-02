from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import tempfile
from pathlib import Path

_src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(_src_dir)

from agent import PredictiveAgent

app = Flask(__name__)
CORS(app)

agent = PredictiveAgent(model_dir='models')
_multimodal_agent = None
_model_type = os.environ.get('MODEL_TYPE', 'fttransformer')


def _ensure_agent():
    if not agent.is_loaded:
        model_dir = os.environ.get('MODEL_DIR', 'models')
        agent.model_dir = model_dir
        agent.load(model_type=_model_type)


def _ensure_multimodal_agent():
    global _multimodal_agent
    if _multimodal_agent is not None:
        return _multimodal_agent

    run_dir = os.environ.get('MULTIMODAL_RUN_DIR', 'outputs/multimodal_runs/multimodal_unpaired')
    if not os.path.isdir(run_dir):
        return None

    try:
        from multimodal_agent import MultimodalPredictiveAgent
        _multimodal_agent = MultimodalPredictiveAgent(run_dir=run_dir)
        if _multimodal_agent.load():
            return _multimodal_agent
    except Exception as exc:
        print(f"Multimodal agent init failed: {exc}")
    _multimodal_agent = None
    return None


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        required_fields = [
            'start_lng', 'start_lat', 'distance', 'junction',
            'weather_condition', 'temperature', 'humidity',
            'pressure', 'visibility', 'wind_speed'
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        mapped_data = {
            'Start_Lng': data['start_lng'],
            'Start_Lat': data['start_lat'],
            'Distance(mi)': data['distance'],
            'Junction': bool(data['junction']),
            'Weather_Condition': data['weather_condition'],
            'Temperature(F)': data['temperature'],
            'Humidity(%)': data['humidity'],
            'Pressure(in)': data['pressure'],
            'Visibility(mi)': data['visibility'],
            'Wind_Speed(mph)': data['wind_speed']
        }

        _ensure_agent()
        if not agent.is_loaded:
            return jsonify({
                'error': 'Model not loaded. Train a model first or set MODEL_DIR to a directory '
                         'containing preprocessor.pkl and a model checkpoint.'
            }), 503

        try:
            result = agent.predict_with_explanation(mapped_data)
            result['message'] = 'Prediction success'
            return jsonify(result), 200
        except Exception as exc:
            return jsonify({'error': f'Prediction failed: {str(exc)}'}), 500

    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/predict/multimodal', methods=['POST'])
def predict_multimodal():
    try:
        m_agent = _ensure_multimodal_agent()
        if m_agent is None:
            return jsonify({'error': 'Multimodal model not loaded. Set MULTIMODAL_RUN_DIR or train first.'}), 503

        # Tabular fields (same as single-modal API)
        tabular = None
        if request.form:
            tabular_raw = {}
            tabular_fields = ['start_lng', 'start_lat', 'distance', 'junction',
                              'weather_condition', 'temperature', 'humidity',
                              'pressure', 'visibility', 'wind_speed']
            for field in tabular_fields:
                if field in request.form:
                    val = request.form[field]
                    try:
                        tabular_raw[field] = float(val)
                    except ValueError:
                        tabular_raw[field] = val

            if any(f in tabular_raw for f in tabular_fields):
                tabular = {
                    'Start_Lng': tabular_raw.get('start_lng', 0),
                    'Start_Lat': tabular_raw.get('start_lat', 0),
                    'Distance(mi)': tabular_raw.get('distance', 0),
                    'Junction': bool(int(tabular_raw.get('junction', 0))),
                    'Weather_Condition': tabular_raw.get('weather_condition', ''),
                    'Temperature(F)': tabular_raw.get('temperature', 0),
                    'Humidity(%)': tabular_raw.get('humidity', 0),
                    'Pressure(in)': tabular_raw.get('pressure', 0),
                    'Visibility(mi)': tabular_raw.get('visibility', 0),
                    'Wind_Speed(mph)': tabular_raw.get('wind_speed', 0),
                }

        elif request.json:
            tabular = request.json.get('tabular', None)

        # Text input
        text = None
        if request.form and 'text' in request.form:
            text = request.form['text'].strip()
        elif request.json and request.json.get('text'):
            text = request.json['text'].strip()

        # Image upload
        image_path = None
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file.filename:
                temp_dir = Path(tempfile.gettempdir()) / 'multimodal_api'
                temp_dir.mkdir(parents=True, exist_ok=True)
                suffix = Path(image_file.filename).suffix or '.png'
                image_path = str(temp_dir / f"upload_{os.urandom(8).hex()}{suffix}")
                image_file.save(image_path)

        # At least one modality must be provided
        if tabular is None and text is None and image_path is None:
            return jsonify({'error': 'Provide at least one modality: tabular data, text, or image.'}), 400

        results = m_agent.predict(tabular=tabular, text=text, image_path=image_path)

        # Clean up temp image
        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except OSError:
                pass

        return jsonify({'results': results, 'message': 'Multimodal prediction success'}), 200

    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    _ensure_agent()
    status = 'healthy' if agent.is_loaded else 'degraded'
    m_agent = _ensure_multimodal_agent()
    multimodal_available = m_agent is not None and m_agent.is_loaded

    return jsonify({
        'status': status,
        'message': f'API running. Tabular: {"loaded" if agent.is_loaded else "not loaded"}, '
                   f'Multimodal: {"available" if multimodal_available else "unavailable"}'
    }), 200


if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', '0').lower() in ('1', 'true', 'yes')
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', '8888'))
    print('Starting API server...')
    print(f'Tabular model: {_model_type}')
    print(f'Multimodal: {"enabled" if os.environ.get("MULTIMODAL_RUN_DIR") else "set MULTIMODAL_RUN_DIR to enable"}')
    print(f'API running at http://{host}:{port}')
    app.run(host=host, port=port, debug=debug)
