from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# 添加src目录到Python路径
_src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.append(_src_dir)

from agent import PredictiveAgent

app = Flask(__name__)
CORS(app)  # 允许跨域请求

agent = PredictiveAgent(model_dir='models')
_model_type = os.environ.get('MODEL_TYPE', 'fttransformer')


def _ensure_agent():
    """懒加载智能体，仅在首次请求时初始化。"""
    if not agent.is_loaded:
        model_dir = os.environ.get('MODEL_DIR', 'models')
        agent.model_dir = model_dir
        agent.load(model_type=_model_type)

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    处理预测请求
    """
    try:
        data = request.json
        
        # 验证必需参数 (保持前端约定的键一致)
        required_fields = [
            'start_lng', 'start_lat', 'distance', 'junction', 
            'weather_condition', 'temperature', 'humidity', 
            'pressure', 'visibility', 'wind_speed'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'缺少必要参数: {field}'}), 400
                
        # 将前端的驼峰或蛇形命名转为数据集对应的特征名 (如果有不一致需要在这里映射)
        # 例如数据集的列是 Start_Lng, Start_Lat, Distance(mi) 等
        mapped_data = {
            'Start_Lng': data['start_lng'],
            'Start_Lat': data['start_lat'],
            'Distance(mi)': data['distance'],
            'Junction': bool(data['junction']), # 可能前端传的是 0/1
            'Weather_Condition': data['weather_condition'],
            'Temperature(F)': data['temperature'],
            'Humidity(%)': data['humidity'],
            'Pressure(in)': data['pressure'],
            'Visibility(mi)': data['visibility'],
            'Wind_Speed(mph)': data['wind_speed']
        }
        
        _ensure_agent()

        # 尝试通过 Agent 推理（如果它已加载）
        if agent.is_loaded:
            try:
                result = agent.predict_with_explanation(mapped_data)
                result['message'] = '预测成功'
                return jsonify(result), 200
            except Exception as e:
                print(f"✗ 智能体预测失败: {str(e)}")
                # 下降到模拟逻辑
                pass
        
        # --- 模拟预测逻辑 (作为备用兜底) ---
        print("⚠ 使用模拟预测逻辑兜底")
        distance = data['distance']
        weather = float(data.get('weather_condition', 0))
        
        if distance > 1 and weather >= 2:
            severity = 3
        elif distance > 0.5 or weather >= 1:
            severity = 2
        elif distance > 0.1:
            severity = 1
        else:
            severity = 0
            
        probability = min(0.7 + (distance * 0.1), 0.95)
        
        result = {
            'severity': severity,
            'probability': probability,
            'probabilities': [0.1, 0.1, 0.1, 0.1], # 假数据
            'feature_contributions': {'Distance(mi)': 0.5},
            'message': '模拟预测（模型未加载或推理出错）'
        }
        result['probabilities'][severity] = probability
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    _ensure_agent()
    status = 'healthy' if agent.is_loaded else 'degraded'
    return jsonify({
        'status': status, 
        'message': f'API服务运行中。模型状态: {"已加载" if agent.is_loaded else "未加载"}'
    }), 200

if __name__ == '__main__':
    debug = os.environ.get('FLASK_DEBUG', '0').lower() in ('1', 'true', 'yes')
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', '8888'))
    print('启动API服务器...')
    print(f'API服务器运行在 http://{host}:{port}')
    app.run(host=host, port=port, debug=debug)

