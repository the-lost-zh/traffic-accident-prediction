document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        predictSeverity();
    });
    
    function predictSeverity() {
        // 显示加载状态
        resultContainer.innerHTML = '<div class="loading"></div><p>正在预测...</p>';
        
        // 收集表单数据
        const formData = new FormData(form);
        const data = {
            start_lng: parseFloat(formData.get('start_lng')),
            start_lat: parseFloat(formData.get('start_lat')),
            distance: parseFloat(formData.get('distance')),
            junction: parseInt(formData.get('junction')),
            weather_condition: parseInt(formData.get('weather_condition')),
            temperature: parseFloat(formData.get('temperature')),
            humidity: parseInt(formData.get('humidity')),
            pressure: parseFloat(formData.get('pressure')),
            visibility: parseFloat(formData.get('visibility')),
            wind_speed: parseFloat(formData.get('wind_speed'))
        };
        
        // 发送API请求
        fetch('http://localhost:8888/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('API请求失败');
            }
            return response.json();
        })
        .then(result => {
            displayResult(result);
        })
        .catch(error => {
            console.error('错误:', error);
            resultContainer.innerHTML = '<div class="error">预测失败，请稍后重试</div>';
        });
    }
    
    function displayResult(result) {
        const severity = result.severity;
        const probability = result.probability || null;
        
        let severityText, severityClass;
        
        switch(severity) {
            case 0:
                severityText = '轻度 (Level 1)';
                severityClass = 'level-1';
                break;
            case 1:
                severityText = '中度 (Level 2)';
                severityClass = 'level-2';
                break;
            case 2:
                severityText = '重度 (Level 3)';
                severityClass = 'level-3';
                break;
            case 3:
                severityText = '极重度 (Level 4)';
                severityClass = 'level-4';
                break;
            default:
                severityText = '未知';
                severityClass = 'level-1';
        }
        
        let html = `
            <div>
                <h3>预测结果</h3>
                <div class="severity ${severityClass}">
                    ${severityText}
                </div>
        `;
        
        if (probability) {
            html += `<p class="probability">预测概率: ${(probability * 100).toFixed(2)}%</p>`;
        }
        
        html += '</div>';
        
        resultContainer.innerHTML = html;
        
        // Render charts if data is present
        if (result.probabilities || result.feature_contributions) {
            document.getElementById('charts-wrapper').style.display = 'block';
            renderCharts(result);
        } else {
            document.getElementById('charts-wrapper').style.display = 'none';
        }
    }
    
    function renderCharts(result) {
        // 1. 概率分布柱状图
        if (result.probabilities && result.probabilities.length > 0) {
            const probChartDom = document.getElementById('prob-chart');
            const probChart = echarts.init(probChartDom);
            const probOption = {
                title: { text: '各损伤等级预测概率', left: 'center' },
                tooltip: { trigger: 'axis', formatter: '{b}: {(c * 100).toFixed(2)}%' },
                xAxis: { type: 'category', data: ['轻度(1)', '中度(2)', '重度(3)', '极重度(4)'] },
                yAxis: { type: 'value', max: 1 },
                series: [{
                    data: result.probabilities,
                    type: 'bar',
                    barWidth: '50%',
                    itemStyle: {
                        color: function(params) {
                            const colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545'];
                            return colors[params.dataIndex];
                        }
                    }
                }]
            };
            probChart.setOption(probOption);
        }
        
        // 2. 特征贡献度条形图 (SHAP)
        if (result.feature_contributions && Object.keys(result.feature_contributions).length > 0) {
            const shapChartDom = document.getElementById('shap-chart');
            const shapChart = echarts.init(shapChartDom);
            
            // 转换为数组并按贡献绝对值排序
            const features = [];
            for (const [key, val] of Object.entries(result.feature_contributions)) {
                features.push({ name: key, value: val, abs: Math.abs(val) });
            }
            features.sort((a, b) => a.abs - b.abs); // 升序，以便在横向条形图中显示最重要的在上面
            
            const shapOption = {
                title: { text: '局部特征贡献度 (SHAP)', left: 'center' },
                tooltip: { trigger: 'axis', formatter: '{b}: {c}' },
                grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
                xAxis: { type: 'value', position: 'top', splitLine: { lineStyle: { type: 'dashed' } } },
                yAxis: { type: 'category', axisLine: { show: false }, axisLabel: { show: false }, axisTick: { show: false }, splitLine: { show: false }, data: features.map(f => f.name) },
                series: [{
                    name: '贡献度',
                    type: 'bar',
                    stack: 'Total',
                    label: { show: true, position: 'inside', formatter: '{b}' },
                    data: features.map(f => {
                        return {
                            value: f.value,
                            itemStyle: { color: f.value >= 0 ? '#ff0055' : '#008bfb' }
                        };
                    })
                }]
            };
            shapChart.setOption(shapOption);
        }
        
        // 处理窗口缩放重绘
        window.addEventListener('resize', function() {
            echarts.getInstanceByDom(document.getElementById('prob-chart'))?.resize();
            echarts.getInstanceByDom(document.getElementById('shap-chart'))?.resize();
        });
    }
    
    // 添加表单验证
    const inputs = form.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('blur', function() {
            if (this.required && !this.value) {
                this.style.borderColor = '#dc3545';
            } else {
                this.style.borderColor = '#ddd';
            }
        });
    });
    
    // 添加输入限制
    document.getElementById('humidity').addEventListener('input', function() {
        if (this.value < 0) this.value = 0;
        if (this.value > 100) this.value = 100;
    });
    
    document.getElementById('distance').addEventListener('input', function() {
        if (this.value < 0) this.value = 0;
    });
    
    document.getElementById('visibility').addEventListener('input', function() {
        if (this.value < 0) this.value = 0;
    });
    
    document.getElementById('wind_speed').addEventListener('input', function() {
        if (this.value < 0) this.value = 0;
    });
});
