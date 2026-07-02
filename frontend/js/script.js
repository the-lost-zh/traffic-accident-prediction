document.addEventListener('DOMContentLoaded', function () {
    const apiBase = window.API_BASE || 'http://localhost:8888';

    // ===== Mode switching =====
    window.switchMode = function (mode) {
        document.querySelectorAll('.mode-tab').forEach(function (btn, i) {
            btn.classList.toggle('active', (i === 0 && mode === 'tabular') || (i === 1 && mode === 'multimodal'));
        });
        document.getElementById('tabular-form').classList.toggle('active', mode === 'tabular');
        document.getElementById('multimodal-form').classList.toggle('active', mode === 'multimodal');
        document.getElementById('result-container').innerHTML = '<p>请输入参数并点击预测按钮</p>';
        document.getElementById('charts-wrapper').style.display = 'none';
    };

    // ===== Populate multimodal tabular fields clone =====
    (function () {
        var tabularForm = document.getElementById('tabular-form');
        var container = document.getElementById('mm-tabular-fields');
        var fields = tabularForm.querySelectorAll('.form-group');
        fields.forEach(function (fg) {
            container.appendChild(fg.cloneNode(true));
        });
        // Remove required on multimodal copies
        container.querySelectorAll('[required]').forEach(function (el) {
            el.removeAttribute('required');
        });
    })();

    // ===== Toggle tabular fields in multimodal form =====
    window.toggleTabularFields = function () {
        var container = document.getElementById('mm-tabular-fields');
        container.style.display = container.style.display === 'none' ? 'block' : 'none';
    };

    // ===== Image preview =====
    document.getElementById('mm-image').addEventListener('change', function (e) {
        var file = e.target.files[0];
        var preview = document.getElementById('image-preview');
        if (file) {
            var reader = new FileReader();
            reader.onload = function (ev) {
                preview.innerHTML = '<img src="' + ev.target.result + '" alt="Preview" style="max-width:300px;max-height:200px;border-radius:8px;margin-top:8px;">';
            };
            reader.readAsDataURL(file);
        } else {
            preview.innerHTML = '';
        }
    });

    // ===== Tabular form submit =====
    document.getElementById('tabular-form').addEventListener('submit', function (e) {
        e.preventDefault();
        predictTabular();
    });

    // ===== Multimodal form submit =====
    document.getElementById('multimodal-form').addEventListener('submit', function (e) {
        e.preventDefault();
        predictMultimodal();
    });

    function predictTabular() {
        var form = document.getElementById('tabular-form');
        var resultContainer = document.getElementById('result-container');
        resultContainer.innerHTML = '<div class="loading"></div><p>Predicting...</p>';

        var formData = new FormData(form);
        var data = {
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

        fetch(apiBase + '/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        })
            .then(function (r) { return r.json(); })
            .then(function (result) {
                displayResult(result);
                if (result.probabilities || result.feature_contributions) {
                    document.getElementById('charts-wrapper').style.display = 'block';
                    renderCharts(result);
                }
            })
            .catch(function (err) {
                console.error(err);
                resultContainer.innerHTML = '<div class="error">Prediction failed, please try again</div>';
            });
    }

    function predictMultimodal() {
        var resultContainer = document.getElementById('result-container');
        resultContainer.innerHTML = '<div class="loading"></div><p>Predicting with multimodal model...</p>';

        var form = document.getElementById('multimodal-form');
        var formData = new FormData(form);

        fetch(apiBase + '/api/predict/multimodal', {
            method: 'POST',
            body: formData
        })
            .then(function (r) { return r.json(); })
            .then(function (resp) {
                if (resp.error) {
                    resultContainer.innerHTML = '<div class="error">' + resp.error + '</div>';
                    return;
                }
                displayMultimodalResult(resp.results);
            })
            .catch(function (err) {
                console.error(err);
                resultContainer.innerHTML = '<div class="error">Multimodal prediction failed, please try again</div>';
            });
    }

    // ===== Display =====
    function displayResult(result) {
        var html = '<div><h3>Prediction Result</h3>' +
            '<div class="severity ' + severityClass(result.severity) + '">' +
            severityText(result.severity) + '</div>';
        if (result.probability) {
            html += '<p class="probability">Confidence: ' + (result.probability * 100).toFixed(2) + '%</p>';
        }
        html += '<p class="msg">' + (result.message || '') + '</p></div>';
        document.getElementById('result-container').innerHTML = html;
    }

    function displayMultimodalResult(results) {
        var html = '<div><h3>Multimodal Prediction Results</h3>';
        var modalities = Object.keys(results);
        modalities.forEach(function (modality) {
            var r = results[modality];
            html += '<div class="mm-result-item"><strong>' + modality.toUpperCase() + '</strong>: ' +
                severityText(r.severity) +
                ' (' + (r.probability * 100).toFixed(1) + '%)</div>';
        });
        html += '</div>';

        // Render probability chart for each modality
        if (modalities.length > 0) {
            document.getElementById('charts-wrapper').style.display = 'block';
            renderMultimodalCharts(results);
        }

        document.getElementById('result-container').innerHTML = html;
    }

    function renderMultimodalCharts(results) {
        var modalities = Object.keys(results);
        var probChartDom = document.getElementById('prob-chart');
        var probChart = echarts.init(probChartDom);

        var series = modalities.map(function (mod) {
            return {
                name: mod,
                type: 'bar',
                data: results[mod].probabilities,
                barGap: '10%'
            };
        });

        probChart.setOption({
            title: { text: 'Severity Probabilities by Modality', left: 'center' },
            tooltip: { trigger: 'axis', formatter: function (p) { return p.map(function (d) { return d.seriesName + ': ' + (d.value * 100).toFixed(2) + '%'; }).join('<br/>'); } },
            legend: { data: modalities, bottom: 0 },
            xAxis: { type: 'category', data: ['Minor(1)', 'Moderate(2)', 'Severe(3)', 'Critical(4)'] },
            yAxis: { type: 'value', max: 1 },
            series: series
        });

        var shapDom = document.getElementById('shap-chart');
        var shapChart = echarts.init(shapDom);
        shapChart.setOption({
            title: { text: 'Prediction Summary', left: 'center' },
            tooltip: {},
            xAxis: { type: 'category', data: modalities },
            yAxis: { type: 'value', max: 1 },
            series: [{
                type: 'bar',
                data: modalities.map(function (m) { return results[m].probability; }),
                itemStyle: {
                    color: function (p) {
                        var colors = ['#5470c6', '#91cc75', '#fac858', '#ee6666'];
                        return colors[p.dataIndex % colors.length];
                    }
                }
            }]
        });

        window.addEventListener('resize', function () {
            echarts.getInstanceByDom(probChartDom) && echarts.getInstanceByDom(probChartDom).resize();
            echarts.getInstanceByDom(shapDom) && echarts.getInstanceByDom(shapDom).resize();
        });
    }

    function renderCharts(result) {
        if (result.probabilities && result.probabilities.length > 0) {
            var probChartDom = document.getElementById('prob-chart');
            var probChart = echarts.init(probChartDom);
            probChart.setOption({
                title: { text: 'Severity Probabilities', left: 'center' },
                tooltip: { trigger: 'axis', formatter: '{b}: {(c * 100).toFixed(2)}%' },
                xAxis: { type: 'category', data: ['Minor(1)', 'Moderate(2)', 'Severe(3)', 'Critical(4)'] },
                yAxis: { type: 'value', max: 1 },
                series: [{
                    data: result.probabilities,
                    type: 'bar',
                    barWidth: '50%',
                    itemStyle: {
                        color: function (params) {
                            var colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545'];
                            return colors[params.dataIndex];
                        }
                    }
                }]
            });
        }

        if (result.feature_contributions && Object.keys(result.feature_contributions).length > 0) {
            var shapChartDom = document.getElementById('shap-chart');
            var shapChart = echarts.init(shapChartDom);
            var features = [];
            for (var key in result.feature_contributions) {
                features.push({ name: key, value: result.feature_contributions[key], abs: Math.abs(result.feature_contributions[key]) });
            }
            features.sort(function (a, b) { return a.abs - b.abs; });
            shapChart.setOption({
                title: { text: 'Feature Contributions (SHAP)', left: 'center' },
                tooltip: { trigger: 'axis', formatter: '{b}: {c}' },
                grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
                xAxis: { type: 'value', position: 'top', splitLine: { lineStyle: { type: 'dashed' } } },
                yAxis: { type: 'category', axisLine: { show: false }, axisLabel: { show: false }, axisTick: { show: false }, splitLine: { show: false }, data: features.map(function (f) { return f.name; }) },
                series: [{
                    name: 'Contribution',
                    type: 'bar',
                    stack: 'Total',
                    label: { show: true, position: 'inside', formatter: '{b}' },
                    data: features.map(function (f) {
                        return {
                            value: f.value,
                            itemStyle: { color: f.value >= 0 ? '#ff0055' : '#008bfb' }
                        };
                    })
                }]
            });
        }

        window.addEventListener('resize', function () {
            echarts.getInstanceByDom(document.getElementById('prob-chart')) && echarts.getInstanceByDom(document.getElementById('prob-chart')).resize();
            echarts.getInstanceByDom(document.getElementById('shap-chart')) && echarts.getInstanceByDom(document.getElementById('shap-chart')).resize();
        });
    }

    // ===== Helpers =====
    function severityText(severity) {
        var map = { 0: 'Minor (Level 1)', 1: 'Moderate (Level 2)', 2: 'Severe (Level 3)', 3: 'Critical (Level 4)' };
        return map[severity] || 'Unknown';
    }

    function severityClass(severity) {
        var map = { 0: 'level-1', 1: 'level-2', 2: 'level-3', 3: 'level-4' };
        return map[severity] || 'level-1';
    }

    // ===== Input validation =====
    var inputs = document.querySelectorAll('#tabular-form input, #tabular-form select');
    inputs.forEach(function (input) {
        input.addEventListener('blur', function () {
            this.style.borderColor = this.required && !this.value ? '#dc3545' : '#ddd';
        });
    });

    ['humidity', 'distance', 'visibility', 'wind_speed'].forEach(function (id) {
        var el = document.getElementById(id);
        if (el) {
            el.addEventListener('input', function () {
                if (this.value < 0) this.value = 0;
                if (id === 'humidity' && this.value > 100) this.value = 100;
            });
        }
    });
});
