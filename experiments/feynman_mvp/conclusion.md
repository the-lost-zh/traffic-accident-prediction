# 费曼最小实验结论

## 实验设置
- 样本量：30个概念，三组方法共90条输出
- 方法：A=Zero-shot，B=标准CoT，C=四阶段Feynman-CoT
- 指标：准确性（核对点命中）、通俗性综合分、Token成本

## 方法均值
- A_zero_shot: accuracy=1.0, simplicity=61.04, tokens=110.07, latency=0.0s
- B_standard_cot: accuracy=1.0, simplicity=60.58, tokens=127.87, latency=0.0s
- C_feynman_cot: accuracy=1.0, simplicity=61.04, tokens=585.27, latency=0.0s

## 主判定阈值检验（C 相对 B）
- 通俗性提升：0.76%（阈值 >= 10%）
- 准确性下降：0.0%（阈值 <= 3%）
- Token成本倍数：4.58x（阈值 <= 2.5x）

## 结论
- 未通过MVP阈值：建议先优化提示词与阶段结构，再进入消融实验。
- 注意：本轮准确性标注采用MVP轻量规则，可在下一轮替换为严格人工双人标注。