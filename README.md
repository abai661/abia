# EAOO: Enhanced Animated Oat Optimization Algorithm
复现论文：10.3390/biomimetics11070485
> Enhanced Animated Oat Optimization Algorithm for Global Optimization

## 项目介绍
基于原始AOO（Animated Oat Optimization），实现改进版本EAOO：
1. 正弦混沌初始化种群
2. 非线性衰减权重因子
3. 自适应t分布变异策略

对比算法：EAOO / AOO / GWO
测试函数：Sphere(F2)、Rastrigin(F1)

## 文件说明
- EAOO.py：主程序，包含训练、指标统计、自动输出收敛对比图
- convergence_curve.png：收敛对比曲线图

## 运行环境
python >=3.8
numpy
matplotlib

## 运行方式
```bash
python EAOO.py
