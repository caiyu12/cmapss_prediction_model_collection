import numpy as np
import matplotlib.pyplot as plt

# 参数设置
Fs = 100  # 采样率，单位 Hz
L = 40  # 信号长度，总采样点数
f_signal = 20  # 信号频率，单位 Hz

# 生成信号
T = 1 / Fs  # 采样间隔
t = np.arange(0, L) * T  # 时间向量
x = np.sin(2 * np.pi * f_signal * t)  # 生成正弦波信号

# 计算 FFT
Y = np.fft.fft(x)
P2 = np.abs(Y / L)
P1 = P2[:L // 2 + 1]
P1[1:-1] = 2 * P1[1:-1]  # 调整为单边频谱的振幅

# 计算频率向量和谐波次数
frequencies = Fs * np.arange(0, (L // 2) + 1) / L  # 频率向量
harmonics = frequencies / f_signal  # 计算谐波次数
print(max(harmonics))

# 计算相位角（单位为弧度）
phase = np.angle(Y[:L // 2 + 1])

# 绘制实验结果图
fig, ax1 = plt.subplots()

# 绘制谐波次数 vs 相位角（左 y 轴）
ax1.plot(harmonics, phase * 180 / np.pi, 'b-s', linewidth=1.5, markerfacecolor='b', markersize=6)
ax1.set_xlabel('谐波次数 (k)')
ax1.set_ylabel('相位 (度)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True)

# 添加双横坐标 (频率 Hz) 的 x 轴
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xlabel('谐波频率 (Hz)')

# 绘制频率 vs 相位角（右 y 轴）
ax3 = ax1.twinx()
ax3.plot(frequencies, phase * 180 / np.pi, 'r-s', linewidth=1.5, markerfacecolor='r', markersize=6)
ax3.set_ylabel('相位 (度)', color='r')
ax3.tick_params(axis='y', labelcolor='r')

# 设置标题
plt.title('相频特性图')

# 显示图形
plt.show()
