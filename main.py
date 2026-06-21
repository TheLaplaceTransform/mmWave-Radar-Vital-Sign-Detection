# main.py
import matplotlib
matplotlib.use('TkAgg') # 指定底层图形后端为 Tkinter，适合跨平台实时绘图

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
import datetime
import config
from radar_driver import RadarDriver
from dsp_algo import SignalProcessor

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS'] # 解决 Matplotlib 中文显示乱码的祖传配置
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题

driver = RadarDriver()
processor = SignalProcessor()
driver.pre_fill()

fig = plt.figure(figsize=(12, 10))
# 创建 3行1列 的网格布局，三个图表高度比例为 1:1:1
gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1])

mode_str = "1D-CNN 推理" if processor.use_cnn else "传统FFT"
fig.suptitle(f'毫米波雷达生命体征监测系统 ({mode_str})', fontsize=18, fontweight='bold', y=0.96)

ax1 = fig.add_subplot(gs[0, 0])
line_raw, = ax1.plot([], [], color='black', lw=1.2, label='原始微动位移')
ax1.set_title("阶段 1: 目标胸腔微动位移 (Raw Micro-displacement)", fontweight='bold')
ax1.set_ylabel("位移 (mm)")

ax2 = fig.add_subplot(gs[1, 0])
line_resp, = ax2.plot([], [], color='#1f77b4', lw=2, label='呼吸波形')
text_resp = ax2.text(0.02, 0.85, '呼吸率: 计算中...', transform=ax2.transAxes, color='#1f77b4', fontsize=14, fontweight='bold')
ax2.set_title("阶段 2: 呼吸特征信号 (Respiration Waveform)", fontweight='bold')
ax2.set_ylabel("幅度")

ax3 = fig.add_subplot(gs[2, 0])
line_heart, = ax3.plot([], [], color='#d62728', lw=1, label='心跳波形')
text_heart = ax3.text(0.02, 0.85, '心率: 计算中...', transform=ax3.transAxes, color='#d62728', fontsize=14, fontweight='bold')
ax3.set_title("阶段 3: 心跳微动特征 (Heartbeat Waveform)", fontweight='bold')
ax3.set_ylabel("幅度")
ax3.set_xlabel("时间窗口 (秒)")

time_axis = np.linspace(0, config.WINDOW_SECONDS, config.BUFFER_SIZE)
for ax in [ax1, ax2, ax3]:
    ax.set_xlim(0, config.WINDOW_SECONDS)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize='small')

current_disp_data = np.zeros(config.BUFFER_SIZE)

def on_key(event):
    if event.key.lower() == 's':# 生成时间戳文件名并用numpy保存为CSV
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"radar_data_{timestamp}.csv"
        np.savetxt(filename, current_disp_data, delimiter=",", header="displacement_mm")
        print(f"\n[实验记录] 数据已导出: {filename}")

fig.canvas.mpl_connect('key_press_event', on_key)

def update(frame):
    global current_disp_data
    raw_iq = driver.get_latest_data() # 拉取雷达 IQ 原始数据
    # 1. 数据对齐补齐机制
    if len(raw_iq) < config.BUFFER_SIZE:
        pad_len = config.BUFFER_SIZE - len(raw_iq)
        raw_iq_padded = np.pad(raw_iq, ((pad_len, 0), (0, 0)), 'constant', constant_values=1e-5)
    else:
        raw_iq_padded = np.array(raw_iq[-config.BUFFER_SIZE:])
    # 2. 信号处理核心，调用DSP_algo模块，将抽象的 IQ 复数信号转化为直观的物理位移，并利用滤波器（带通滤波）将呼吸和心跳这两个不同频段的信号剥离
    try:
        displacement = processor.process_phase_to_displacement(raw_iq_padded)
        current_disp_data = displacement

        resp_wave = processor.apply_filter_and_smooth(displacement, signal_type="resp")
        heart_wave = processor.apply_filter_and_smooth(displacement, signal_type="heart")

        line_raw.set_data(time_axis, displacement)
        line_resp.set_data(time_axis, resp_wave)
        line_heart.set_data(time_axis, heart_wave)
        # 3. UI 动态自适应缩放
        for ax, data, pad in zip([ax1, ax2, ax3], [displacement, resp_wave, heart_wave], [0.5, 0.2, 0.5]):
            mi, ma = np.min(data), np.max(data)
            if ma - mi > 0.001:
                margin = max((ma - mi) * 0.2, pad)
                ax.set_ylim(mi - margin, ma + pad)
        # 4. 降频计算心率/呼吸率，UI 可能每秒刷新 20 次（间隔 50ms），但心率不需要更新这么快。frame % 10 == 0 相当于降频采样，每 0.5 秒计算一次 BPM，极大节省了 CPU/GPU 算力，也防止了 UI 上的数字疯狂闪烁
        if frame % 10 == 0:
            if processor.use_cnn:
                br, hr = processor.estimate_bpm_with_cnn(displacement)
                tag = " (CNN)"
            else:
                br = processor.estimate_bpm(resp_wave, [config.BREATH_LOW, config.BREATH_HIGH])
                hr = processor.estimate_bpm(heart_wave, [config.HEART_LOW, config.HEART_HIGH], is_heart=True, resp_bpm=br)
                tag = ""

            if br > 0: text_resp.set_text(f"呼吸率: {br:.1f} BPM{tag}")
            else: text_resp.set_text("呼吸率: 检测中...")

            if hr > 0: text_heart.set_text(f"心率: {hr:.1f} BPM{tag}")
            else: text_heart.set_text("心率: 检测中...")

    except Exception as e:
        # 取消 pass，如果这里报错，直接在控制台打印出来
        print(f"UI数据处理异常: {e}")

    return line_raw, line_resp, line_heart, text_resp, text_heart

if __name__ == "__main__":
    print("毫米波雷达生命体征系统正在启动...")
    print(f"当前模式: {'[1D-CNN]' if processor.use_cnn else '[FFT频谱]'}")

    try:
        driver.start()
        ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        driver.stop()
        print("系统已安全退出。")