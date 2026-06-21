# dsp_algo.py
import numpy as np
from scipy.signal import butter, sosfilt, detrend
from scipy.ndimage import gaussian_filter1d
import os
import config

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    class VitalCNN(nn.Module):
        def __init__(self, input_len=300):
            super(VitalCNN, self).__init__()
            self.conv1 = nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7)
            self.bn1 = nn.BatchNorm1d(32)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3)
            self.bn2 = nn.BatchNorm1d(64)
            self.flatten = nn.Flatten()

            self.fc_input_dim = self._get_conv_output(input_len)
            self.fc1 = nn.Linear(self.fc_input_dim, 128)
            self.fc2 = nn.Linear(128, 2)

        def _get_conv_output(self, length):
            with torch.no_grad():
                x = torch.zeros(1, 1, length)
                x = self.pool(self.relu(self.bn1(self.conv1(x))))
                x = self.pool(self.relu(self.bn2(self.conv2(x))))
                return x.numel()

        def forward(self, x):
            x = self.pool(self.relu(self.bn1(self.conv1(x))))
            x = self.pool(self.relu(self.bn2(self.conv2(x))))
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            return self.fc2(x)

class SignalProcessor:
    def __init__(self):
        self.fs = config.FS
        self.sos_resp = butter(6, [config.BREATH_LOW, config.BREATH_HIGH], btype='bandpass', fs=self.fs, output='sos')
        self.sos_heart = butter(6, [config.HEART_LOW, config.HEART_HIGH], btype='bandpass', fs=self.fs, output='sos')

        self.use_cnn = getattr(config, 'USE_CNN_ENGINE', False) and TORCH_AVAILABLE
        self.cnn_model = None
        if self.use_cnn: self._init_cnn()

    def _init_cnn(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.cnn_model = VitalCNN(input_len=config.BUFFER_SIZE).to(self.device)
        model_path = config.CNN_MODEL_PATH
        if os.path.exists(model_path):
            self.cnn_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.cnn_model.eval()
            print(f"CNN 模型加载成功 (设备: {self.device})")
        else:
            self.use_cnn = False
            print("未找到权重文件，CNN 已禁用")

    def process_phase_to_displacement(self, raw_iq_data):
        data_arr = np.array(raw_iq_data)
        if len(data_arr) < 10: return np.zeros(config.BUFFER_SIZE)

        I = data_arr[:, 1].astype(float)
        Q = data_arr[:, 0].astype(float)

        # 使用局部滑动平均提取 I/Q 的直流偏置 (窗口约1秒)
        # 这能防止体动导致的复数圆心偏移，让后续解算的相位不再出现巨大台阶
        window = int(self.fs * 1.0)
        if window % 2 == 0: window += 1  # 确保窗口为奇数

        #估算直流分量（低通滤波）：np.convolve是 NumPy 的卷积函数。
        #np.ones(window) / window创建了一个长度为 window 的全 1 向量并进行了归一化。这本质上是一个滑动平均滤波器
        #对同相分量 I 和正交分量 Q 进行滑动平均，高频的交流信号（有用信号）被滤除，剩下的就是随时间缓慢变化的直流基准
        I_dc = np.convolve(I, np.ones(window) / window, mode='same')
        Q_dc = np.convolve(Q, np.ones(window) / window, mode='same')

        z = (I - I_dc) + 1j * (Q - Q_dc)#去除偏置并合成复信号

        delta_phase = np.angle(z[1:] * np.conj(z[:-1])) #计算相位差
        recovered_phase = np.concatenate([[0], np.cumsum(delta_phase)]) #相位重构
        smoothed_phase = gaussian_filter1d(recovered_phase, sigma=1.2) #平滑处理

        displacement = (3.896 * smoothed_phase) / (4 * np.pi) #物理量转换，将电信号角度转换为物理距离

        #严格保留论文里的“三阶多项式去趋势”：从位移数据中移除由于设备漂移、环境缓慢变化或受测物（如人体）轻微位移产生的低频非线性趋势
        x = np.arange(len(displacement)) #建立时间/索引轴：创建一个
        poly = np.polyfit(x, displacement, 3) #多项式拟合：使用最小二乘法将位移数据拟合为一个 3 次多项式
        return displacement - np.polyval(poly, x) #去除趋势：根据拟合出的系数计算出每一时刻对应的“趋势值”，用原始信号减去这个趋势

    def apply_filter_and_smooth(self, data, signal_type="resp"):
        from scipy.signal import sosfilt, sosfilt_zi

        sos = self.sos_resp if signal_type == "resp" else self.sos_heart #动态选择滤波器

        #强制减去均值，防止直流偏置冲毁滤波器
        #IIR 滤波器对直流分量非常敏感。如果信号带有一个很大的常数偏移，进入滤波器的一瞬间会产生巨大冲击，导致输出信号的前几秒完全不可用。
        data_centered = data - np.mean(data)
        #计算滤波器的稳态初始条件 (完美契合论文差分方程)
        zi = sosfilt_zi(sos) * data_centered[0] #sosfilt_zi计算滤波器的阶跃响应稳态值；* data_centered[0]将滤波器的内部寄存器（延迟链）预热到与信号第一个采样点对齐的状态
        #带初始状态进行滤波，彻底消灭前 2 秒的瞬态扭曲
        filtered, _ = sosfilt(sos, data_centered, zi=zi)

        #幅度归一化：将滤波后的信号进行标准差归一化。
        #由于不同人的呼吸/心跳强度（幅度）不同，归一化后可以将信号缩放到统一的量级，方便后续进行峰值检测或特征提取
        std = np.std(filtered)
        if std > 1e-5: filtered /= std
        # 呼吸信号的二次平滑
        # 呼吸频率较低（通常 0.1–0.5Hz），波形相对圆润；心跳频率较高且波形尖锐，不需要这种平滑，否则会把心跳特征磨平
        if signal_type == "resp":#如果是呼吸信号，额外加了一个窗口大小为 5 的滑动平均滤波
            return np.convolve(filtered, np.ones(5) / 5, mode='same')
        return filtered

    def estimate_bpm(self, sig, freq_range, is_heart=False, resp_bpm=None):
        N = len(sig)
        if np.std(sig) < 1e-6:
            return 0.0
        #信号加窗与 FFT 变换
        # 加窗，将原始信号 sig 与一个两端平滑过渡到 0 的钟形曲线（汉宁窗）逐点相乘，解决频谱泄露
        # 直接做 FFT 会导致边缘的强行截断，在频域上产生很多虚假的“毛刺”。加窗能让信号首尾完美衔接，让真正的心跳/呼吸峰值更加凸显
        windowed = sig * np.hanning(N)
        #np.fft.rfft：做实数快速傅里叶变换，位移信号是纯实数，它的频谱是对称的，只计算返回正频率部分，节省内存和计算时间
        #np.abs：FFT 算出来的结果是复数（包含幅度和相位）。套上abs是为了求模，提取出每个频点的真实能量/幅值
        mag = np.abs(np.fft.rfft(windowed))
        #生成物理频率坐标轴
        #它负责生成与上一行能量mag一一对应的X轴坐标，利用了信号长度 N 和雷达的采样率 self.fs，把原本抽象的数组索引，转换成了我们可以直接读取的物理频率
        freqs = np.fft.rfftfreq(N, 1 / self.fs)
        #频段截取
        idx = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
        if len(idx) == 0: return 0.0
        #提取幅值并深度拷贝
        v_mag = mag[idx].copy()
        #提取对应物理频率
        v_freq = freqs[idx]
        #呼吸谐波抑制
        #胸腔由于呼吸产生的位移通常在 1mm 到 12mm，而心跳微动只有 0.1mm 到 0.5mm。呼吸信号过于强大，它的二次、三次谐波很容易落入心跳的频段
        #解决方案：已经算出了呼吸频率 resp_f，这段代码会在心跳频谱中，精准地把呼吸频率的 1倍、2倍、3倍频附近的能量强行压低（乘以 0.1）。这样就排除了假峰值的干扰，让真正的心跳峰值暴露出来
        if is_heart and resp_bpm:
            resp_f = resp_bpm / 60.0
            for h in [1, 2, 3]:
                harm_mask = np.abs(v_freq - resp_f * h) < 0.08
                v_mag[harm_mask] *= 0.1

        peak_idx_v = np.argmax(v_mag)

        # 动态门限：呼吸放宽到 1.2 倍，防止蓝波显示"检测中..."
        #最大峰值的能量必须大于当前频段平均能量的 1.5 倍（心跳）或 1.2 倍（呼吸），才认为这是一个有效生命体征
        threshold_ratio = 1.5 if is_heart else 1.2
        if v_mag[peak_idx_v] < np.mean(v_mag) * threshold_ratio:
            return 0.0
        #抛物线插值：FFT 的频率分辨率受到采样时间限制，比如窗长 10 秒，分辨率就是 0.1Hz（对应6BPM的步进），这会导致算出的心率总是一阶一阶跳变的
        p = idx[peak_idx_v]
        if 0 < p < len(mag) - 1:
            y1, y2, y3 = mag[p - 1], mag[p], mag[p + 1]#利用最大峰值 y2 以及它左右相邻的两个点 y1 和 y3，拟合出一条抛物线，并计算该抛物线的顶点。
            denom = (y1 - 2 * y2 + y3 + 1e-12) #加1e-12防止除零
            offset = 0.5 * (y1 - y3) / denom #计算峰值偏移量
            f_refined = freqs[p] + offset * (freqs[1] - freqs[0]) #将无量纲的“频点偏移量”还原为真实的物理频率 解决了FFT栅栏效应
        else:
            f_refined = freqs[p] #边界回退机制

        return f_refined * 60 #转化为呼吸频率

    def estimate_bpm_with_cnn(self, displacement):
        # 只有在系统开启了 CNN 模式，并且雷达积累的数据长度达到模型训练时设定的输入窗口大小（config.BUFFER_SIZE）时，才进行推理
        # 深度学习模型对输入维度的要求是死板的，数据不够强行输入会导致维度报错
        if not self.use_cnn or len(displacement) < config.BUFFER_SIZE:
            return 0.0, 0.0
        try:
            # 数据标准化，由于不同测试者的体型、距离雷达的远近不同，雷达测出的位移幅度差异巨大。
            # 如果不做标准化，大尺度的信号会引发梯度/激活值的剧烈波动，导致模型输出乱码
            sig = np.array(displacement, dtype=np.float32)
            sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-6)#+1e-6防止除0
            #torch.from_numpy().float()：转为 PyTorch 需要的单精度浮点张量
            #.view(1, 1, -1)：PyTorch 的 nn.Conv1d 默认要求的输入维度是 [Batch_Size, Channels, Sequence_Length]
            tensor_in = torch.from_numpy(sig).float().view(1, 1, -1).to(self.device)#批次大小为1（实时单帧推理），通道数为1（单通道距离位移信号），-1：让 PyTorch 自动推断序列长度（通常等于 BUFFER_SIZE）
            #无梯度推理
            with torch.no_grad():
                out = self.cnn_model(tensor_in)[0].cpu().numpy()#.cpu().numpy()：如果模型在 GPU 上跑，必须先拉回 CPU 内存，才能转换回 NumPy 数组供后续业务逻辑使用

            return float(out[0]), float(out[1])#多任务输出
        except Exception as e:
            print(f"CNN 错误: {e}")
            return 0.0, 0.0