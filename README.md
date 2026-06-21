本设计是基于数字信号处理和1D-CNN的毫米波雷达呼吸心率提取算法。
先对原始中频信号进行距离向快速傅里叶变换Range-FFT，以准确定位受试者胸腔所在的距离单元，然后对该距离单元随时间变化的复数基带序列进行提取。
传统反正切解包裹算法容易产生相位阶跃，本设计引入复数矢量差分算法来提取瞬时相位并使用三阶多项式去趋势，同时结合高斯平滑技术以克服躯干缓慢晃动引起的非线性基线漂移。
本设计在信号去噪阶段设计了一个6阶巴特沃斯二阶分节级联数字带通滤波器，该滤波器可以把0.1Hz至0.6Hz的呼吸特征波形和0.8Hz至2.0Hz的心跳波形进行了分离。
系统在进行核心频率的计算时，运用了加Hanning窗的FFT算法，同时采用二次抛物线插值方法来对频谱峰值进行精细修正，从而可以克服短观测窗口下存在的栅栏效应。
面对大尺度非线性体动干扰的情况，系统并行建立了一个一维卷积神经网络分支以便进行切换。

This project implements a millimeter-wave (mmWave) radar-based respiratory and heart rate extraction algorithm integrating Digital Signal Processing (DSP) and a One-Dimensional Convolutional Neural Network (1D-CNN).
Initially, a Range Fast Fourier Transform (Range-FFT) is performed on the raw Intermediate Frequency (IF) signals to accurately locate the range bin corresponding to the subject's chest cavity. The time-varying complex baseband sequence from this target range bin is then extracted.
While conventional arctan phase-unwrapping algorithms are highly susceptible to phase steps, this design introduces a Complex Vector Difference (CVD) algorithm to extract instantaneous phase. To overcome the non-linear baseline drift induced by slow body movements, a third-order polynomial detrending method is applied in combination with Gaussian smoothing techniques.
During the signal denoising phase, a 6th-order Butterworth bandpass filter utilizing a Second-Order Sections (SOS) cascaded digital architecture was designed. This filter successfully isolates the respiratory characteristic waveforms (0.1 Hz to 0.6 Hz) and the heartbeat waveforms (0.8 Hz to 2.0 Hz).
For the core frequency estimation, an FFT algorithm with a Hanning window is implemented. Concurrently, a parabolic interpolation method is utilized to fine-tune and correct the spectral peaks, effectively mitigating the picket-fence effect inherent in short observation windows.
To address scenarios involving large-scale, non-linear body movement interference, the system concurrently establishes a parallel 1D-CNN branch to enable adaptive switching.

本設計は、デジタル信号処理（DSP）および1D-CNN（一次元畳み込みニューラルネットワーク）に基づく、ミリ波レーダーを用いた呼吸・心拍数抽出アルゴリズムです。
まず、生の中間周波数（IF）信号に対して距離方向の高速フーリエ変換（Range-FFT）を実行し、被験者の胸腔が位置する距離ビン（Range Bin）を正確に特定します。その後、当該距離ビンにおける時間変動複素ベースバンド系列を抽出します。
従来の逆正接（アークタンジェント）によるフェーズアンラップ法は位相のステップ状の跳び（フェーズステップ）が生じやすいため、本設計では複素ベクトル差分（CVD）アルゴリズムを導入して瞬時位相を抽出します。また、体躯の緩慢な揺れに起因する非線形なベースライン変動（トレンド）を克服するため、3次多項式デトレンド（去趨勢）技術とガウス平滑化フィルタを組み合わせて適用しています。
信号のノイズ除去段階では、6次バタワース二次のセクション（SOS）縦続（カスケード）型デジタルバンドパスフィルタを設計しました。このフィルタにより、0.1 Hz〜0.6 Hzの呼吸特徴波形と0.8 Hz〜2.0 Hzの心拍波形を鮮明に分離します。
核心となる周波数算出においては、ハニング窓（Hanning Window）を適用したFFTアルゴリズムを運用し、同時に2次放物線補間法を用いてスペクトルピークの微修正を行うことで、短い観測ウィンドウ下で発生するバリア効果（Picket-fence Effect / 標本化歪み）を克服しています。
また、大規模な非線形体動（寝返り等）による干渉が発生したケースに対応するため、システム内に並行して1D-CNN（一次元畳み込みニューラルネットワーク）の推論ブランチを構築し、適応的な動的切り替えを可能にしています。
