# radar_driver.py
import serial
import time
import threading
import struct
import numpy as np
import config

MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'


class RadarDriver:
    def __init__(self):
        self.running = False
        self.data_buffer = []
        self.lock = threading.Lock()
        self.byte_buffer = b''
        self.frame_count = 0
        self.ser_data = None
        self.ser_cli = None

    def _init_serial(self):
        try:
            self.ser_data = serial.Serial(config.SERIAL_PORT_DATA, config.BAUD_RATE_DATA, timeout=0.01)#数据串口，发送二进制数据
            self.ser_cli = serial.Serial(config.SERIAL_PORT_CLI, config.BAUD_RATE_CLI, timeout=1)#配置串口，逐行发送纯文本配置（比如 chirp 参数、帧率）
            self.ser_data.reset_input_buffer()
            self.ser_data.dtr = self.ser_data.rts = True
            return True
        except Exception as e:
            print(f"串口连接失败: {e}")
            return False

    def _send_config(self):
        print("正在握手下发配置")
        for cmd in config.RADAR_CFG_COMMANDS:
            self.ser_cli.write((cmd + '\n').encode())
            res = ""
            t_end = time.time() + 1.2
            while time.time() < t_end:
                if self.ser_cli.in_waiting:
                    res += self.ser_cli.read(self.ser_cli.in_waiting).decode(errors='ignore')
                    if "mmwDemo:/>" in res: break
            print(f"{'✅' if 'Done' in res else '⚠️'} | {cmd[:18]}...")
        time.sleep(1)

    def _parse_oob_frame(self):
        while True:
            idx = self.byte_buffer.find(MAGIC_WORD)
            if idx == -1:
                if len(self.byte_buffer) > 16384: self.byte_buffer = b''
                break

            self.byte_buffer = self.byte_buffer[idx:]
            if len(self.byte_buffer) < 40: break

            try:
                total_len = struct.unpack('<I', self.byte_buffer[12:16])[0]
                num_tlvs = struct.unpack('<I', self.byte_buffer[32:36])[0]
                if len(self.byte_buffer) < total_len: break

                payload = self.byte_buffer[40:total_len]
                self.byte_buffer = self.byte_buffer[total_len:]
                self.frame_count += 1

                cursor = 0
                tlv_info = []
                target_iq_data = None

                for _ in range(num_tlvs):
                    if cursor + 8 > len(payload): break
                    t, l = struct.unpack('<II', payload[cursor:cursor + 8])
                    v = payload[cursor + 8: cursor + 8 + l]
                    cursor += (8 + l)

                    tlv_info.append(f"Type {t} (Len:{l})")

                    # 同时兼容 Type 4 和 Type 8
                    if t == 4 or t == 8:
                        try:
                            # 1个复数点=4字节 (16bit I + 16bit Q)
                            total_complex_points = l // 4
                            num_bins = 64  # profileCfg 中定义的 ADC 采样点

                            # 动态推算虚拟天线数，彻底解决维度报错
                            num_virtual_ant = total_complex_points // num_bins

                            # 转化为 [总点数, 2] 的平铺一维复数对
                            complex_data = np.frombuffer(v, dtype=np.int16).reshape(-1, 2)

                            # 巧妙提取第 0 个虚拟天线的所有 Range Bins 的 I/Q
                            I_all = complex_data[0::num_virtual_ant, 1].astype(float)
                            Q_all = complex_data[0::num_virtual_ant, 0].astype(float)

                            mag = np.sqrt(I_all ** 2 + Q_all ** 2)

                            # 在室内微动区间 (0.5m ~ 3m) 寻峰
                            target = np.argmax(mag[5:45]) + 5
                            target_iq_data = (I_all[target], Q_all[target])

                        except Exception as e:
                            print(f"Type {t} 数据解包失败: {e}")

                # 前 10 帧强制打印内部结构，确认雷达状态
                if self.frame_count <= 10 or self.frame_count % 50 == 0:
                    print(f"[诊断] 帧 {self.frame_count} 包含包体: {tlv_info}")

                # 只要拿到了复数数据，就返回去画图
                if target_iq_data:
                    return target_iq_data

            except Exception as e:
                print(f"帧解析全局异常: {e}")
                self.byte_buffer = self.byte_buffer[8:]
                continue

            return None

    def _update_loop(self):
        while self.running:
            if self.ser_data.in_waiting > 0:
                self.byte_buffer += self.ser_data.read(self.ser_data.in_waiting)
                val = self._parse_oob_frame()
                if val:
                    with self.lock:
                        self.data_buffer.append(val)
                        if len(self.data_buffer) > config.BUFFER_SIZE: self.data_buffer.pop(0)
            else:
                time.sleep(0.005)

    def start(self):
        self.running = True
        if self._init_serial():
            self._send_config()
            threading.Thread(target=self._update_loop, daemon=True).start()

    def get_latest_data(self):
        with self.lock: return list(self.data_buffer)

    def pre_fill(self):
        self.data_buffer = [(1e-5, 1e-5)] * config.BUFFER_SIZE

    def stop(self):
        self.running = False
        if self.ser_data: self.ser_data.close()
        if self.ser_cli: self.ser_cli.close()