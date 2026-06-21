import serial
import time

# 请确保这里的串口号和您的 Mac 完全一致
SERIAL_PORT_CLI = '/dev/cu.usbserial-01C842A90'
SERIAL_PORT_DATA = '/dev/cu.usbserial-01C842A91'

# 18xx 生命体征标准配置指令
RADAR_CFG_COMMANDS = [
    "sensorStop",
    "flushCfg",
    "dfeDataOutputMode 1",
    # 🚀 修复1：仅开启 1 个发射天线(TX1)和 4 个接收天线
    "channelCfg 15 1 0",
    "adcCfg 2 1",
    "adcbufCfg -1 0 0 1 0",
    "profileCfg 0 77 7 7 214.28 0 0 44 1 256 5209 0 0 30",
    # 🚀 修复2：只保留 Chirp 0，删除那个调用 TX3 的 Chirp 1
    "chirpCfg 0 0 0 0 0 0 0 1",
    # 🚀 修复3：帧结构必须对应修改为只跑 chirp 0 到 chirp 0
    "frameCfg 0 0 2 0 50 1 0",
    "guiMonitor 1 1 0 0 0 1",
    "vitalSignsCfg 1 0 0.1 0.5 0.8 2.0 40 0.1",
    "sensorStart"
]


def test_radar_boot():
    print("🔌 正在连接雷达 CLI 控制口...")
    try:
        cli_port = serial.Serial(SERIAL_PORT_CLI, 115200, timeout=0.5)

        # 先发一个回车，清理雷达开机时的乱码
        cli_port.write(b'\n')
        time.sleep(0.1)
        cli_port.reset_input_buffer()

        print("📝 开始逐条下发配置指令：\n")
        for cmd in RADAR_CFG_COMMANDS:
            cli_port.write((cmd + '\n').encode('utf-8'))
            time.sleep(0.2)  # 给雷达芯片处理的时间

            # 读取雷达的回复
            response = cli_port.read(cli_port.in_waiting).decode('utf-8', errors='ignore').strip()

            # 格式化打印：重点看有没有 Error!
            if "Error" in response or "error" in response:
                print(f"❌ 报错指令: {cmd}")
                print(f"   雷达回复: {response}\n")
            else:
                print(f"✅ 发送: {cmd}")

        cli_port.close()

    except Exception as e:
        print(f"串口打开失败: {e}")
        return

    print("\n-----------------------------------")
    print("⏳ 正在检查数据口是否有波形输出...")
    try:
        data_port = serial.Serial(SERIAL_PORT_DATA, 921600, timeout=1)
        for i in range(5):
            print(f"缓冲区积压字节数: {data_port.in_waiting}")
            time.sleep(0.5)
        data_port.close()
    except Exception as e:
        print(f"数据串口打开失败: {e}")


if __name__ == "__main__":
    test_radar_boot()
