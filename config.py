# config.py
SERIAL_PORT_DATA = '/dev/cu.usbserial-01C842A91'
SERIAL_PORT_CLI  = '/dev/cu.usbserial-01C842A90'
BAUD_RATE_DATA   = 921600
BAUD_RATE_CLI    = 115200

FS = 20.0
WINDOW_SECONDS = 15
BUFFER_SIZE = int(FS * WINDOW_SECONDS)

# 放宽呼吸频率上限，适应自然喘息
BREATH_LOW, BREATH_HIGH = 0.1, 0.6
HEART_LOW, HEART_HIGH = 0.8, 2.0

USE_CNN_ENGINE = True
CNN_MODEL_PATH = "vital_cnn_robust.pth"

RADAR_CFG_COMMANDS = [
    "sensorStop",
    "flushCfg",
    "dfeDataOutputMode 1",
    "channelCfg 15 7 0",
    "adcCfg 2 1",
    "adcbufCfg -1 0 1 1 1",
    "profileCfg 0 77 267 7 57.14 0 0 70 1 64 5209 0 0 30",
    "chirpCfg 0 0 0 0 0 0 0 1",     # 仅需单 chirp
    "frameCfg 0 0 16 0 50 1 0",     # 匹配单 chirp 结构，50ms=20Hz
    "lowPower 0 0",
    "guiMonitor -1 1 1 0 1 0 0",
    "cfarCfg -1 0 2 8 4 3 0 15 1",
    "cfarCfg -1 1 0 4 2 3 1 15 1",
    "multiObjBeamForming -1 1 0.5",
    "clutterRemoval -1 0",
    "calibDcRangeSig -1 0 -5 8 64",
    "extendedMaxVelocity -1 0",
    "lvdsStreamCfg -1 0 0 0",
    "compRangeBiasAndRxChanPhase 0.0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0",
    "measureRangeBiasAndRxChanPhase 0 1.5 0.2",
    "CQRxSatMonitor 0 3 5 121 0",
    "CQSigImgMonitor 0 127 4",
    "analogMonitor 0 0",
    "aoaFovCfg -1 -90 90 -90 90",
    "cfarFovCfg -1 0 0 8.92",
    "cfarFovCfg -1 1 -1 1.00",
    "calibData 0 0 0",
    "sensorStart"
]