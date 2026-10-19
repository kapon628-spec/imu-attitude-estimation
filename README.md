# IMU 姿态估计与传感器融合

基于 **STM32F411CEU6 + MPU6050** 的实时姿态估计系统，用于学习并实践多传感器融合，作为嵌入式求职作品。

## 硬件配置

| 项目 | 参数 |
|---|---|
| MCU | STM32F411CEU6，96 MHz（HSE 25 MHz, PLLM=25 / PLLN=384 / PLLP=4） |
| IMU | MPU6050，I2C 地址 0x68（AD0 接地） |
| 陀螺仪量程 | ±250 °/s，灵敏度 131 LSB/(°/s) |
| 加速度量程 | ±2 g，灵敏度 16384 LSB/g |
| 串口 | USART1 @ 115200（PA9 = TX, PA10 = RX → CH340） |
| 采样 | TIM2 每 5 ms 中断，200 Hz |
| 时间戳 | TIM5 自由运行，1 MHz（1 µs 分辨率） |
| 调试 | ST-Link V2，SWD 接口 |

## 软件架构

```
MPU6050 ──I2C──> STM32F411CEU6 ──UART──> CH340 ──> PC
                       │
              TIM2  5 ms 中断触发采样
              TIM5  提供微秒级时间戳
```

| 文件 | 说明 |
|---|---|
| `imu_plot.py` | 实时波形显示（pyserial + matplotlib） |
| `imu_log.py` | 采集并统计噪声基线，输出 CSV |

## 数据协议

每 20 ms 发送一行（50 Hz）：

```
D,<dt_us>,<ax>,<ay>,<az>,<gx>,<gy>,<gz>
```

| 字段 | 含义 |
|---|---|
| `D` | 帧头 |
| `dt_us` | 本帧与上一帧的实际时间间隔（µs） |
| `ax / ay / az` | 加速度原始值（LSB） |
| `gx / gy / gz` | 角速度原始值（LSB） |

**换算关系：**

- 加速度：`mg = raw × 1000 / 16384`
- 角速度：`°/s = raw / 131`
- 温度（未上传）：`℃ = raw / 340 + 36.53`

## 基线数据

2026-10-19，静止 60 秒，3001 帧：

| 指标 | 数值 |
|---|---|
| 采样间隔 | 5000 µs（标准差 0.0，峰峰值 0） |
| 陀螺零偏 | x = −3.687, y = −2.493, z = −1.139 °/s |
| 陀螺噪声 σ | 0.077 / 0.096 / 0.093 °/s |
| 加速度均值 | 1038.9 / −12.3 / −193.7 mg（\|a\| = 1.057 g） |
| 加速度噪声 σ | 3.35 / 2.93 / 4.09 mg |

> **零偏的影响**：−3.687 °/s 直接积分 60 秒会漂 221°，而噪声只贡献约 0.04°。
> 因此必须先做零偏标定（Day 9），再用加速度计融合纠正残余漂移（Day 10）。

## 进度

- [x] **Day 1–2**：环境搭建、点灯、串口 + printf 重定向
- [x] **Day 3**：I2C 扫描，识别 MPU6050（0x68）
- [x] **Day 4**：读六轴原始值，单位换算，物理验证（重力 ±1 g）
- [x] **Day 5**：TIM2 定时采样 + TIM5 时间戳，dt 稳定 5000 µs
- [x] **Day 6**：CSV 输出 + Python 实时波形上位机
- [x] **Day 7**：噪声基线测量与统计（3001 帧）
- [ ] **Day 8–14**：姿态解算、互补滤波、Mahony、3D 可视化
- [ ] **Day 15–21**：磁力计接入、椭圆校准、定量分析
- [ ] **Day 22–28**：RMSE 统计、报告、演示视频

## 环境

```bash
pip install pyserial matplotlib numpy
```

## 运行

1. 修改 `imu_plot.py` / `imu_log.py` 中的 `PORT` 为实际串口号
2. 关闭其他占用串口的程序（串口助手会独占端口）
3. 运行 `python imu_plot.py`（实时波形）或 `python imu_log.py`（采集统计）

> 采集基线数据时，板子需静置桌面、手离开，避免振动引入额外噪声。

## 已知踩坑记录

| 问题 | 原因 | 解决                                 |
|---|---|--------------------------------------|
| `dt` 恒定 502000 µs | 主循环里 `HAL_Delay(500)` 阻塞了采样 | 改用计数器分频做低频动作             |
| 时间戳读数放大 96 倍 | TIM5 Prescaler 未设置（默认 0，跑在 96 MHz） | Prescaler 设为 95                    |
| `conflicting types for GetMicros` | C 语言要求先声明后使用 | 在 `USER CODE BEGIN PFP` 加前置声明  |
| 数据全为 0x00 | 唤醒写早于芯片上电稳定时间 | 首次 I2C 操作前 `HAL_Delay(100)`     |
| `HAL_BUSY` 打不开总线 | MPU6050 模块掉电或接线松动 | 检查 VCC、GND、SCL、SDA 四根线       |
| SWD 报 `Target no device found` | SWCLK/SWDIO/GND 接错或焊点虚焊 | 万用表逐根量通断；确认 SWSCK = SWCLK |

## 参考资料

- [MPU-6050 Register Map and Descriptions](https://invensense.tdk.com/wp-content/uploads/2015/02/MPU-6000-Register-Map1.pdf)
- [MPU-6050 Product Specification](https://invensense.tdk.com/wp-content/uploads/2015/02/MPU-6000-Datasheet1.pdf)
