/**
 * @file dm_imu.cpp
 * @brief DM IMU设备驱动实现
 * @details 实现DM IMU设备的串口通信、数据解析、CRC校验和四元数插值功能
 */

#include "dm_imu.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

#include "tools/crc/crc.hpp"
#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"

namespace io
{

// DM IMU 专用 CRC16
namespace
{
static const uint16_t DM_CRC16_TABLE[256] = {
  0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50A5, 0x60C6, 0x70E7, 0x8108, 0x9129,
  0xA14A, 0xB16B, 0xC18C, 0xD1AD, 0xE1CE, 0xF1EF, 0x1231, 0x0210, 0x3273, 0x2252,
  0x52B5, 0x4294, 0x72F7, 0x62D6, 0x9339, 0x8318, 0xB37B, 0xA35A, 0xD3BD, 0xC39C,
  0xF3FF, 0xE3DE, 0x2462, 0x3443, 0x0420, 0x1401, 0x64E6, 0x74C7, 0x44A4, 0x5485,
  0xA56A, 0xB54B, 0x8528, 0x9509, 0xE5EE, 0xF5CF, 0xC5AC, 0xD58D, 0x3653, 0x2672,
  0x1611, 0x0630, 0x76D7, 0x66F6, 0x5695, 0x46B4, 0xB75B, 0xA77A, 0x9719, 0x8738,
  0xF7DF, 0xE7FE, 0xD79D, 0xC7BC, 0x48C4, 0x58E5, 0x6886, 0x78A7, 0x0840, 0x1861,
  0x2802, 0x3823, 0xC9CC, 0xD9ED, 0xE98E, 0xF9AF, 0x8948, 0x9969, 0xA90A, 0xB92B,
  0x5AF5, 0x4AD4, 0x7AB7, 0x6A96, 0x1A71, 0x0A50, 0x3A33, 0x2A12, 0xDBFD, 0xCBDC,
  0xFBBF, 0xEB9E, 0x9B79, 0x8B58, 0xBB3B, 0xAB1A, 0x6CA6, 0x7C87, 0x4CE4, 0x5CC5,
  0x2C22, 0x3C03, 0x0C60, 0x1C41, 0xEDAE, 0xFD8F, 0xCDEC, 0xDDCD, 0xAD2A, 0xBD0B,
  0x8D68, 0x9D49, 0x7E97, 0x6EB6, 0x5ED5, 0x4EF4, 0x3E13, 0x2E32, 0x1E51, 0x0E70,
  0xFF9F, 0xEFBE, 0xDFDD, 0xCFFC, 0xBF1B, 0xAF3A, 0x9F59, 0x8F78, 0x9188, 0x81A9,
  0xB1CA, 0xA1EB, 0xD10C, 0xC12D, 0xF14E, 0xE16F, 0x1080, 0x00A1, 0x30C2, 0x20E3,
  0x5004, 0x4025, 0x7046, 0x6067, 0x83B9, 0x9398, 0xA3FB, 0xB3DA, 0xC33D, 0xD31C,
  0xE37F, 0xF35E, 0x02B1, 0x1290, 0x22F3, 0x32D2, 0x4235, 0x5214, 0x6277, 0x7256,
  0xB5EA, 0xA5CB, 0x95A8, 0x8589, 0xF56E, 0xE54F, 0xD52C, 0xC50D, 0x34E2, 0x24C3,
  0x14A0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405, 0xA7DB, 0xB7FA, 0x8799, 0x97B8,
  0xE75F, 0xF77E, 0xC71D, 0xD73C, 0x26D3, 0x36F2, 0x0691, 0x16B0, 0x6657, 0x7676,
  0x4615, 0x5634, 0xD94C, 0xC96D, 0xF90E, 0xE92F, 0x99C8, 0x89E9, 0xB98A, 0xA9AB,
  0x5844, 0x4865, 0x7806, 0x6827, 0x18C0, 0x08E1, 0x3882, 0x28A3, 0xCB7D, 0xDB5C,
  0xEB3F, 0xFB1E, 0x8BF9, 0x9BD8, 0xABBB, 0xBB9A, 0x4A75, 0x5A54, 0x6A37, 0x7A16,
  0x0AF1, 0x1AD0, 0x2AB3, 0x3A92, 0xFD2E, 0xED0F, 0xDD6C, 0xCD4D, 0xBDAA, 0xAD8B,
  0x9DE8, 0x8DC9, 0x7C26, 0x6C07, 0x5C64, 0x4C45, 0x3CA2, 0x2C83, 0x1CE0, 0x0CC1,
  0xEF1F, 0xFF3E, 0xCF5D, 0xDF7C, 0xAF9B, 0xBFBA, 0x8FD9, 0x9FF8, 0x6E17, 0x7E36,
  0x4E55, 0x5E74, 0x2E93, 0x3EB2, 0x0ED1, 0x1EF0};

/**
 * @brief 计算DM IMU专用的CRC16校验值
 * @param data 待校验的数据指针
 * @param len 数据长度（字节数）
 * @return 计算得到的CRC16校验值
 * @details 使用CCITT标准的CRC16算法，配合DM_CRC16_TABLE查找表进行快速计算
 */
static uint16_t dm_crc16_ccitt(const uint8_t * data, uint16_t len)
{
  uint16_t crc = 0xFFFF;
  for (uint16_t i = 0; i < len; ++i) {
    uint8_t index = static_cast<uint8_t>((crc >> 8) ^ data[i]);
    crc = static_cast<uint16_t>((crc << 1) ^ DM_CRC16_TABLE[index]);
  }
  return crc;
}
}  // namespace

/**
 * @brief 构造函数：初始化DM IMU设备
 * @details 初始化串口连接，启动数据接收线程，并从队列中预取两个数据点用于插值计算
 */
DM_IMU::DM_IMU() : queue_(5000)
{
  init_serial();
  rec_thread_ = std::thread(&DM_IMU::get_imu_data_thread, this);
  queue_.pop(data_ahead_);
  queue_.pop(data_behind_);
  tools::logger()->info("[DM_IMU] initialized");
}

/**
 * @brief 析构函数：清理DM IMU资源
 * @details 停止数据接收线程，等待线程结束，并关闭串口连接
 */
DM_IMU::~DM_IMU()
{
  stop_thread_ = true;
  if (rec_thread_.joinable()) {
    rec_thread_.join();
  }
  if (serial_.isOpen()) {
    serial_.close();
  }
}

/**
 * @brief 初始化串口连接
 * @details 配置串口参数（端口、波特率、数据位、停止位等），打开串口连接
 * @note 串口配置：/dev/ttyACM0, 921600波特率, 8数据位, 1停止位, 无校验, 无流控
 * @exception 如果串口打开失败，程序将退出
 */
void DM_IMU::init_serial()
{
  try {
    serial_.setPort("/dev/ttyACM0");
    serial_.setBaudrate(921600);
    serial_.setFlowcontrol(serial::flowcontrol_none);
    serial_.setParity(serial::parity_none);  //default is parity_none
    serial_.setStopbits(serial::stopbits_one);
    serial_.setBytesize(serial::eightbits);
    serial::Timeout time_out = serial::Timeout::simpleTimeout(20);
    serial_.setTimeout(time_out);
    serial_.open();
    usleep(1000000);  //1s

    tools::logger()->info("[DM_IMU] serial port opened");
  }

  catch (serial::IOException & e) {
    tools::logger()->warn("[DM_IMU] failed to open serial port ");
    exit(0);
  }
}

/**
 * @brief IMU数据接收线程函数
 * @details 在独立线程中持续从串口读取IMU数据帧，进行CRC校验，解析加速度、角速度和欧拉角数据，
 *          并将其转换为四元数后推入队列。数据帧格式包含三个子帧：加速度、角速度和欧拉角，每个子帧都有独立的CRC校验。
 * @note 帧头格式：0x55 0xAA 0xD8 0x01
 * @note 如果CRC校验失败或帧头不正确，将跳过该数据帧并记录警告信息
 */
void DM_IMU::get_imu_data_thread()
{
  while (!stop_thread_) {
    if (!serial_.isOpen()) {
      tools::logger()->warn("In get_imu_data_thread,imu serial port unopen");
    }

    serial_.read((uint8_t *)(&receive_data.FrameHeader1), 4);

    // 帧头应为 0x55 0xAA 0xD8 0x01
    if (
      receive_data.FrameHeader1 == 0x55 && receive_data.flag1 == 0xAA &&
      receive_data.slave_id1 == 0xD8 && receive_data.reg_acc == 0x01)
    {
      serial_.read((uint8_t *)(&receive_data.accx_u32), 57 - 4);

      constexpr uint16_t CRC_DATA_LEN = 16;

      uint16_t crc1_calc =
        dm_crc16_ccitt(reinterpret_cast<uint8_t *>(&receive_data.FrameHeader1), CRC_DATA_LEN);
      uint16_t crc2_calc =
        dm_crc16_ccitt(reinterpret_cast<uint8_t *>(&receive_data.FrameHeader2), CRC_DATA_LEN);
      uint16_t crc3_calc =
        dm_crc16_ccitt(reinterpret_cast<uint8_t *>(&receive_data.FrameHeader3), CRC_DATA_LEN);

      auto * crc1_bytes = reinterpret_cast<uint8_t *>(&receive_data.crc1);
      auto * crc2_bytes = reinterpret_cast<uint8_t *>(&receive_data.crc2);
      auto * crc3_bytes = reinterpret_cast<uint8_t *>(&receive_data.crc3);

      // 设备协议为 CRC_L 在前、CRC_H 在后，因此需要按 [H<<8 | L] 还原
      uint16_t crc1_recv = (static_cast<uint16_t>(crc1_bytes[1]) << 8) | crc1_bytes[0];
      uint16_t crc2_recv = (static_cast<uint16_t>(crc2_bytes[1]) << 8) | crc2_bytes[0];
      uint16_t crc3_recv = (static_cast<uint16_t>(crc3_bytes[1]) << 8) | crc3_bytes[0];

      if (crc1_calc != crc1_recv || crc2_calc != crc2_recv || crc3_calc != crc3_recv) {
        tools::logger()->warn(
          "[DM_IMU] CRC error: acc calc={:#06x}/recv={:#06x}, gyro calc={:#06x}/recv={:#06x}, "
          "euler calc={:#06x}/recv={:#06x}",
          crc1_calc, crc1_recv, crc2_calc, crc2_recv, crc3_calc, crc3_recv);
        continue;
      }

      // CRC 校验通过，解析 float 数据
      data.accx = *((float *)(&receive_data.accx_u32));
      data.accy = *((float *)(&receive_data.accy_u32));
      data.accz = *((float *)(&receive_data.accz_u32));

      data.gyrox = *((float *)(&receive_data.gyrox_u32));
      data.gyroy = *((float *)(&receive_data.gyroy_u32));
      data.gyroz = *((float *)(&receive_data.gyroz_u32));

      data.roll = *((float *)(&receive_data.roll_u32));
      data.pitch = *((float *)(&receive_data.pitch_u32));
      data.yaw = *((float *)(&receive_data.yaw_u32));
      auto timestamp = std::chrono::steady_clock::now();
      Eigen::Quaterniond q = Eigen::AngleAxisd(data.yaw * M_PI / 180, Eigen::Vector3d::UnitZ()) *
                             Eigen::AngleAxisd(data.pitch * M_PI / 180, Eigen::Vector3d::UnitY()) *
                             Eigen::AngleAxisd(data.roll * M_PI / 180, Eigen::Vector3d::UnitX());
      q.normalize();
      queue_.push({q, timestamp});
    } else {
      // 帧头不正确时，打印前 4 字节，方便检查是否对齐到正确帧
      auto *raw4 = reinterpret_cast<uint8_t *>(&receive_data.FrameHeader1);
      tools::logger()->info(
        "[DM_IMU] failed to get correct data, header: {:02X} {:02X} {:02X} {:02X}",
        raw4[0], raw4[1], raw4[2], raw4[3]);
    }
  }
}

/**
 * @brief 根据指定时间戳获取IMU四元数（通过插值计算）
 * @param timestamp 目标时间戳
 * @return 对应时间戳的IMU姿态四元数
 * @details 从队列中查找时间戳前后最近的两个数据点，使用球面线性插值（SLERP）计算目标时间戳的精确姿态。
 *          如果目标时间戳在已有数据之后，会从队列中持续读取新数据直到找到合适的时间范围。
 */
Eigen::Quaterniond DM_IMU::imu_at(std::chrono::steady_clock::time_point timestamp)
{
  if (data_behind_.timestamp < timestamp) data_ahead_ = data_behind_;

  while (true) {
    queue_.pop(data_behind_);
    if (data_behind_.timestamp > timestamp) break;
    data_ahead_ = data_behind_;
  }

  Eigen::Quaterniond q_a = data_ahead_.q.normalized();
  Eigen::Quaterniond q_b = data_behind_.q.normalized();
  auto t_a = data_ahead_.timestamp;
  auto t_b = data_behind_.timestamp;
  auto t_c = timestamp;
  std::chrono::duration<double> t_ab = t_b - t_a;
  std::chrono::duration<double> t_ac = t_c - t_a;

  // 四元数插值
  auto k = t_ac / t_ab;
  Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();

  return q_c;
}

}  // namespace io
