#ifndef IO__GIMBAL_STATE_HPP
#define IO__GIMBAL_STATE_HPP

#include <cstdint>

namespace io
{
enum class GimbalMode
{
  IDLE,        // 空闲
  AUTO_AIM,    // 自瞄
  SMALL_BUFF,  // 小符
  BIG_BUFF     // 大符
};

struct GimbalState
{
  float yaw;
  float yaw_vel;
  float pitch;
  float pitch_vel;
  float bullet_speed;
  uint16_t bullet_count;
};

}  // namespace io

#endif  // IO__GIMBAL_STATE_HPP
