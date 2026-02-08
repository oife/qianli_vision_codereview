#ifndef AUTO_AIM__QUINTIC_POLYNOMIAL_HPP
#define AUTO_AIM__QUINTIC_POLYNOMIAL_HPP

#include <Eigen/Dense>
#include <vector>

namespace auto_aim
{

// 定义6维向量类型（五次多项式有6个系数）
using Vector6d = Eigen::Matrix<double, 6, 1>;

/**
 * @brief 五次多项式求解器
 * 
 * 用于生成平滑的过渡轨迹，满足起点和终点的位置、速度、加速度约束
 * 五次多项式形式：q(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
 */
class QuinticPolynomialSolver
{
public:
  /**
   * @brief 边界条件结构体
   */
  struct BoundaryCondition
  {
    double pos;   // 位置
    double vel;   // 速度
    double acc;   // 加速度
  };

  /**
   * @brief 求解五次多项式系数
   * @param start 起点边界条件
   * @param end 终点边界条件
   * @param T 过渡时间（秒）
   * @return 6个系数 [a0, a1, a2, a3, a4, a5]
   */
  static Vector6d solve(
    const BoundaryCondition & start, const BoundaryCondition & end, double T);

  /**
   * @brief 计算多项式在时刻t的值
   * @param coeffs 多项式系数 [a0, a1, a2, a3, a4, a5]
   * @param t 时间（相对于起点，0到T之间）
   * @return 位置值
   */
  static double evaluate_pos(const Vector6d & coeffs, double t);

  /**
   * @brief 计算多项式在时刻t的一阶导数（速度）
   * @param coeffs 多项式系数
   * @param t 时间
   * @return 速度值
   */
  static double evaluate_vel(const Vector6d & coeffs, double t);

  /**
   * @brief 计算多项式在时刻t的二阶导数（加速度）
   * @param coeffs 多项式系数
   * @param t 时间
   * @return 加速度值
   */
  static double evaluate_acc(const Vector6d & coeffs, double t);

  /**
   * @brief 验证过渡段是否满足速度和加速度约束
   * @param coeffs 多项式系数
   * @param T 过渡时间
   * @param v_max 最大速度约束
   * @param a_max 最大加速度约束
   * @param sample_num 采样点数（默认200）
   * @return 如果满足约束返回true，否则返回false
   */
  static bool check_constraints(
    const Vector6d & coeffs, double T, double v_max, double a_max,
    int sample_num = 200);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__QUINTIC_POLYNOMIAL_HPP

