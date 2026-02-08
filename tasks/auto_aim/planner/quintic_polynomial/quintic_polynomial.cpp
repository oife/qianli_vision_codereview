#include "quintic_polynomial.hpp"

#include <cmath>
#include <algorithm>

namespace auto_aim
{

Vector6d QuinticPolynomialSolver::solve(
  const BoundaryCondition & start, const BoundaryCondition & end, double T)
{
  // 五次多项式：q(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
  // 边界条件：
  // q(0) = start.pos, q'(0) = start.vel, q''(0) = start.acc
  // q(T) = end.pos, q'(T) = end.vel, q''(T) = end.acc

  if (T <= 0) {
    // 如果时间无效，返回零系数
    return Vector6d::Zero();
  }

  // 构建线性方程组 A * coeffs = b
  // 其中 coeffs = [a0, a1, a2, a3, a4, a5]^T

  Eigen::Matrix6d A;
  Eigen::Vector6d b;

  double T2 = T * T;
  double T3 = T2 * T;
  double T4 = T3 * T;
  double T5 = T4 * T;

  // q(0) = a0 = start.pos
  A.row(0) << 1, 0, 0, 0, 0, 0;
  b(0) = start.pos;

  // q'(0) = a1 = start.vel
  A.row(1) << 0, 1, 0, 0, 0, 0;
  b(1) = start.vel;

  // q''(0) = 2*a2 = start.acc
  A.row(2) << 0, 0, 2, 0, 0, 0;
  b(2) = start.acc;

  // q(T) = a0 + a1*T + a2*T² + a3*T³ + a4*T⁴ + a5*T⁵ = end.pos
  A.row(3) << 1, T, T2, T3, T4, T5;
  b(3) = end.pos;

  // q'(T) = a1 + 2*a2*T + 3*a3*T² + 4*a4*T³ + 5*a5*T⁴ = end.vel
  A.row(4) << 0, 1, 2 * T, 3 * T2, 4 * T3, 5 * T4;
  b(4) = end.vel;

  // q''(T) = 2*a2 + 6*a3*T + 12*a4*T² + 20*a5*T³ = end.acc
  A.row(5) << 0, 0, 2, 6 * T, 12 * T2, 20 * T3;
  b(5) = end.acc;

  // 求解线性方程组
  Vector6d coeffs = A.colPivHouseholderQr().solve(b);

  return coeffs;
}

double QuinticPolynomialSolver::evaluate_pos(const Vector6d & coeffs, double t)
{
  double t2 = t * t;
  double t3 = t2 * t;
  double t4 = t3 * t;
  double t5 = t4 * t;

  return coeffs(0) + coeffs(1) * t + coeffs(2) * t2 + coeffs(3) * t3 + coeffs(4) * t4 +
         coeffs(5) * t5;
}

double QuinticPolynomialSolver::evaluate_vel(const Vector6d & coeffs, double t)
{
  double t2 = t * t;
  double t3 = t2 * t;
  double t4 = t3 * t;

  return coeffs(1) + 2 * coeffs(2) * t + 3 * coeffs(3) * t2 + 4 * coeffs(4) * t3 +
         5 * coeffs(5) * t4;
}

double QuinticPolynomialSolver::evaluate_acc(const Vector6d & coeffs, double t)
{
  double t2 = t * t;
  double t3 = t2 * t;

  return 2 * coeffs(2) + 6 * coeffs(3) * t + 12 * coeffs(4) * t2 + 20 * coeffs(5) * t3;
}

bool QuinticPolynomialSolver::check_constraints(
  const Vector6d & coeffs, double T, double v_max, double a_max, int sample_num)
{
  // 在过渡段内均匀采样检查约束
  for (int i = 0; i <= sample_num; ++i) {
    double t = T * i / sample_num;
    double vel = std::abs(evaluate_vel(coeffs, t));
    double acc = std::abs(evaluate_acc(coeffs, t));

    if (vel > v_max || acc > a_max) {
      return false;
    }
  }
  return true;
}

}  // namespace auto_aim

