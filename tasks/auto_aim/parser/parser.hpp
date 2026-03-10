#include <opencv2/opencv.hpp>

namespace auto_aim
{

/**
 * @brief 解析和存放某一类结果的解析器
 */
template <typename T>
class Parser
{
public:
  explicit Parser(int begin, int length);
  virtual ~Parser() = default;

  /**
   * @brief 读取最后一个被放入解析器的结果
   * @return 最后一个被放入解析器的结果
   */
  const T & back() const;

  /**
   * @brief 获取解析器中的数据引用
   * @return 解析器中的数据
   */
  const std::vector<T> & data() const;

  const T & operator[](int index) const;

  /**
   * @brief 对数据行进行解析
   * @param row 待解析的数据行
   */
  void parse(const cv::Mat & row);

  /**
   * @brief 对数据行进行处理
   * @param raw_data 待解析的原始数据
   * @return 处理得到的目标结果
   */
  virtual T process(const cv::Mat & raw_data) = 0;

protected:
  std::vector<T> datas_;
  int begin_;
  int length_;
};

class KeyPointsParse : public Parser<std::vector<cv::Point2f>>
{
public:
  KeyPointsParse(int begin);

  /**
   * @brief 对数据行进行处理
   * @param raw_data 待解析的原始数据
   * @return 处理得到的4个关键点
   */
  std::vector<cv::Point2f> process(const cv::Mat & raw_data) override;

private:
  /**
   * @brief 对关键点进行排序，使其按顺序为：左上、右上、右下、左下
   * @param keypoints 待排序的关键点向量（输入输出参数，会被修改）
   */
  static void sort_keypoints(std::vector<cv::Point2f> & keypoints);
};
}  // namespace auto_aim