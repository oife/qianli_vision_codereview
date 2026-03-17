#ifndef TOOLS__PLOTTER_HPP
#define TOOLS__PLOTTER_HPP

#include <netinet/in.h>  // sockaddr_in

#include <mutex>
#include <nlohmann/json.hpp>
#include <string>

namespace tools
{
class Plotter
{
public:
  // protocol: "udp" (default, backward compatible) or "tcp"
  Plotter(
    std::string host = "127.0.0.1", uint16_t port = 9870, std::string protocol = "udp");

  ~Plotter();

  void plot(const nlohmann::json & json);

private:
  int socket_ = -1;
  sockaddr_in destination_;
  std::mutex mutex_;
  bool tcp_;
  bool connected_ = false;

  void tcp_connect();
};

}  // namespace tools

#endif  // TOOLS__PLOTTER_HPP