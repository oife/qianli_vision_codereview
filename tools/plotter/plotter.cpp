#include "plotter.hpp"

#include <arpa/inet.h>   // htons, inet_addr
#include <sys/socket.h>  // socket, sendto, connect, send
#include <unistd.h>      // close

#include "tools/logger/logger.hpp"

namespace tools
{
Plotter::Plotter(std::string host, uint16_t port, std::string protocol) : tcp_(protocol == "tcp")
{
  destination_.sin_family = AF_INET;
  destination_.sin_port = ::htons(port);
  destination_.sin_addr.s_addr = ::inet_addr(host.c_str());

  if (tcp_) {
    tcp_connect();
  } else {
    socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);
  }
}

Plotter::~Plotter()
{
  if (socket_ >= 0) ::close(socket_);
}

void Plotter::tcp_connect()
{
  if (socket_ >= 0) ::close(socket_);
  socket_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (socket_ < 0) {
    connected_ = false;
    return;
  }
  // 设置非阻塞连接超时（100ms）
  struct timeval tv;
  tv.tv_sec = 0;
  tv.tv_usec = 100000;
  setsockopt(socket_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

  if (::connect(socket_, reinterpret_cast<sockaddr *>(&destination_), sizeof(destination_)) == 0) {
    connected_ = true;
  } else {
    connected_ = false;
  }
}

void Plotter::plot(const nlohmann::json & json)
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto data = json.dump() + "\n";  // 每条 JSON 以换行分隔

  if (tcp_) {
    if (!connected_) tcp_connect();
    if (!connected_) return;
    ssize_t n = ::send(socket_, data.c_str(), data.length(), MSG_NOSIGNAL);
    if (n <= 0) {
      connected_ = false;  // 断开，下次重连
    }
  } else {
    ::sendto(
      socket_, data.c_str(), data.length(), 0, reinterpret_cast<sockaddr *>(&destination_),
      sizeof(destination_));
  }
}

}  // namespace tools