#include "sse-server.hpp"

#include <httplib.h>
#include <opencv2/imgcodecs.hpp>

#include <fstream>
#include <iostream>
#include <sstream>

// ── SSE client tracking ─────────────────────────────────────────────────────
struct SseClients {
  std::mutex mutex;
  std::vector<httplib::DataSink*> sinks;

  void add(httplib::DataSink* sink) {
    std::lock_guard<std::mutex> lock(mutex);
    sinks.push_back(sink);
  }

  void remove(httplib::DataSink* sink) {
    std::lock_guard<std::mutex> lock(mutex);
    sinks.erase(std::remove(sinks.begin(), sinks.end(), sink), sinks.end());
  }

  void broadcast(const std::string& data) {
    std::lock_guard<std::mutex> lock(mutex);
    std::string message = "data: " + data + "\n\n";
    for (auto it = sinks.begin(); it != sinks.end();) {
      if (!(*it)->write(message.data(), message.size())) {
        it = sinks.erase(it);
      } else {
        ++it;
      }
    }
  }
};

// ── Helpers ─────────────────────────────────────────────────────────────────
static bool endsWith(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() &&
         str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static std::string contentType(const std::string& path) {
  if (endsWith(path, ".html")) return "text/html";
  if (endsWith(path, ".js")) return "application/javascript";
  if (endsWith(path, ".css")) return "text/css";
  if (endsWith(path, ".json")) return "application/json";
  if (endsWith(path, ".png")) return "image/png";
  return "text/plain";
}

static std::string readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return "";
  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

// ── SseServer implementation ────────────────────────────────────────────────
SseServer::SseServer(const SseServerConfig& config) : config_(config) {}

SseServer::~SseServer() {
  if (running_) stop();
}

void SseServer::broadcast(const std::string& json) {
  if (sseClients_) sseClients_->broadcast(json);
}

void SseServer::updateFrame(const cv::Mat& frame) {
  std::vector<unsigned char> buf;
  cv::imencode(".jpg", frame, buf, {cv::IMWRITE_JPEG_QUALITY, 70});

  {
    std::lock_guard<std::mutex> lock(frameMutex_);
    jpegBuffer_ = std::move(buf);
    frameSeq_++;
  }
  frameReady_.notify_all();
}

void SseServer::start() {
  if (running_) return;
  running_ = true;

  auto clients = std::make_shared<SseClients>();

  // Replace the stub broadcast() with one that uses the shared clients
  // We use a pointer-to-shared trick since broadcast() is a member function
  // Store clients pointer as member-accessible via lambda capture in httpThread_

  httpThread_ = std::thread([this, clients]() {
    httplib::Server svr;

    // Health check
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
      res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    // SSE endpoint
    svr.Get("/events",
            [clients](const httplib::Request&, httplib::Response& res) {
              res.set_header("Cache-Control", "no-cache");
              res.set_header("Access-Control-Allow-Origin", "*");
              auto sinkPtr = std::make_shared<httplib::DataSink*>(nullptr);
              res.set_chunked_content_provider(
                  "text/event-stream",
                  [clients, sinkPtr](size_t, httplib::DataSink& sink) {
                    *sinkPtr = &sink;
                    clients->add(&sink);
                    while (true) {
                      std::this_thread::sleep_for(
                          std::chrono::milliseconds(100));
                    }
                    return true;
                  },
                  [clients, sinkPtr](bool) {
                    if (*sinkPtr) clients->remove(*sinkPtr);
                  });
            });

    // MJPEG video stream endpoint
    svr.Get("/video", [this](const httplib::Request&, httplib::Response& res) {
      res.set_header("Cache-Control", "no-cache");
      res.set_header("Access-Control-Allow-Origin", "*");
      res.set_chunked_content_provider(
          "multipart/x-mixed-replace; boundary=frame",
          [this](size_t, httplib::DataSink& sink) {
            uint64_t lastSeq = 0;
            while (running_) {
              // Wait for new frame
              std::unique_lock<std::mutex> lock(frameMutex_);
              frameReady_.wait_for(lock, std::chrono::milliseconds(100),
                                    [this, lastSeq] {
                                      return frameSeq_ > lastSeq || !running_;
                                    });

              if (!running_) break;
              if (frameSeq_ == lastSeq) continue;

              lastSeq = frameSeq_;
              // Copy JPEG buffer while holding lock
              std::vector<unsigned char> jpeg = jpegBuffer_;
              lock.unlock();

              // Write MJPEG frame boundary
              std::string header = "--frame\r\nContent-Type: image/jpeg\r\n"
                                   "Content-Length: " +
                                   std::to_string(jpeg.size()) + "\r\n\r\n";

              if (!sink.write(header.data(), header.size())) return false;
              if (!sink.write(reinterpret_cast<const char*>(jpeg.data()),
                              jpeg.size()))
                return false;
              if (!sink.write("\r\n", 2)) return false;
            }
            return false;
          },
          [](bool) {});
    });

    // Serve index.html at root
    svr.Get("/", [this](const httplib::Request&, httplib::Response& res) {
      auto content = readFile(config_.staticDir + "index.html");
      if (content.empty()) {
        res.status = 404;
        res.set_content("Not found", "text/plain");
      } else {
        res.set_content(content, "text/html");
      }
    });

    // Static files
    svr.Get("/static/(.*)",
            [this](const httplib::Request& req, httplib::Response& res) {
              std::string path = req.matches[1];
              if (path.find("..") != std::string::npos) {
                res.status = 403;
                res.set_content("Forbidden", "text/plain");
                return;
              }
              auto content = readFile(config_.staticDir + path);
              if (content.empty()) {
                res.status = 404;
                res.set_content("Not found", "text/plain");
              } else {
                res.set_content(content, contentType(path));
              }
            });

    std::cout << "SSE server listening on http://" << config_.bindAddress << ":"
              << config_.port << "\n";
    svr.listen(config_.bindAddress, config_.port);
  });

  // Store clients pointer for broadcast() calls from main thread
  // We use a small indirection: replace broadcast() behavior via stored ptr
  sseClients_ = clients;
}

void SseServer::stop() {
  if (!running_) return;
  running_ = false;
  frameReady_.notify_all();

  if (httpThread_.joinable()) httpThread_.detach();
  std::cout << "SSE server stopped\n";
}
