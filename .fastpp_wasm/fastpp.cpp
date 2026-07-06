#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "warm_pool.h"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

struct ImageView {
  const uint8_t* data = nullptr;
  int width;
  int height;
  int channels;
  int stride;
};

struct Keypoint {
  int x, y;
};

// ---------------------------------------------------------------------------
// FAST corner detector
// ---------------------------------------------------------------------------

static std::vector<Keypoint> FAST(const int w, const int h,
                                  const unsigned char* grayImageData, int N = 9,
                                  float threshold = 0.15f, int nmsWindow = 2,
                                  WarmPool* pool = nullptr) {
  // Bresenham circle radius-3 (16 pixels, clockwise from top)
  static const int cx[16] = {0, 1,  2,  3,  3,  3,  2,  1,
                             0, -1, -2, -3, -3, -3, -2, -1};
  static const int cy[16] = {-3, -3, -2, -1, 0, 1,  2,  3,
                             3,  3,  2,  1,  0, -1, -2, -3};

  // Cross: 4 cardinal pixels (indices 0,4,8,12)
  static const int cross_x[4] = {0, 3, 0, -3};
  static const int cross_y[4] = {-3, 0, 3, 0};

  std::vector<float> corner_score(w * h, 0.f);
  // one bucket per row: each parallel_for iteration only ever touches its own
  // y, so writing into row_keypoints[y] needs no locking
  std::vector<std::vector<Keypoint>> row_keypoints(h);

  auto process_row = [&](size_t y_) {
    int y = (int)y_;
    std::vector<Keypoint>& local = row_keypoints[y];
    for (int x = 3; x < w - 3; ++x) {
      float Ip = grayImageData[y * w + x];
      float t = (threshold < 1.f) ? threshold * Ip : threshold;

      int bright = 0, dark = 0;
      for (int i = 0; i < 4; ++i) {
        float v = grayImageData[(y + cross_y[i]) * w + (x + cross_x[i])];
        if (v >= Ip + t) ++bright;
        if (v <= Ip - t) ++dark;
      }
      if (bright < 3 && dark < 3) continue;

      int b_cnt = 0, d_cnt = 0, b_max = 0, d_max = 0;
      for (int i = 0; i < 16 * 2; ++i) {
        float v = grayImageData[(y + cy[i % 16]) * w + (x + cx[i % 16])];
        if (v >= Ip + t) {
          ++b_cnt;
          b_max = std::max(b_max, b_cnt);
        } else
          b_cnt = 0;
        if (v <= Ip - t) {
          ++d_cnt;
          d_max = std::max(d_max, d_cnt);
        } else
          d_cnt = 0;
      }
      if (b_max < N && d_max < N) continue;

      float score = 0.f;
      for (int i = 0; i < 16; ++i)
        score += std::fabs(Ip - grayImageData[(y + cy[i]) * w + (x + cx[i])]);

      local.push_back({x, y});
      corner_score[y * w + x] = score;
    }
  };

  if (pool) {
    pool->parallel_for(3, h - 3, process_row);
  } else {
    for (int y = 3; y < h - 3; ++y) process_row(y);
  }

  std::vector<Keypoint> keypoints;
  for (auto& row : row_keypoints) {
    keypoints.insert(keypoints.end(), row.begin(), row.end());
  }

  if (nmsWindow == 0) return keypoints;

  std::vector<Keypoint> result;
  for (const auto& kp : keypoints) {
    int x = kp.x, y = kp.y;
    int best_x = x, best_y = y;
    float best = 0.f;

    for (int dy = -nmsWindow; dy <= nmsWindow; ++dy) {
      for (int dx = -nmsWindow; dx <= nmsWindow; ++dx) {
        int nx = x + dx, ny = y + dy;
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        float s = corner_score[ny * w + nx];
        if (s > best) {
          best = s;
          best_x = nx;
          best_y = ny;
        }
      }
    }

    Keypoint winner{best_x, best_y};
    auto it = std::find_if(
        result.begin(), result.end(),
        [&](const Keypoint& k) { return k.x == winner.x && k.y == winner.y; });
    if (it == result.end()) result.push_back(winner);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Harris response + orientation for a single pixel in a 1-channel uint8 buffer
// ---------------------------------------------------------------------------

inline int computeHarrisResponse(const unsigned char* src, const int width,
                                 const int height, const int px, const int py,
                                 const int blockSize, const double k,
                                 float& score, float& angle) {
  int h_block = blockSize / 2;
  int border = h_block + 1;  // 3x3 Sobel needs 1-pixel aperture border

  if (px < border || px >= width - border || py < border ||
      py >= height - border) {
    score = 0.0f;
    angle = 0.0f;
    return -1;
  }

  double M00 = 0, M01 = 0, M11 = 0;

  for (int dr = -h_block; dr <= h_block; ++dr) {
    for (int dc = -h_block; dc <= h_block; ++dc) {
      int r = py + dr;
      int c = px + dc;

      float dx =
          (float)(-src[(r - 1) * width + (c - 1)] +
                  src[(r - 1) * width + (c + 1)] -
                  2 * src[r * width + (c - 1)] + 2 * src[r * width + (c + 1)] -
                  src[(r + 1) * width + (c - 1)] +
                  src[(r + 1) * width + (c + 1)]);

      float dy = (float)(-src[(r - 1) * width + (c - 1)] -
                         2 * src[(r - 1) * width + c] -
                         src[(r - 1) * width + (c + 1)] +
                         src[(r + 1) * width + (c - 1)] +
                         2 * src[(r + 1) * width + c] +
                         src[(r + 1) * width + (c + 1)]);

      M00 += dx * dx;
      M11 += dy * dy;
      M01 += dx * dy;
    }
  }

  double det = (M00 * M11 - M01 * M01) / (blockSize * blockSize);
  double trace = (M00 + M11) / blockSize;
  score = (float)(det - k * trace * trace);

  double u = (M00 + M11) * 0.5;
  double v = std::sqrt((M00 - M11) * (M00 - M11) * 0.25 + M01 * M01);
  double l1 = u + v;
  angle = (float)std::atan2(l1 - M00, M01);

  return 0;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

extern "C" {

// Detects up to `maxPoints` interest points in a 4-channel (RGBA) image.
//
// Inputs:
//   w, h           : image width and height in pixels
//   rgbaImageData  : pointer to a contiguous buffer of size w * h * 4 bytes
//                    (RGBA, row-major order); alpha channel is ignored
//   maxPoints      : maximum number of keypoints to return
//   outPointer     : caller-allocated buffer of maxPoints * 4 floats.
//                    Layout per keypoint: [x, y, angle, response]
//
// Returns:
//   number of keypoints written (0..maxPoints), or negative on error.
int fastpp(const int w, const int h, const unsigned char* rgbaImageData,
           const int maxPoints, float* outPointer) {
  if (!rgbaImageData || !outPointer || w <= 0 || h <= 0 || maxPoints <= 0)
    return -1;

  // RGBA -> grayscale (luminance, integer approximation; alpha ignored)
  int numPixels = w * h;
  std::vector<uint8_t> gray(numPixels);
  for (int i = 0; i < numPixels; ++i) {
    const unsigned char* p = rgbaImageData + i * 4;
    gray[i] = (uint8_t)((77 * p[0] + 150 * p[1] + 29 * p[2]) >> 8);
  }

  // FAST corner detection, parallelized across a persistent worker pool
  static WarmPool pool(std::max(1u, std::thread::hardware_concurrency()));
  std::vector<Keypoint> kps = FAST(w, h, gray.data(), 9, 0.15f, 2, &pool);

  // Score each FAST keypoint with Harris response
  struct Scored {
    int x, y;
    float score, angle;
  };
  std::vector<Scored> scored;
  scored.reserve(kps.size());

  for (const auto& kp : kps) {
    float score = 0.f, angle = 0.f;
    if (computeHarrisResponse(gray.data(), w, h, kp.x, kp.y, 3, 0.04, score,
                              angle) == 0 &&
        score > 0.f) {
      scored.push_back({kp.x, kp.y, score, angle});
    }
  }

  // Sort descending by Harris response
  std::sort(scored.begin(), scored.end(),
            [](const Scored& a, const Scored& b) { return a.score > b.score; });

  int n = (int)std::min((int)scored.size(), maxPoints);
  std::fill_n(outPointer, maxPoints * 4, -1.f);

  for (int i = 0; i < n; ++i) {
    outPointer[i * 4 + 0] = (float)scored[i].x;
    outPointer[i * 4 + 1] = (float)scored[i].y;
    outPointer[i * 4 + 2] = scored[i].angle;
    outPointer[i * 4 + 3] = scored[i].score;
  }
  return n;
}

}  // extern "C"

#ifdef FASTPP_DEMO
#include <chrono>
#include <fstream>
#include <iostream>

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <image.bin>\n";
    return 1;
  }

  std::ifstream f(argv[1], std::ios::binary);
  if (!f) {
    std::cout << "open " << argv[1] << " failed\n";
    return 1;
  }

  int32_t width, height, channels, stride;
  if (!f.read(reinterpret_cast<char*>(&width), sizeof(int32_t)) ||
      !f.read(reinterpret_cast<char*>(&height), sizeof(int32_t)) ||
      !f.read(reinterpret_cast<char*>(&channels), sizeof(int32_t)) ||
      !f.read(reinterpret_cast<char*>(&stride), sizeof(int32_t))) {
    std::cout << "read header failed\n";
    return 1;
  }

  const int data_bytes = width * channels * height;
  std::vector<uint8_t> data(data_bytes);
  f.read(reinterpret_cast<char*>(data.data()), data_bytes);

  const int max_points = 100;
  std::vector<float> out(max_points * 4);

  auto start = std::chrono::high_resolution_clock::now();
  auto kps = fastpp(width, height, data.data(), max_points, out.data());
  auto end = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < kps; ++i) {
    if (out[i * 4 + 0] > 0.f && out[i * 4 + 1] > 0.f && out[i * 4 + 3] > 0.f) {
      printf("kp[%d]: x=%f y=%f angle=%f score=%f\n", i, out[i * 4 + 0],
             out[i * 4 + 1], out[i * 4 + 2], out[i * 4 + 3]);
    }
  }
  std::cout << "fastpp() found " << kps << " keypoints in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " ms\n";

  return 0;
}
#endif