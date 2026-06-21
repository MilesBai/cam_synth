#include "feat_extractor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace vision
{

// ---------------------------------------------------------------------------
// Internal helpers (file-scope)
// ---------------------------------------------------------------------------

static inline int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : v > hi ? hi : v;
}

static void gaussian_blur(float* buf, int w, int h)
{
    static const float k[3][3] = {
        {1/16.f, 2/16.f, 1/16.f},
        {2/16.f, 4/16.f, 2/16.f},
        {1/16.f, 2/16.f, 1/16.f}
    };

    std::vector<float> tmp(w * h);
    for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
    {
        float acc = 0.f;
        for (int ky = -1; ky <= 1; ++ky)
        for (int kx = -1; kx <= 1; ++kx)
        {
            int sy = clampi(y + ky, 0, h - 1);
            int sx = clampi(x + kx, 0, w - 1);
            acc += k[ky+1][kx+1] * buf[sy * w + sx];
        }
        tmp[y * w + x] = acc;
    }
    std::memcpy(buf, tmp.data(), w * h * sizeof(float));
}

// Bresenham circle of radius 3 (16 pixels, starting at top, clockwise)
static constexpr int kCircleX[16] = { 0, 1, 2, 3, 3, 3, 2, 1, 0,-1,-2,-3,-3,-3,-2,-1};
static constexpr int kCircleY[16] = {-3,-3,-2,-1, 0, 1, 2, 3, 3, 3, 2, 1, 0,-1,-2,-3};

// Cardinal cross (indices 0,4,8,12 on the circle)
static constexpr int kCrossX[4] = { 0, 3, 0,-3};
static constexpr int kCrossY[4] = {-3, 0, 3, 0};

static FeatureSet run_fast(const ImageView& img,
                            int   N          = 9,
                            float threshold  = 0.15f,
                            int   nms_window = 2,
                            int   max_features = 0)
{
    const int w      = img.width;
    const int h      = img.height;
    const int stride = img.stride ? img.stride : w * img.channels;

    std::vector<float> buf(w * h);
    for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
        buf[y * w + x] = img.data[y * stride + x * img.channels];

    std::vector<Feature> candidates;
    std::vector<float>   corner_score(w * h, 0.f);

    for (int y = 3; y < h - 3; ++y)
    for (int x = 3; x < w - 3; ++x)
    {
        float Ip = buf[y * w + x];
        float t  = threshold < 1.f ? threshold * Ip : threshold;

        // Fast early-exit: 3 of 4 cardinal points must pass
        int bright = 0, dark = 0;
        for (int i = 0; i < 4; ++i)
        {
            float v = buf[(y + kCrossY[i]) * w + (x + kCrossX[i])];
            if (v >= Ip + t) ++bright;
            if (v <= Ip - t) ++dark;
        }
        if (bright < 3 && dark < 3) continue;

        // Full circle: N consecutive pixels brighter or darker
        int b_cnt = 0, d_cnt = 0;
        int b_max = 0, d_max = 0;
        for (int i = 0; i < 16 * 2; ++i)
        {
            float v = buf[(y + kCircleY[i % 16]) * w + (x + kCircleX[i % 16])];
            if (v >= Ip + t) { ++b_cnt; b_max = std::max(b_max, b_cnt); }
            else b_cnt = 0;
            if (v <= Ip - t) { ++d_cnt; d_max = std::max(d_max, d_cnt); }
            else d_cnt = 0;
        }
        if (b_max < N && d_max < N) continue;

        float score = 0.f;
        for (int i = 0; i < 16; ++i)
            score += std::fabs(Ip - buf[(y + kCircleY[i]) * w + (x + kCircleX[i])]);

        Feature f;
        f.x     = static_cast<float>(x);
        f.y     = static_cast<float>(y);
        f.score = score;
        candidates.push_back(f);
        corner_score[y * w + x] = score;
    }

    // Non-maximum suppression
    std::vector<Feature> result;
    if (nms_window == 0)
    {
        result = std::move(candidates);
    }
    else
    {
        result.reserve(candidates.size());
        for (const auto& f : candidates)
        {
            int   x    = static_cast<int>(f.x);
            int   y    = static_cast<int>(f.y);
            int   bx   = x, by = y;
            float best = 0.f;

            for (int dy = -nms_window; dy <= nms_window; ++dy)
            for (int dx = -nms_window; dx <= nms_window; ++dx)
            {
                int nx = x + dx, ny = y + dy;
                if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                float s = corner_score[ny * w + nx];
                if (s > best) { best = s; bx = nx; by = ny; }
            }

            auto it = std::find_if(result.begin(), result.end(),
                [bx, by](const Feature& k) {
                    return static_cast<int>(k.x) == bx && static_cast<int>(k.y) == by;
                });
            if (it == result.end())
            {
                Feature winner;
                winner.x     = static_cast<float>(bx);
                winner.y     = static_cast<float>(by);
                winner.score = best;
                result.push_back(winner);
            }
        }
    }

    // Keep top-N by score
    if (max_features > 0 && static_cast<int>(result.size()) > max_features)
    {
        std::partial_sort(result.begin(),
                          result.begin() + max_features,
                          result.end(),
                          [](const Feature& a, const Feature& b) {
                              return a.score > b.score;
                          });
        result.resize(max_features);
    }

    FeatureSet fs;
    fs.features = std::move(result);
    return fs;
}

// ---------------------------------------------------------------------------
// FastFeatureExtractor
// ---------------------------------------------------------------------------

FastFeatureExtractor::FastFeatureExtractor(const FeatureExtractorOptions& opts)
    : opts_(opts) {}

std::string FastFeatureExtractor::name() const { return "FAST"; }

FeatureSet FastFeatureExtractor::extract(const ImageView& image) const
{
    // score_threshold == 0 → use FAST default of 0.15 (relative to pixel value)
    float threshold = opts_.score_threshold > 0.f ? opts_.score_threshold : 0.15f;
    return run_fast(image, /*N=*/9, threshold, /*nms_window=*/2, opts_.max_features);
}

void FastFeatureExtractor::set_options(const FeatureExtractorOptions& opts)
{
    opts_ = opts;
}

FeatureExtractorOptions FastFeatureExtractor::options() const { return opts_; }

// ---------------------------------------------------------------------------
// HarrisFeatureExtractor (stub — not yet implemented)
// ---------------------------------------------------------------------------

HarrisFeatureExtractor::HarrisFeatureExtractor(const FeatureExtractorOptions& opts)
    : opts_(opts) {}

std::string HarrisFeatureExtractor::name() const { return "Harris"; }

FeatureSet HarrisFeatureExtractor::extract(const ImageView&) const { return {}; }

void HarrisFeatureExtractor::set_options(const FeatureExtractorOptions& opts)
{
    opts_ = opts;
}

FeatureExtractorOptions HarrisFeatureExtractor::options() const { return opts_; }

// ---------------------------------------------------------------------------
// SuperPointFeatureExtractor (stub — not yet implemented)
// ---------------------------------------------------------------------------

SuperPointFeatureExtractor::SuperPointFeatureExtractor(
    const std::string& model_path, const FeatureExtractorOptions& opts)
    : model_path_(model_path), opts_(opts) {}

std::string SuperPointFeatureExtractor::name() const { return "SuperPoint"; }

FeatureSet SuperPointFeatureExtractor::extract(const ImageView&) const { return {}; }

void SuperPointFeatureExtractor::set_options(const FeatureExtractorOptions& opts)
{
    opts_ = opts;
}

FeatureExtractorOptions SuperPointFeatureExtractor::options() const { return opts_; }

// ---------------------------------------------------------------------------
// Factory helpers
// ---------------------------------------------------------------------------

FeatureExtractorPtr make_harris(const FeatureExtractorOptions& opts)
{
    return std::make_shared<HarrisFeatureExtractor>(opts);
}

FeatureExtractorPtr make_fast(const FeatureExtractorOptions& opts)
{
    return std::make_shared<FastFeatureExtractor>(opts);
}

FeatureExtractorPtr make_superpoint(const std::string& model_path,
                                    const FeatureExtractorOptions& opts)
{
    return std::make_shared<SuperPointFeatureExtractor>(model_path, opts);
}

}  // namespace vision
