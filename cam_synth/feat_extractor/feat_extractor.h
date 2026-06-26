#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace vision
{

struct ImageView
{
    const uint8_t* data = nullptr;
    int width           = 0;
    int height          = 0;
    int channels        = 1;
    int stride          = 0;  // bytes per row; 0 means width * channels
};

struct Feature
{
    float x           = 0.0f;
    float y           = 0.0f;
    float score       = 0.0f;
    float scale       = 1.0f;
    float orientation = 0.0f;
};

struct FeatureSet
{
    std::vector<Feature> features;
};

struct FeatureExtractorOptions
{
    int   max_features        = 1000;
    float score_threshold     = 0.0f;
    bool  compute_orientation = false;
};

class FeatureExtractor
{
public:
    virtual ~FeatureExtractor() = default;

    virtual std::string name() const = 0;

    virtual FeatureSet extract(const ImageView& image) const = 0;

    virtual void set_options(const FeatureExtractorOptions& options) = 0;

    virtual FeatureExtractorOptions options() const = 0;
};

using FeatureExtractorPtr = std::shared_ptr<FeatureExtractor>;

// ---------------------------------------------------------------------------
// Concrete extractors
// ---------------------------------------------------------------------------

class HarrisFeatureExtractor : public FeatureExtractor
{
public:
    explicit HarrisFeatureExtractor(
        const FeatureExtractorOptions& opts = {});

    std::string             name() const override;
    FeatureSet              extract(const ImageView& image) const override;
    void                    set_options(const FeatureExtractorOptions& opts) override;
    FeatureExtractorOptions options() const override;

private:
    FeatureExtractorOptions opts_;
};

class FastFeatureExtractor : public FeatureExtractor
{
public:
    explicit FastFeatureExtractor(
        const FeatureExtractorOptions& opts = {});

    std::string             name() const override;
    FeatureSet              extract(const ImageView& image) const override;
    void                    set_options(const FeatureExtractorOptions& opts) override;
    FeatureExtractorOptions options() const override;

private:
    FeatureExtractorOptions opts_;
};

class SuperPointFeatureExtractor : public FeatureExtractor
{
public:
    explicit SuperPointFeatureExtractor(
        const std::string&             model_path,
        const FeatureExtractorOptions& opts = {});

    std::string             name() const override;
    FeatureSet              extract(const ImageView& image) const override;
    void                    set_options(const FeatureExtractorOptions& opts) override;
    FeatureExtractorOptions options() const override;

private:
    std::string             model_path_;
    FeatureExtractorOptions opts_;
};

// Factory helpers
FeatureExtractorPtr make_harris(const FeatureExtractorOptions& opts = {});
FeatureExtractorPtr make_fast(const FeatureExtractorOptions& opts = {});
FeatureExtractorPtr make_superpoint(
    const std::string&             model_path,
    const FeatureExtractorOptions& opts = {});

}  // namespace vision
