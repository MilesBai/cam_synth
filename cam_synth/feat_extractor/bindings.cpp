// pybind11 bindings for vision::FeatureExtractor hierarchy.
// Build via CMakeLists.txt in this directory.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "feat_extractor.h"

namespace py = pybind11;
using namespace vision;

// ---------------------------------------------------------------------------
// Trampoline — lets Python subclass FeatureExtractor
// ---------------------------------------------------------------------------
class PyFeatureExtractor : public FeatureExtractor
{
public:
    using FeatureExtractor::FeatureExtractor;

    std::string name() const override
    {
        PYBIND11_OVERRIDE_PURE(std::string, FeatureExtractor, name);
    }

    FeatureSet extract(const ImageView& image) const override
    {
        PYBIND11_OVERRIDE_PURE(FeatureSet, FeatureExtractor, extract, image);
    }

    void set_options(const FeatureExtractorOptions& opts) override
    {
        PYBIND11_OVERRIDE_PURE(void, FeatureExtractor, set_options, opts);
    }

    FeatureExtractorOptions options() const override
    {
        PYBIND11_OVERRIDE_PURE(FeatureExtractorOptions, FeatureExtractor, options);
    }
};

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_feat_extractor, m)
{
    m.doc() = "Feature extractor C++ extension module";

    // -- ImageView -----------------------------------------------------------
    py::class_<ImageView>(m, "ImageView")
        .def(py::init<>())
        .def_readwrite("width",    &ImageView::width)
        .def_readwrite("height",   &ImageView::height)
        .def_readwrite("channels", &ImageView::channels)
        .def_readwrite("stride",   &ImageView::stride)
        // `data` is a raw pointer; expose as read-only int for debugging only.
        .def_property_readonly("data_ptr",
            [](const ImageView& v) { return reinterpret_cast<std::uintptr_t>(v.data); })
        .def("__repr__", [](const ImageView& v) {
            return "<ImageView " + std::to_string(v.width) + "x" +
                   std::to_string(v.height) + "x" + std::to_string(v.channels) + ">";
        });

    // -- Feature -------------------------------------------------------------
    py::class_<Feature>(m, "Feature")
        .def(py::init<>())
        .def_readwrite("x",           &Feature::x)
        .def_readwrite("y",           &Feature::y)
        .def_readwrite("score",       &Feature::score)
        .def_readwrite("scale",       &Feature::scale)
        .def_readwrite("orientation", &Feature::orientation)
        .def("__repr__", [](const Feature& f) {
            return "<Feature (" + std::to_string(f.x) + ", " + std::to_string(f.y) +
                   ") score=" + std::to_string(f.score) + ">";
        });

    // -- FeatureSet ----------------------------------------------------------
    py::class_<FeatureSet>(m, "FeatureSet")
        .def(py::init<>())
        .def_readwrite("features", &FeatureSet::features)
        .def("__len__", [](const FeatureSet& fs) { return fs.features.size(); })
        .def("__repr__", [](const FeatureSet& fs) {
            return "<FeatureSet n=" + std::to_string(fs.features.size()) + ">";
        });

    // -- FeatureExtractorOptions ---------------------------------------------
    py::class_<FeatureExtractorOptions>(m, "FeatureExtractorOptions")
        .def(py::init<>())
        .def_readwrite("max_features",        &FeatureExtractorOptions::max_features)
        .def_readwrite("score_threshold",     &FeatureExtractorOptions::score_threshold)
        .def_readwrite("compute_orientation", &FeatureExtractorOptions::compute_orientation)
        .def("__repr__", [](const FeatureExtractorOptions& o) {
            return "<FeatureExtractorOptions max=" + std::to_string(o.max_features) +
                   " thresh=" + std::to_string(o.score_threshold) + ">";
        });

    // -- FeatureExtractor (abstract base) ------------------------------------
    py::class_<FeatureExtractor, PyFeatureExtractor,
               std::shared_ptr<FeatureExtractor>>(m, "FeatureExtractor")
        .def(py::init<>())
        .def("name",        &FeatureExtractor::name)
        .def("extract",     &FeatureExtractor::extract,     py::arg("image"))
        .def("set_options", &FeatureExtractor::set_options, py::arg("options"))
        .def("options",     &FeatureExtractor::options);

    // -- HarrisFeatureExtractor ----------------------------------------------
    py::class_<HarrisFeatureExtractor,
               FeatureExtractor,
               std::shared_ptr<HarrisFeatureExtractor>>(m, "HarrisFeatureExtractor")
        .def(py::init<const FeatureExtractorOptions&>(),
             py::arg("options") = FeatureExtractorOptions{})
        .def("name",        &HarrisFeatureExtractor::name)
        .def("extract",     &HarrisFeatureExtractor::extract,     py::arg("image"))
        .def("set_options", &HarrisFeatureExtractor::set_options, py::arg("options"))
        .def("options",     &HarrisFeatureExtractor::options);

    // -- FastFeatureExtractor ------------------------------------------------
    py::class_<FastFeatureExtractor,
               FeatureExtractor,
               std::shared_ptr<FastFeatureExtractor>>(m, "FastFeatureExtractor")
        .def(py::init<const FeatureExtractorOptions&>(),
             py::arg("options") = FeatureExtractorOptions{})
        .def("name",        &FastFeatureExtractor::name)
        .def("extract",     &FastFeatureExtractor::extract,     py::arg("image"))
        .def("set_options", &FastFeatureExtractor::set_options, py::arg("options"))
        .def("options",     &FastFeatureExtractor::options);

    // -- SuperPointFeatureExtractor ------------------------------------------
    py::class_<SuperPointFeatureExtractor,
               FeatureExtractor,
               std::shared_ptr<SuperPointFeatureExtractor>>(m, "SuperPointFeatureExtractor")
        .def(py::init<const std::string&, const FeatureExtractorOptions&>(),
             py::arg("model_path"),
             py::arg("options") = FeatureExtractorOptions{})
        .def("name",        &SuperPointFeatureExtractor::name)
        .def("extract",     &SuperPointFeatureExtractor::extract,     py::arg("image"))
        .def("set_options", &SuperPointFeatureExtractor::set_options, py::arg("options"))
        .def("options",     &SuperPointFeatureExtractor::options);

    // -- Factory helpers -----------------------------------------------------
    m.def("make_harris",
          &make_harris,
          py::arg("options") = FeatureExtractorOptions{});

    m.def("make_fast",
          &make_fast,
          py::arg("options") = FeatureExtractorOptions{});

    m.def("make_superpoint",
          &make_superpoint,
          py::arg("model_path"),
          py::arg("options") = FeatureExtractorOptions{});
}
