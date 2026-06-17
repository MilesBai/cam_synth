"""Feature extractor sub-package.

The C++ extension module ``_feat_extractor`` is built separately via CMake.
When available it is imported here and its public symbols are re-exported so
callers only need ``from cam_synth.feat_extractor import HarrisFeatureExtractor``.
"""

try:
    from ._feat_extractor import (  # noqa: F401
        ImageView,
        Feature,
        FeatureSet,
        FeatureExtractorOptions,
        FeatureExtractor,
        HarrisFeatureExtractor,
        FastFeatureExtractor,
        SuperPointFeatureExtractor,
        make_harris,
        make_fast,
        make_superpoint,
    )

    __all__ = [
        "ImageView",
        "Feature",
        "FeatureSet",
        "FeatureExtractorOptions",
        "FeatureExtractor",
        "HarrisFeatureExtractor",
        "FastFeatureExtractor",
        "SuperPointFeatureExtractor",
        "make_harris",
        "make_fast",
        "make_superpoint",
    ]

except ImportError:
    # C++ extension not yet built — package is importable but classes are absent.
    __all__ = []
