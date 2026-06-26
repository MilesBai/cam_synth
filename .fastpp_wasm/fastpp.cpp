#include <cmath>
#include <cstdint>

extern "C" {

// Detects up to `maxPoints` interest points in a 3-channel (RGB) image.
//
// Inputs:
//   w, h          : image width and height in pixels
//   data_pointer  : start of RGB image data (length = w * h * 3, row-major,
//                   3 bytes per pixel: R, G, B)
//   maxPoints     : maximum number of keypoints to return
//   out_pointer   : caller-allocated output buffer. Must hold
//                   maxPoints * 4 floats, i.e. maxPoints * 16 bytes.
//                   Layout per keypoint (4 floats, interleaved):
//                     [0] x        : sub-column position (pixels)
//                     [1] y        : sub-row position (pixels)
//                     [2] angle    : orientation in radians (-PI..PI)
//                     [3] response : corner strength (higher = stronger)
//
// Returns:
//   the number of keypoints actually written (0..maxPoints).
//   Returns a negative value on error (e.g. null pointer, bad size).
int fastpp(int w,
           int h,
           const unsigned char* dataPointer,
           int maxPoints,
           float* outPointer);
}


// Computes Harris corner response and dominant orientation for a single pixel
// in a 1-channel uint8 image buffer (row-major, stride = width).
// Uses a fixed 3x3 Sobel kernel; border required = blockSize/2 + 1.
//
// Returns 0 on success, -1 if (px, py) is too close to the image boundary.
int computeHarrisResponse(
    const unsigned char* src, int width, int height,
    int px, int py,
    int blockSize, double k,
    float& score, float& angle)
{
    int h_block = blockSize / 2;
    int border  = h_block + 1; // 3x3 Sobel needs 1-pixel aperture border

    if (px < border || px >= width - border || py < border || py >= height - border) {
        score = 0.0f;
        angle = 0.0f;
        return -1;
    }

    // Structure tensor accumulated over the blockSize x blockSize window.
    double M00 = 0, M01 = 0, M11 = 0;

    for (int dr = -h_block; dr <= h_block; ++dr) {
        for (int dc = -h_block; dc <= h_block; ++dc) {
            int r = py + dr;
            int c = px + dc;

            // 3x3 Sobel X: [[-1,0,1],[-2,0,2],[-1,0,1]]
            float dx = (float)(
                - src[(r-1)*width + (c-1)] + src[(r-1)*width + (c+1)]
                - 2*src[r*width   + (c-1)] + 2*src[r*width   + (c+1)]
                - src[(r+1)*width + (c-1)] + src[(r+1)*width + (c+1)]
            );

            // 3x3 Sobel Y: [[-1,-2,-1],[0,0,0],[1,2,1]]
            float dy = (float)(
                - src[(r-1)*width + (c-1)] - 2*src[(r-1)*width + c] - src[(r-1)*width + (c+1)]
                + src[(r+1)*width + (c-1)] + 2*src[(r+1)*width + c] + src[(r+1)*width + (c+1)]
            );

            M00 += dx * dx;
            M11 += dy * dy;
            M01 += dx * dy;
        }
    }

    // Harris score: det(M) - k * trace(M)^2
    double det   = M00 * M11 - M01 * M01;
    double trace = M00 + M11;
    score = (float)(det - k * trace * trace);

    // Dominant orientation: eigenvector of the larger eigenvalue
    double u  = (M00 + M11) * 0.5;
    double v  = std::sqrt((M00 - M11) * (M00 - M11) * 0.25 + M01 * M01);
    double l1 = u + v;
    angle = (float)std::atan2(l1 - M00, M01);

    return 0;
}