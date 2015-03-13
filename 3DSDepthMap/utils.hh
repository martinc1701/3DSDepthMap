#ifndef UTILS_HH
#define UTILS_HH

#include <opencv2/core.hpp>
extern "C" {
#include <libavutil/frame.h>
}

/**
   Makes the given directory. It is NOT recursive.
*/
void makeDirectory(const char *name);

/**
   Converts the given video frame, which must be in YUV420 format and have
   the given width/height, to OpenCV's BGR format. The result will be a
   (3*w) x h matrix, created appropriately.
*/
void convertYUV420ToRGB(AVFrame *frame, int w, int h, cv::Mat &res);

void convertYUV420ToY(AVFrame *frame, int w, int h, cv::Mat &res);


#endif
