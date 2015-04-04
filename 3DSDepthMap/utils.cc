#include "utils.hh"
#include <opencv2/opencv.hpp>
#ifdef __linux__
#include <unistd.h>
#else
#include <direct.h>
#endif


void makeDirectory(const char *name) {
#ifdef __linux__
	mkdir(name, 0755);
#else
	_mkdir(name);
#endif
}


void convertYUV420ToRGB(AVFrame *frame, int w, int h, cv::Mat &res) {
	res.create(cv::Size(w, h), CV_8UC3);

	// Expand the 420 format to normal 444. We use a bit of loop
	// unrolling here for speed, so we need to ensure that the width of
	// the image is a multiple of 8 (almost always the case ...)

	const int w8 = w & ~7;
	const uint8_t *ySrc = frame->data[0];
	const uint8_t *uSrc = frame->data[1];
	const uint8_t *vSrc = frame->data[2];

	for (int i = 0; i < h; ++i) {
		uint8_t *dest = res.ptr(i);
		for (int j = 0; j < w8 / 8; ++j) {
#define LOOP { 	*(dest++) = *(ySrc++);			\
				*(dest++) = *vSrc;				\
				*(dest++) = *uSrc;				\
				*(dest++) = *(ySrc++);			\
				*(dest++) = *(vSrc++);			\
				*(dest++) = *(uSrc++);  }
			LOOP;
			LOOP;
			LOOP;
			LOOP;
#undef LOOP
		}
		ySrc += frame->linesize[0] - w8;
		if (i & 1) { // Move on to next U/V row?
			uSrc += frame->linesize[1] - w8 / 2;
			vSrc += frame->linesize[2] - w8 / 2;
		}
		else { // Stay on current row.
			uSrc -= w8 / 2;
			vSrc -= w8 / 2;
		}
	}

	// Finally, convert the YUV values to BGR.
	cv::cvtColor(res, res, cv::COLOR_YCrCb2BGR);
}


void convertYUV420ToY(AVFrame *frame, int w, int h, cv::Mat &res) {
	res.create(cv::Size(w, h), CV_8UC1);

	const int w8 = w & ~7;
	const uint8_t *ySrc = frame->data[0];

	for (int i = 0; i < h; ++i) {
		uint8_t *dest = res.ptr(i);
		for (int j = 0; j < w8 / 8; ++j) {
#define LOOP  	*(dest++) = *(ySrc++);
			LOOP;
			LOOP;
			LOOP;
			LOOP;
			LOOP;
			LOOP;
			LOOP;
			LOOP;
#undef LOOP
		}
		ySrc += frame->linesize[0] - w8;
	}
}