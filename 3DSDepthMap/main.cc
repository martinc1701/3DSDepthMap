#include <cstdio>
#include <cstdlib>
#include <string>
#include <iomanip>
#include <sstream>

#include "utils.hh"
#include "n3dsvideo.hh"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>


// The type of matrix returned by StereoBM::compute(). I believe this is correct ...
#define DEPTH_MAP_TYPE short

/**
	To try and preserve as much detail as possible, we run the block matching algorithm
	several times with different block sizes. Smaller block sizes can capture fine edge
	details, but fail miserably on low-texture surfaces. Larger block sizes tend to
	handle low-texture surfaces better but have poor resolution. The approach we take
	here is to combine them - i.e. take the result from the smallest block size as long
	as it gives us some information. With sufficient filtering, this should work well.
*/
static const int MAX_MATCHERS = 1;
static const int MATCHER_BLOCK_SIZES[MAX_MATCHERS] = {
	// MUST be odd, and sorted from smallest to largest
	//9,
	//21,
	33,
};

static cv::Ptr<cv::StereoBM> matchers[MAX_MATCHERS];

static void initMatchers() {
	for (int i = 0; i < MAX_MATCHERS; ++i) {
		matchers[i] = cv::StereoBM::create();

		// These settings were infered through trial-and-error by using a simple tool 
		// called StereoBMTunner, with sources available here: 
		// http://blog.martinperis.com/2011/08/opencv-stereo-matching.html

		// The input images are NOISY - filter as much as we can.
		matchers[i]->setPreFilterType(cv::StereoBM::PREFILTER_XSOBEL);
		matchers[i]->setPreFilterCap(63);
		// A larger matching block can get more information, but too large gives no 
		// fine detail.
		matchers[i]->setBlockSize(MATCHER_BLOCK_SIZES[i]);
		// The 3DS cameras are a fair distance apart, so push the images closer together.
		// 48 seems good for objects that are at least 2ft from the cameras. If this is
		// too large, then close objects won't be detected.
		matchers[i]->setMinDisparity(48);
		// A larger disparity range lets us handle deeper scenes, but really crops the
		// edges of the depth image.
		matchers[i]->setNumDisparities(32);
		// This filtering step removes erratic depth values (i.e. salt-and-pepper noise).
		// It's better to remove too much than have inaccurate values ...
		matchers[i]->setTextureThreshold(3000);
	}
}

static cv::Mat computeDisparity(const cv::Mat& left, const cv::Mat& right) {
	cv::Mat disparity, tmp;

	matchers[0]->compute(left, right, disparity);

	// The minimum value corresponds to the "UNKNOWN" measurement. We need to know
	// this for merging, since we assume that each matcher uses a progressively
	// larger block size. We merge disparity values only if the current value for
	// the pixel is UNKNOWN.

	double dispUnknown;
	cv::minMaxIdx(disparity, &dispUnknown);

	for (int i = 1; i < MAX_MATCHERS; ++i) {
		matchers[i]->compute(left, right, tmp);
		double tmpUnknown;
		cv::minMaxIdx(tmp, &tmpUnknown);

		// Merge.
		auto itSrc = tmp.begin<DEPTH_MAP_TYPE>();
		auto itDisp = disparity.begin<DEPTH_MAP_TYPE>();
		for ( ; itSrc != tmp.end<DEPTH_MAP_TYPE>(); ++itSrc, ++itDisp)
			if (*itDisp == dispUnknown && *itSrc != tmpUnknown)
				*itDisp = *itSrc;
	}

	return disparity;
}


int main(int argc, char **argv) {
	if (argc < 2) {
		printf("no video file provided - exiting\n");
		return 0;
	}
	
	// The output path is built from the input filename, minus the file
	// extension.

	std::string inputPath = argv[1];
	std::string outputPath = inputPath;
	size_t lastSlash = outputPath.find_last_of("\\/");
	if (lastSlash != std::string::npos)
		outputPath = outputPath.substr(lastSlash + 1);
	size_t lastDot = outputPath.rfind('.');
	if (lastDot != std::string::npos)
		outputPath = outputPath.substr(0, lastDot);
	
	try {
		// Load the input video.

		N3DSVideo *video = new N3DSVideo(inputPath.c_str(), true, true);
		makeDirectory(outputPath.c_str());
		makeDirectory((outputPath + "/image").c_str());
		makeDirectory((outputPath + "/right").c_str());
		makeDirectory((outputPath + "/depth").c_str());

		printf("Processing video ...\n");

		cv::namedWindow("Diff", cv::WINDOW_AUTOSIZE);
		cv::namedWindow("Disparity", cv::WINDOW_AUTOSIZE);
		cv::namedWindow("Combined", cv::WINDOW_AUTOSIZE);
		
		initMatchers();

		int frame = 0;
		int timeMs = 0;
		int dispMaxi = 1;

		while (video->processStep()) {
			if (!video->hasNewStereoImage()) continue;

			std::ostringstream colourFile;
			std::ostringstream depthFile;
			std::ostringstream rightFile;
			colourFile << outputPath << "/image/" 
				<< std::setw(6) << std::setfill('0') << frame << "-" 
				<< std::setw(6) << std::setfill('0') << timeMs << ".jpg";
			depthFile << outputPath << "/depth/"
				<< std::setw(6) << std::setfill('0') << frame << "-"
				<< std::setw(6) << std::setfill('0') << timeMs << ".pgm";
			rightFile << outputPath << "/right/"
				<< std::setw(6) << std::setfill('0') << frame << "-"
				<< std::setw(6) << std::setfill('0') << timeMs << ".jpg";

			frame++;
			timeMs += 33;

			cv::Mat disparity = computeDisparity(video->leftImage(), video->rightImage());
			
			cv::imwrite(colourFile.str(), video->leftImage());
			cv::imwrite(depthFile.str(), disparity);
			cv::imwrite(rightFile.str(), video->rightImage());

			// Since the depth image is likely to be very dark, rescale it before showing it.
			double mini, maxi;
			cv::minMaxIdx(disparity, &mini, &maxi);
			//printf("%d %f %f %d\n", disparityHiRes.type(), mini, maxi, dispMaxi);
			if (maxi > dispMaxi)
				dispMaxi = maxi;
			double scale = 255.0 / dispMaxi;
			disparity.convertTo(disparity, CV_8UC1, scale, -mini*scale);
			
			cv::imshow("Diff", 0.5 * (video->rightImage() - video->leftImage()) + 127);
			cv::imshow("Disparity", disparity);

			cv::Mat colouredDisparity;
			cv::applyColorMap(disparity, colouredDisparity, cv::COLORMAP_HOT);
			cv::Mat left;
			cv::cvtColor(video->leftImage(), left, cv::COLOR_GRAY2BGR);

			cv::imshow("Combined", left + colouredDisparity);

			cv::waitKey(33);
		}

		printf("... done.\n");
		delete video;
	}
	catch (const std::exception& ex) {
		printf("an error occured: %s\n", ex.what());
	}
	catch (...) {
		printf("an unknown error occured");
	}

	return 0;
}
