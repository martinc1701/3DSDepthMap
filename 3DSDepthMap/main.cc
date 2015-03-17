/**
	3DS Depth Map generation.

	Currently, this takes a single argument (the .AVI video file recorded by the 3DS)
	and converts it into a sequence of images stored in a folder with the same name as
	the file. There are two sets of images: the view from the left camera, and the
	depth (disparity) map corresponding to it.

	We assume that "no information" is better than "false information"; thus, we
	rigorously filter the results until we have something satisfactory, although it
	may not have much actual depth information. This is something that needs to be
	experimented with.

	The current algorithm is as follows. We first generate the depth map using OpenCV's
	StereoBM (block matcher) algorithm, configured to perform both pre- and post-
	filtering to remove noise. This works well (especially if the block size is 
	reasonably large), but suffers from "ballooning" - depth values for a foreground
	object tend to be duplicated around the silouette of the object as well.
	
	To remedy this, we use the following intuition: any significant change in depth
	should occur on the edge of an object. We apply this idea by running an edge 
	detection pass on both the depth and colour images, and then for each pixel row
	of the depth image, if we find a depth-edge, then we fill the depth image to the
	left/right of this edge with the "unknown" depth value until we come across a
	colour edge. Since there is inevitably noise edges, we apply a median filter to
	the result to remove any thin lines that were left behind.
*/

#include <cstdio>
#include <cstdlib>
#include <string>
#include <iomanip>
#include <sstream>

#include "utils.hh"
#include "n3dsvideo.hh"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>


//
// Hard-coded constants (for now ...)
//

// Block size for matching. Larger is slower, and tends to be less accurate, but
// can find matches on less textured surfaces. MUST be odd.
static const int MATCHER_BLOCK_SIZE = 21; 

// Edge detection thresholds for "deflating" the depth values. We want the colour
// threshold to be low, and the depth threshold to be high.
static const int COLOUR_EDGE_THRESHOLD = 5;
static const int DEPTH_EDGE_THRESHOLD = 150;


static cv::Ptr<cv::StereoBM> matcher;

static void initMatcher() {
	matcher = cv::StereoBM::create();

	// These settings were infered through trial-and-error by using a simple tool 
	// called StereoBMTunner, with sources available here: 
	// http://blog.martinperis.com/2011/08/opencv-stereo-matching.html
	
	// The input images are NOISY - filter as much as we can.
	matcher->setPreFilterType(cv::StereoBM::PREFILTER_XSOBEL);
	matcher->setPreFilterCap(63);
	// A larger matching block can get more information, but too large gives no 
	// fine detail.
	matcher->setBlockSize(MATCHER_BLOCK_SIZE);
	// The 3DS cameras are a fair distance apart, so push the images closer together.
	// 48 seems good for objects that are at least 2ft from the cameras. If this is
	// too large, then close objects won't be detected.
	matcher->setMinDisparity(45);
	// A larger disparity range lets us handle deeper scenes, but really crops the
	// edges of the depth image.
	matcher->setNumDisparities(32);
	// This filtering step removes erratic depth values (i.e. salt-and-pepper noise).
	// It's better to remove too much than have inaccurate values ...
	matcher->setTextureThreshold(3000);
}


static cv::Mat computeDisparity(const cv::Mat& left, const cv::Mat& right) {
	cv::Mat disparity, tmp;

	matcher->compute(left, right, disparity);
	// For whatever reason, compute() gives us signed values ... likely a bug ...
	disparity.convertTo(disparity, CV_16UC1);

	//
	// Deal with the "ballooning" effect.
	//

	// The minimum value corresponds to the "UNKNOWN" measurement.
	double dispUnknown, dispMaxi;
	cv::minMaxIdx(disparity, &dispUnknown, &dispMaxi);

	cv::Mat colourEdges, disparityEdges;

	// For the colour edges, blur first to remove noise.
	cv::blur(left, tmp, cv::Size(7,7));
	cv::Canny(tmp, colourEdges, COLOUR_EDGE_THRESHOLD, 3 * COLOUR_EDGE_THRESHOLD);

	// For the disparity edges, rescale to 8-bit range, and use a slight blur.
	double scale = 255.0 / (dispMaxi - dispUnknown + 1);
	disparity.convertTo(tmp, CV_8U, scale, -dispUnknown * scale);
	cv::blur(tmp, tmp, cv::Size(3, 3));
	cv::Canny(tmp, disparityEdges, DEPTH_EDGE_THRESHOLD, 3 * DEPTH_EDGE_THRESHOLD);

	//cv::imshow("colourEdges", colourEdges);
	//cv::imshow("disparityEdges", disparityEdges);

	// Search for depth edges and force them to coincide with colour edges.
	for (int i = 0; i < colourEdges.rows; ++i) {
		auto *cEdgeRow = colourEdges.ptr<uchar>(i);
		auto *dEdgeRow = disparityEdges.ptr<uchar>(i);
		auto *dst = disparity.ptr<ushort>(i);
		
		bool onEdge = false;
		for (int j = 0; j < colourEdges.cols; ++j) {
			if (!onEdge && dEdgeRow[j] > 0) {
				if (cEdgeRow[j] == 0)
					dst[j] = dispUnknown;
				for (int k = j - 1; k >= 0 && dEdgeRow[k] == 0 && cEdgeRow[k] == 0; --k)
					dst[k] = dispUnknown;
				onEdge = true;
			}
			else if (onEdge && dEdgeRow[j] == 0) {
				for (int k = j; k < colourEdges.cols && dEdgeRow[k] == 0 && cEdgeRow[k] == 0; ++k)
					dst[k] = dispUnknown;
				onEdge = false;
			}
		}
	}

	// Since we only do the above loop in 1 dimension, we may have thin lines due to noise.
	// Remove these with a median filter (we do NOT want averages here ...)
	cv::medianBlur(disparity, disparity, 5);

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
		//makeDirectory((outputPath + "/right").c_str());
		makeDirectory((outputPath + "/depth").c_str());

		printf("Processing video ...\n");

		cv::namedWindow("Diff", cv::WINDOW_AUTOSIZE);
		cv::namedWindow("Disparity", cv::WINDOW_AUTOSIZE);
		cv::namedWindow("Combined", cv::WINDOW_AUTOSIZE);
		
		initMatcher();

		int frame = 0;
		int timeMs = 0;
		int dispMaxi = 1;

		while (video->processStep()) {
			if (!video->hasNewStereoImage()) continue;

			std::ostringstream colourFile;
			std::ostringstream depthFile;
			//std::ostringstream rightFile;
			colourFile << outputPath << "/image/" 
				<< std::setw(6) << std::setfill('0') << frame << "-" 
				<< std::setw(6) << std::setfill('0') << timeMs << ".jpg";
			depthFile << outputPath << "/depth/"
				<< std::setw(6) << std::setfill('0') << frame << "-"
				<< std::setw(6) << std::setfill('0') << timeMs << ".png";
			//rightFile << outputPath << "/right/"
			//	<< std::setw(6) << std::setfill('0') << frame << "-"
			//	<< std::setw(6) << std::setfill('0') << timeMs << ".jpg";

			frame++;
			timeMs += 33;

			cv::Mat disparity = computeDisparity(video->leftImage(), video->rightImage());
			
			cv::imwrite(colourFile.str(), video->leftImage());
			cv::imwrite(depthFile.str(), disparity);
			//cv::imwrite(rightFile.str(), video->rightImage());

			// Since the depth image is likely to be very dark, rescale it before showing it.
			double mini, maxi;
			cv::minMaxIdx(disparity, &mini, &maxi);
			if (maxi > dispMaxi)
				dispMaxi = maxi;
			double scale = 255.0 / dispMaxi;
			disparity.convertTo(disparity, CV_8UC1, scale, -mini*scale);

			if (frame == 1) {
				printf("Min disparity   = %d\n"
					   "Num disparities = %d\n"
					   "Min pixel value = %f\n"
					   "Divide pixel values by 16 to get the disparity. < 1 is no match/unknown.\n",
					   matcher->getMinDisparity(), matcher->getNumDisparities(), mini);
			}
			
			cv::imshow("Diff", 0.5 * (video->rightImage() - video->leftImage()) + 127);
			
			cv::Mat colouredDisparity;
			cv::applyColorMap(disparity, colouredDisparity, cv::COLORMAP_JET);

			cv::imshow("Disparity", colouredDisparity);

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
