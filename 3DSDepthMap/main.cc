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
#include <fstream>

#include "utils.hh"
#include "n3dsvideo.hh"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#define USE_STEREO_SGBM 1

//
// Hard-coded constants (for now ...)
//

// Block size for matching. Larger is slower, and tends to be less accurate, but
// can find matches on less textured surfaces. MUST be odd.
#if !USE_STEREO_SGBM
static const int MATCHER_BLOCK_SIZE = 21; // 21 
#else
static const int MATCHER_BLOCK_SIZE = 7;
#endif

// The 3DS cameras are a fair distance apart, so we need a suitable minimum distance
// for the block matching. 48 seems good for objects that are at least 2ft from the
// cameras. If this is too large, then close objects won't be detected; too small and
// far objects won't be detected.
static const int MIN_DISPARITY = 45; // 45

// Edge detection thresholds for "deflating" the depth values. We want the colour
// threshold to be low, and the depth threshold to be high.
static const int COLOUR_EDGE_THRESHOLD = 5;
static const int DEPTH_EDGE_THRESHOLD = 150;

// The distance between the cameras, in metres
static const double N3DSXL_CAM_DIST = 0.035;

// The camera focal length, in pixels
static const double N3DSXL_FOCAL_LEN = 565.0;

// The 3DS cameras aren't perfectly aligned; the centre rays seem to converge at a
// point ~25cm in front of the cameras. We can still approximate the depth in this
// case. This value is in metres.
static const double N3DSXL_CONVERGENCE = 0.25;

#if !USE_STEREO_SGBM
static cv::Ptr<cv::StereoBM> matcher;
#else
static cv::Ptr<cv::StereoSGBM> matcher;
#endif

static void initMatcher() {
#if !USE_STEREO_SGBM
	matcher = cv::StereoBM::create();

	// These settings were infered through trial-and-error by using a simple tool 
	// called StereoBMTunner, with sources available here: 
	// http://blog.martinperis.com/2011/08/opencv-stereo-matching.html
	
	// The input images are NOISY - filter as much as we can.
	matcher->setPreFilterType(cv::StereoBM::PREFILTER_XSOBEL);
	matcher->setPreFilterCap(63);
	
	matcher->setBlockSize(MATCHER_BLOCK_SIZE);
	matcher->setMinDisparity(MIN_DISPARITY);

	// A larger disparity range lets us handle deeper scenes, but really crops the
	// edges of the depth image.
	matcher->setNumDisparities(32);
	// This filtering step removes erratic depth values (i.e. salt-and-pepper noise).
	// It's better to remove too much than have inaccurate values ...
	matcher->setTextureThreshold(3000);
#else
	matcher = cv::StereoSGBM::create(MIN_DISPARITY, 32, MATCHER_BLOCK_SIZE,
									 8 * MATCHER_BLOCK_SIZE*MATCHER_BLOCK_SIZE,
									 32 * MATCHER_BLOCK_SIZE*MATCHER_BLOCK_SIZE);
	// The input images are NOISY - filter as much as we can.
	matcher->setPreFilterCap(1);
	matcher->setUniquenessRatio(5);
	matcher->setSpeckleWindowSize(250);
	matcher->setSpeckleRange(1);
	//matcher->setMode(true);
#endif
}


static cv::Mat computeDepth(const cv::Mat& left, const cv::Mat& right) {
	cv::Mat disparity, tmp;

	matcher->compute(left, right, disparity);
	// For whatever reason, compute() gives us signed values ... likely a bug ...
	disparity.convertTo(disparity, CV_16UC1);

	// The minimum value corresponds to the "UNKNOWN" measurement.
	double dispUnknown, dispMaxi;
	cv::minMaxIdx(disparity, &dispUnknown, &dispMaxi);

#if !USE_STEREO_SGBM
	//
	// Deal with the "ballooning" effect.
	//

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

	// Search for disparity edges and force them to coincide with colour edges.
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
#endif
	
	// Convert the disparity to a Kinect-style depth image. That is, we compute the depth to mm
	// precision, then convert to some strange fixed-point format (reference: SiftFu.m:383).
	for (int i = 0; i < disparity.rows; ++i) {
		auto *row = disparity.ptr<ushort>(i);
		for (int j = 0; j < disparity.cols; ++j) {
			if (row[j] == dispUnknown)
				row[j] = 0;
			else {
				// The disparity values are in 12:4 fixed point format, so be careful ...
				double disp = (double)row[j] / 16.0;
				double depth = 1000.0 * abs(
					N3DSXL_CAM_DIST / ((N3DSXL_CAM_DIST / N3DSXL_CONVERGENCE) - (disp / N3DSXL_FOCAL_LEN)));
				//printf("%f\n", depth);
				if (depth < 65535)
					row[j] = (ushort)depth;
				else
					row[j] = 0;
				//row[j] = (d << 3) | (d >> 13); //???
			}
		}
	}

	return disparity;
}



static bool quiet = false;
static bool saveRaw = false;
static bool noDepth = false;
static std::string inputPath = "";

static bool parseArgs(int argc, char **argv) {
	for (int i = 1; i < argc; ++i) {
		if (_stricmp(argv[i], "--quiet") == 0)
			quiet = true;
		else if (_stricmp(argv[i], "--saveRaw") == 0)
			saveRaw = true;
		else if (_stricmp(argv[i], "--noDepth") == 0)
			noDepth = true;
		else if (argv[i][0] == '-') {
			if (_stricmp(argv[i], "--help") != 0)
				printf("Unknown option '%s'\n", argv[i]);
			printf("Valid arguments: [--quiet] [--saveRaw] [--noDepth] [--help] FILENAME.AVI\n\n"
				   "Synopsis:\n"
				   "  This program converts a video recorded by the Nintendo 3DS video app to depth\n"
				   "  images for 3D reconstruction applications. The quality of the depth images is\n"
				   "  directly related to how much detail is present in the images; if insufficient\n"
				   "  detail is present, the reconstructed depth will have a large number of unknown\n"
				   "  areas in it.\n\n"
				   "Options:\n"
				   "  --quiet           Don't display processed images as they are computed\n"
				   "  --saveRaw         Save the left/right camera images\n"
				   "  --noDepth         Don't compute depth maps\n"
				   "  --help            Show this help text\n");
			return false;
		}
		else
			inputPath = argv[i];
	}

	if (inputPath == "") {
		printf("No video file provided\n");
		return false;
	}

	return true;
}


int main(int argc, char **argv) {
	if (!parseArgs(argc, argv))
		return 0;
	
	// The output path is built from the input filename, minus the file
	// extension.

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
		N3DSVideo *rgbVideo = new N3DSVideo(inputPath.c_str(), false, true);
		makeDirectory(outputPath.c_str());
		makeDirectory((outputPath + "/raw").c_str());
		makeDirectory((outputPath + "/image").c_str());
		makeDirectory((outputPath + "/depth").c_str());

		printf("Processing video ...\n");
		
		if (!quiet) {
			cv::namedWindow("Diff", cv::WINDOW_AUTOSIZE);
			cv::namedWindow("Disparity", cv::WINDOW_AUTOSIZE);
			cv::namedWindow("Combined", cv::WINDOW_AUTOSIZE);
		}
		
		initMatcher();

		int frame = 0;
		int timeMs = 0;
		int dMaxi = 1;

		while (video->processStep() && rgbVideo->processStep()) {
			if (!video->hasNewStereoImage()) continue;

			std::ostringstream filename;
			filename << std::setw(6) << std::setfill('0') << frame << "-"
				<< std::setw(6) << std::setfill('0') << timeMs;

			std::ostringstream colourFile;
			std::ostringstream depthFile;
			std::ostringstream rawLFile, rawRFile;
			colourFile << outputPath << "/image/" << filename.str() << ".jpg";
			depthFile << outputPath << "/depth/" << filename.str() << ".png";
			rawLFile << outputPath << "/raw/" << filename.str() << "L.jpg";
			rawRFile << outputPath << "/raw/" << filename.str() << "R.jpg";

			frame++;
			timeMs += 50; // 3DS video is 20fps

			if (saveRaw) {
				cv::imwrite(rawLFile.str(), rgbVideo->leftImage());
				cv::imwrite(rawRFile.str(), rgbVideo->rightImage());
			}

			if (noDepth) continue;
			
			cv::Mat depth = computeDepth(video->leftImage(), video->rightImage());
			
			// Write the two images - left camera and depth. However, for testing we want
			// the output here to look like it came from the Kinect - that means we need to
			// crop/rescale the images to 640x480.

			double imScale = 480.0 / depth.rows;

			cv::Rect region(depth.cols/2 - 320.0 / imScale, 0, 
							640.0 / imScale, depth.rows);
			cv::Mat rescaledDepth, rescaledLeft;
			cv::resize(depth(region), rescaledDepth, cv::Size(640, 480));
			cv::resize(rgbVideo->leftImage()(region), rescaledLeft, cv::Size(640, 480));
			
			cv::imwrite(colourFile.str(), rescaledLeft);
			cv::imwrite(depthFile.str(), rescaledDepth);

			if (frame == 1) {
				std::ofstream intrinsics(outputPath + "/intrinsics.txt");
				intrinsics << N3DSXL_FOCAL_LEN / imScale << " 0 320\n0 " 
				           << N3DSXL_FOCAL_LEN / imScale << " 240\n0 0 1\n";
				intrinsics.close();
			}

			if (!quiet) {
				// Since the depth image is likely to be very dark, rescale it before showing it.
				double mini, maxi;
				cv::minMaxIdx(depth, &mini, &maxi);
				if (maxi > dMaxi)
					dMaxi = maxi;
				double scale = 255.0 / dMaxi;
				depth.convertTo(depth, CV_8UC1, scale);

				cv::imshow("Diff", 0.5 * (video->rightImage() - video->leftImage()) + 127);

				cv::Mat colouredDepth;
				cv::applyColorMap(depth, colouredDepth, cv::COLORMAP_JET);

				cv::imshow("Disparity", colouredDepth);

				cv::Mat left;
				cv::cvtColor(video->leftImage(), left, cv::COLOR_GRAY2BGR);

				cv::imshow("Combined", left + colouredDepth);

				cv::waitKey(33);
			}
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
