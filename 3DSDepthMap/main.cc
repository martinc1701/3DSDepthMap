#include <cstdio>
#include <cstdlib>
#include <string>
#include <iomanip>
#include <sstream>

#include "utils.hh"
#include "n3dsvideo.hh"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>


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

		N3DSVideo *video = new N3DSVideo(inputPath.c_str(), true);
		makeDirectory(outputPath.c_str());
		makeDirectory((outputPath + "/image").c_str());
		makeDirectory((outputPath + "/right").c_str());
		makeDirectory((outputPath + "/depth").c_str());

		printf("Processing video ...\n");

		cv::namedWindow("Diff", cv::WINDOW_AUTOSIZE);
		cv::namedWindow("Disparity", cv::WINDOW_AUTOSIZE);
		
		auto stereoLoRes = cv::StereoBM::create();
		auto stereoHiRes = cv::StereoBM::create();

		// These settings were infered through trial-and-error by using a simple tool 
		// called StereoBMTunner, with sources available here: 
		// http://blog.martinperis.com/2011/08/opencv-stereo-matching.html
		
		// The input images are NOISY - filter as much as we can.
		stereoLoRes->setPreFilterType(stereoLoRes->PREFILTER_XSOBEL);
		stereoLoRes->setPreFilterCap(63);
		stereoHiRes->setPreFilterType(stereoHiRes->PREFILTER_XSOBEL);
		stereoHiRes->setPreFilterCap(63);
		// A larger matching block can get more information, but too large gives no 
		// fine detail.
		stereoLoRes->setBlockSize(21);
		stereoHiRes->setBlockSize(9);
		// The 3DS cameras are a fair distance apart, so push the images closer together.
		// 48 seems good for objects that are at least 2ft from the cameras. If this is
		// too large, then close objects won't be detected.
		stereoLoRes->setMinDisparity(48);
		stereoHiRes->setMinDisparity(48);
		// A larger disparity range lets us handle deeper scenes, but will crop the depth
		// map and reduce distance accuracy (since the map is stored as 8-bit values).
		// 32 seems to be good for average size rooms.
		stereoLoRes->setNumDisparities(32);
		stereoHiRes->setNumDisparities(32);
		// This filtering step removes erratic depth values (i.e. salt-and-pepper noise).
		// It's better to remove too much than have inaccurate values ...
		stereoLoRes->setTextureThreshold(3000);
		stereoHiRes->setTextureThreshold(3000);

		int frame = 0;
		int timeMs = 0;

		const int downsample = 2;
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
				<< std::setw(6) << std::setfill('0') << timeMs << ".png";
			rightFile << outputPath << "/right/"
				<< std::setw(6) << std::setfill('0') << frame << "-"
				<< std::setw(6) << std::setfill('0') << timeMs << ".jpg";

			frame++;
			timeMs += 33;

			cv::Mat disparityLoRes;
			cv::Mat disparityHiRes;
			stereoLoRes->compute(video->rightImage(), video->leftImage(), disparityLoRes);
			stereoHiRes->compute(video->rightImage(), video->leftImage(), disparityHiRes);

			// Merge the high-res and low-res disparity maps. We basically just prefer the
			// high-res version unless we had no match at that pixel.
			disparityHiRes = cv::max(disparityLoRes, disparityHiRes);
			//disparityHiRes.convertTo(disparityHiRes, CV_16U);
			
			cv::imwrite(colourFile.str(), video->leftImage());
			cv::imwrite(depthFile.str(), disparityHiRes);
			cv::imwrite(rightFile.str(), video->rightImage());

			// Since the depth image is likely to be very dark, rescale it before showing it.
			double mini, maxi;
			cv::minMaxIdx(disparityHiRes, &mini, &maxi);
			//printf("%d %f %f %d\n", disparityHiRes.type(), mini, maxi, dispMaxi);
			if (maxi > dispMaxi)
				dispMaxi = maxi;
			double scale = 255.0 / dispMaxi;
			disparityHiRes.convertTo(disparityHiRes, CV_8UC1, scale, -mini*scale);
			
			cv::imshow("Diff", 0.5 * (video->rightImage() - video->leftImage()) + 127);
			cv::imshow("Disparity", disparityHiRes);
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
