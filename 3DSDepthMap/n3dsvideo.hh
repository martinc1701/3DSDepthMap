#ifndef N3DS_VIDEO_HH
#define N3DS_VIDEO_HH

#include <opencv2/core.hpp>
#include <queue>
#include <string>
struct AVFormatContext;
struct AVFrame;
struct AVPacket;


class N3DSVideo {
public:

	/**
		Creates a new instance ready to decode the given video file. If
		something went wrong (the file doesn't exist, wrong file format,
		etc), a cv::Exception instance will be thrown describing the error.
	*/
	N3DSVideo(const char *filename, bool wantGrayscale, bool flipCameras);
	~N3DSVideo();

	/**
		Prints the internal video information to the console. This is really
		only useful for debugging.
	*/
	void dumpVideoInfo() const;

	/**
		The width/height of the video.
	*/
	int width() const {	return m_width;	}
	int height() const { return m_height; }

	/**
		Call to process a portion of the video. If there is no video left
		to process, then this returns false.
	*/
	bool processStep();

	/**
		After a call to processNextFrame(), we may have a new stereo image
		pair available. If we do, then this returns true, and the leftImage()
		and rightImage() methods will return the most recently decoded stereo
		image pair. This flag will stay set until another call to 
		processNextFrame().
	*/
	bool hasNewStereoImage() const { return m_newStereoImage; }
	
	/**
		Returns the left/right image of the most recently decoded stereo image
		pair. These images will remain constant until two new corresponding
		images are decoded from the video.
	*/
	const cv::Mat leftImage() const { return m_curLeft; }
	const cv::Mat rightImage() const { return m_curRight;	}

private:
	
	N3DSVideo(const N3DSVideo&);
	N3DSVideo& operator=(const N3DSVideo&);

	static bool s_libavReady;

	std::string m_filename;
	int m_width;
	int m_height;
	std::queue<cv::Mat> m_leftUnmatched;
	std::queue<cv::Mat> m_rightUnmatched;
	cv::Mat m_curLeft;
	cv::Mat m_curRight;
	bool m_newStereoImage;

	AVFormatContext *m_fmtCtx;
	int m_leftStreamIdx;
	int m_rightStreamIdx;
	bool m_flushingPacket;
	bool m_wantGrayscale;

	AVFrame *m_tmpFrame;	
	AVPacket *m_packet;

	/**
		Decodes the current packet. It returns true if a video frame was decoded.
	*/
	bool decodePacket();
};


#endif