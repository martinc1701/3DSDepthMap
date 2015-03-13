#include "n3dsvideo.hh"
#include "utils.hh"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
}

#include <opencv2/opencv.hpp>


bool N3DSVideo::s_libavReady = false;


N3DSVideo::N3DSVideo(const char *filename, bool wantGrayscale) {
	if (!s_libavReady) {
		av_register_all();
		s_libavReady = true;
	}

	m_filename = filename;
	m_newStereoImage = false;
	m_flushingPacket = false;
	m_wantGrayscale = wantGrayscale;
	m_fmtCtx = nullptr;
	m_tmpFrame = nullptr;
	m_packet = nullptr;

	// Open the file. Hopefully this succeeds ...
	
	if (avformat_open_input(&m_fmtCtx, filename, nullptr, nullptr) < 0)
		CV_Error(cv::Error::StsBadArg, "file not found");
	if (avformat_find_stream_info(m_fmtCtx, nullptr) < 0)
		CV_Error(cv::Error::StsBadArg, "video stream information is invalid");
	
	// The Nintendo 3DS stores the left and right channels in two separate
	// video streams. We need to find these streams, and initialize the
	// left/rightStream variables accordingly.

	int nStreams = 0;
	int *curStreamIdx = &m_leftStreamIdx;

	for (unsigned i = 0; i < m_fmtCtx->nb_streams; ++i) {
		int stream = av_find_best_stream(
			m_fmtCtx, AVMEDIA_TYPE_VIDEO, i, -1, nullptr, 0);
		// If stream i isn't a video stream, or we can't decode it, then ignore it.
		if (stream < 0)
			continue;
		// If we found too many streams, then just stop here.
		if (nStreams >= 2)
			break;

		++nStreams;
		AVStream *curStream = m_fmtCtx->streams[stream];
		*curStreamIdx = stream;

		AVCodecContext *decCtx = curStream->codec;

		// Initialize the width/height members. These must remain constant for
		// both streams.

		if (nStreams == 1) {
			m_width = decCtx->width;
			m_height = decCtx->height;
		}
		else if (m_width != decCtx->width || m_height != decCtx->height) {
			--nStreams;
			continue;
		}
		
		// Find a decoder for the stream and initialize it. After we do this,
		// the stream will be ready for use.
		
		AVCodec *dec = avcodec_find_decoder(decCtx->codec_id);
		AVDictionary *opts = nullptr; // unused

		if (dec == nullptr || avcodec_open2(decCtx, dec, &opts) < 0)
			CV_Error_(cv::Error::StsUnsupportedFormat, ("cannot decode stream ", i));

		// Done - move to the next stream.
		curStreamIdx = &m_rightStreamIdx;
	}

	// Sanity check - we want exactly two video streams.
	if (nStreams < 2)
		CV_Error(cv::Error::StsBadArg, "cannot find matching L/R video streams");

	m_tmpFrame = av_frame_alloc();
	m_packet = new AVPacket;
	av_init_packet(m_packet);
}


N3DSVideo::~N3DSVideo() {
	if (m_fmtCtx) {
		avcodec_close(m_fmtCtx->streams[m_leftStreamIdx]->codec);
		avcodec_close(m_fmtCtx->streams[m_rightStreamIdx]->codec);
		avformat_close_input(&m_fmtCtx);
	}
	if (m_tmpFrame)
		av_frame_free(&m_tmpFrame);
	if (m_packet)
		delete m_packet;
}


void N3DSVideo::dumpVideoInfo() const {
	av_dump_format(m_fmtCtx, 0, m_filename.c_str(), 0);
}


bool N3DSVideo::processStep() {
	m_newStereoImage = false;

	if (m_flushingPacket || av_read_frame(m_fmtCtx, m_packet) < 0) {
		m_packet->data = nullptr;
		m_packet->size = 0;
		m_flushingPacket = decodePacket();
		return m_flushingPacket;
	}

	decodePacket();
	av_free_packet(m_packet);
	return true;
}


bool N3DSVideo::decodePacket() {
	// If this packet doesn't belong to the L/R video streams, then skip it.
	if (m_packet->stream_index != m_leftStreamIdx &&
		m_packet->stream_index != m_rightStreamIdx)
		return false;

	int gotFrame = false;
	AVCodecContext *decCtx = m_fmtCtx->streams[m_packet->stream_index]->codec;

	// Decode a frame from the packet.
	int res = avcodec_decode_video2(decCtx, m_tmpFrame, &gotFrame, m_packet);

	// Make sure nothing went wrong.
	if (res < 0) {
		char buf[256];
		av_strerror(res, buf, sizeof(buf));
		CV_Error_(cv::Error::StsError, ("while decoding video frame: ", buf));
	}
	if (!gotFrame)
		return false;
	if (decCtx->width != m_width || decCtx->height != m_height) 
		CV_Error_(cv::Error::StsError, ("frame size changed: got ", decCtx->width, "x", decCtx->height));
	if (decCtx->pix_fmt != AV_PIX_FMT_YUV420P &&
		decCtx->pix_fmt != AV_PIX_FMT_YUVJ420P) // JPEG
		CV_Error_(cv::Error::StsError, ("unsupported pixel format: ", av_get_pix_fmt_name(decCtx->pix_fmt)));

	cv::Mat convertedFrame;
	if (m_wantGrayscale)
		convertYUV420ToY(m_tmpFrame, m_width, m_height, convertedFrame);
	else
		convertYUV420ToRGB(m_tmpFrame, m_width, m_height, convertedFrame);

	if (m_packet->stream_index == m_leftStreamIdx)
		m_leftUnmatched.push(convertedFrame);
	else
		m_rightUnmatched.push(convertedFrame);

	while (m_leftUnmatched.size() && m_rightUnmatched.size()) {
		m_newStereoImage = true;
		m_curLeft = m_leftUnmatched.front();
		m_curRight = m_rightUnmatched.front();
		m_leftUnmatched.pop();
		m_rightUnmatched.pop();
	}
	
	return true;
}