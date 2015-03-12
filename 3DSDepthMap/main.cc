#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>
#include "utils.hh"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
}

#include <opencv2/opencv.hpp>


AVFormatContext *fmtCtx = nullptr;
int leftStreamIdx = 0;
int rightStreamIdx = 0;
std::string outputPath = "./";


/**
   Opens the file, and initializes the format context and left/right streams.
   */
void initializeStreams(const char *inputFilename) {

	// Open the input file and determine the file format.

	if (avformat_open_input(&fmtCtx, inputFilename, nullptr, nullptr) < 0) {
		printf("Unable to open input video %s\n", inputFilename);
		exit(1);
	}
	if (avformat_find_stream_info(fmtCtx, nullptr) < 0) {
		printf("Video stream information is invalid - is the file corrupt?\n");
		exit(1);
	}

	// Dump our format information to stderr, for debugging.
	av_dump_format(fmtCtx, 0, inputFilename, 0);

	// The Nintendo 3DS stores the left and right channels in two separate
	// video streams. We need to find these streams, and initialize the
	// left/rightStream variables accordingly.

	int nStreams = 0;
	int *curStreamIdx = &leftStreamIdx;

	for (unsigned i = 0; i < fmtCtx->nb_streams; ++i) {
		int stream = av_find_best_stream(
			fmtCtx, AVMEDIA_TYPE_VIDEO, i, -1, nullptr, 0);
		// If stream i isn't a video stream, or we can't decode it, then ignore it.
		if (stream < 0)
			continue;
		// If we found too many streams, then just stop here.
		if (nStreams >= 2) {
			printf("Video contains more than 2 streams - ignoring the rest\n");
			break;
		}

		++nStreams;
		AVStream *curStream = fmtCtx->streams[stream];
		*curStreamIdx = stream;

		// Find a decoder for the stream and initialize it. After we do this,
		// the stream will be ready for use.

		AVCodecContext *decCtx = curStream->codec;
		AVCodec *dec = avcodec_find_decoder(decCtx->codec_id);
		AVDictionary *opts = nullptr; // unused

		if (dec == nullptr) {
			printf("Unable to find codec to decode video stream %d\n", i);
			exit(1);
		}
		if (avcodec_open2(decCtx, dec, &opts) < 0) {
			printf("Unable to open codec to decode video stream %d\n", i);
			exit(1);
		}

		// Done - move to the next stream.
		curStreamIdx = &rightStreamIdx;
	}

	// Sanity check - we want exactly two video streams ...
	if (nStreams < 2) {
		printf("Unable to find left/right video streams in the input\n");
		exit(1);
	}
}


/**
   Decodes a portion of a single packet, using the given frame as a
   temporary buffer. This returns the number of bytes read from the packet,
   which for video should always be the entire packet.

   If gotFrame is non-null, then that flag will be set if any frame of
   a video stream is decoded. This is needed when we are flushing out
   the input buffer, since some codecs may compress multiple frames in one
   packet.
*/
int decodePacket(
	AVPacket *packet, AVFrame *frame,
	cv::Mat &destFrame, bool *gotFrame = nullptr)
{
	if (gotFrame) *gotFrame = false;
	int gotFrameF;

	if (packet->stream_index != leftStreamIdx &&
		packet->stream_index != rightStreamIdx)
		return packet->size; // Ignore the packet.

	// Decode a frame from the packet.
	AVCodecContext *decCtx = fmtCtx->streams[packet->stream_index]->codec;
	int bytes = avcodec_decode_video2(decCtx, frame, &gotFrameF, packet);

	// Check for errors.
	if (bytes < 0) {
		char buf[256];
		av_strerror(bytes, buf, sizeof(buf));
		printf("Error while decoding video frame: %s\n", buf);
		return bytes;
	}
	if (!gotFrameF)
		return packet->size;
	/*
	if (decCtx->width != destFrame->width() ||
	decCtx->height != destFrame->height())
	{
	printf("Error while decoding video frame: the size of the video changed\n"
	"  expected: %dx%d, got: %dx%d\n",
	destFrame->width(), destFrame->height(),
	decCtx->width, decCtx->height);
	return packet->size;
	}
	*/

	// Should have a valid video frame now. However, we need to ensure
	// that it's in the format we expect (RGB). The image is likely
	// in some version of YUV420 format, which will require some
	// conversion.

	if (decCtx->pix_fmt != AV_PIX_FMT_YUV420P &&
		decCtx->pix_fmt != AV_PIX_FMT_YUVJ420P) // JPEG - any differences?
	{
		printf("Error while decoding video frame: unsupported pixel format\n"
			"  expected: yuv420p, got: %s\n",
			av_get_pix_fmt_name(decCtx->pix_fmt));
		return packet->size;
	}

	convertYUV420ToRGB(frame, decCtx->width, decCtx->height, destFrame);

	if (gotFrame) *gotFrame = true;
	return packet->size;
}




int main(int argc, char **argv) {

	if (argc < 2) {
		printf("No video file provided - exiting\n");
		return 0;
	}

	av_register_all();

	const std::string inputPath = argv[1];
	initializeStreams(inputPath.c_str());

	// The output path is built from the input filename, minus the file
	// extension.

	outputPath = inputPath;
	size_t lastSlash = outputPath.find_last_of("\\/");
	if (lastSlash != std::string::npos)
		outputPath = outputPath.substr(lastSlash + 1);
	size_t lastDot = outputPath.rfind('.');
	if (lastDot != std::string::npos)
		outputPath = outputPath.substr(0, lastDot);

	// Build our output directories.

	makeDirectory(outputPath.c_str());
	makeDirectory((outputPath + "/image").c_str());
	makeDirectory((outputPath + "/depth").c_str());

	// When we process the file, we read it in chunks, and then decode whatever
	// frames of each stream we get. This means that the left and right streams
	// may not be decoded in sync, so we need to keep unmatched images around
	// until both are decoded.

	AVFrame *tmpFrame = av_frame_alloc();
	AVPacket packet; av_init_packet(&packet);
	packet.data = nullptr;
	packet.size = 0;
	cv::Mat decFrame;

	printf("Processing video ...\n");

	//XXX
	cv::namedWindow("Left", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Right", cv::WINDOW_AUTOSIZE);

	while (av_read_frame(fmtCtx, &packet) >= 0) {
		AVPacket savedPacket = packet;
		bool gotFrame = true;
		do {
			int bytesUsed = decodePacket(&packet, tmpFrame, decFrame, &gotFrame);
			if (bytesUsed < 0)
				break;
			if (gotFrame) {
				if (packet.stream_index == leftStreamIdx)
					cv::imshow("Left", decFrame);
				else
					cv::imshow("Right", decFrame);
				cv::waitKey(16);
			}
			packet.data += bytesUsed;
			packet.size -= bytesUsed;
		} while (packet.size > 0);
		av_free_packet(&savedPacket);
	}
	// We may still have some frames cached, so flush them.
	packet.data = nullptr;
	packet.size = 0;
	bool gotFrame = true;
	while (gotFrame) {
		decodePacket(&packet, tmpFrame, decFrame, &gotFrame);
	}

	printf("Success.\n");

	// Clean up.

	avcodec_close(fmtCtx->streams[leftStreamIdx]->codec);
	avcodec_close(fmtCtx->streams[rightStreamIdx]->codec);
	avformat_close_input(&fmtCtx);
	av_frame_free(&tmpFrame);

	return 0;
}
