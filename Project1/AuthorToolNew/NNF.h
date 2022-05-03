#pragma once
#include <Magick++.h> 
#include <iostream> 
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core/utility.hpp>

#define MAX_DIST 65535

using namespace cv;
using namespace std;
using namespace Magick;

class NNF
{
	cv::Mat input, output;

	int S;  //Patch size

	// NNF field. RGB channel is -> { x_target, y_target, distance_scaled } 
	cv::Mat field;

	public: 
		NNF(cv::Mat input, cv::Mat output, int patchsize);

		// initialize field with random values
		void randomize();

		// initialize field from an existing (possibily smaller) NNF
		void initialize(NNF nnf);

		// compute initial value of the distance term
		void initialize();

		// multi-pass NN-field minimization (see "PatchMatch" - page 4)
		void minimize(int pass);

		// minimize a single link 
		void minimizeLink(int x, int y, int dir);

		// compute distance between two patch 
		int distance(int x, int y, int xp, int yp);

		// Image reconstruction step
		Mat reconstruction();

		bool is_clamped(int h, int w, int i0, int i1, int j0, int j1);

};

