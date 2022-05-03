#pragma once
#include "NNF.h"
#include <Magick++.h> 
#include <iostream> 
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core/utility.hpp>

using namespace cv;
using namespace std;
using namespace Magick;


class PatchMatch
{
public:
    Mat nnf_approx(Mat A, Mat B,
        Mat nnf,
        int patch_size,
        int iterations,
        bool nnf_initliazed,
        bool reconstruct);

};