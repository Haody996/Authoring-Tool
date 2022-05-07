#include "NNF.h"

NNF::NNF(cv::Mat input, cv::Mat output, int patchsize)
{
	this->input = input;
	this->output = output;
	this->S = patchsize;
}

void NNF::randomize()
{
	// field
	int input_width = input.size().width;
	int input_height = input.size().height;

	int output_width = output.size().width;
	int output_height = output.size().height;

	Mat nnf(input_height, input_width, CV_8UC3, Scalar(0, 0, 0));
	this->field = nnf;

	for (int y = 0; y < input_height; y++) {
		for (int x = 0; x < input_width; x++) {
			nnf.at<cv::Vec3b>(y, x)[0] = rand() % output_width;
			nnf.at<cv::Vec3b>(y, x)[1] = rand() % output_height; 
			nnf.at<cv::Vec3b>(y, x)[2] = MAX_DIST;
		}
	}
	initialize();
}

void NNF::initialize(NNF nnf)
{
	// field
	int input_width = input.size().width;
	int input_height = input.size().height;

	int output_width = output.size().width;
	int output_height = output.size().height;

	Mat nnf_new(input_height, input_width, CV_8UC3, Scalar(0, 0, 0));
	this->field = nnf_new;

	int fx = input.size().width / nnf.input.size().width;
	int fy = input.size().height / nnf.input.size().height;

	//System.out.println("nnf upscale by "+fx+"x"+fy+" : "+nnf.input.W+","+nnf.input.H+" -> "+input.W+","+input.H);
	
	for (int y = 0; y < input.size().height; y++) {
		for (int x = 0; x < input.size().width; x++) {
			int xlow = min(x / fx, nnf.input.size().width - 1);
			int ylow = min(y / fy, nnf.input.size().height - 1);

			nnf_new.at<cv::Vec3b>(y, x)[0] = nnf.field.at<cv::Vec3b>(ylow, xlow)[0] * fx;
			nnf_new.at<cv::Vec3b>(y, x)[1] = nnf.field.at<cv::Vec3b>(ylow, xlow)[1] * fy;
			nnf_new.at<cv::Vec3b>(y, x)[2] = MAX_DIST;
		}
	}
	initialize();
}

void NNF::initialize()
{
	for (int y = 0; y < input.size().height; y++) {
		for (int x = 0; x < input.size().width; x++) {

			field.at<cv::Vec3b>(y, x)[2] = distance(x, y, field.at<cv::Vec3b>(y, x)[0], field.at<cv::Vec3b>(y, x)[1]);

			int output_width = output.size().width;
			int output_height = output.size().height;

			int iter = 0, maxretry = 20;
			while (field.at<cv::Vec3b>(y, x)[2] == MAX_DIST && iter < maxretry) {
				field.at<cv::Vec3b>(y, x)[0] = rand() % output_width;
				field.at<cv::Vec3b>(y, x)[1] = rand() % output_height;
				field.at<cv::Vec3b>(y, x)[2] = distance(x, y, field.at<cv::Vec3b>(y, x)[0], field.at<cv::Vec3b>(y, x)[1]);
				iter++;
			}
		}
	}
}

void NNF::minimizeLink(int x, int y, int dir)
{
	int xp, yp, dp;

	//Propagation Left/Right
	if (x - dir > 0 && x - dir < input.size().width) {
		xp = field.at<cv::Vec3b>(x - dir, y)[0] + dir;
		yp = field.at<cv::Vec3b>(x - dir, y)[1];
		dp = distance(x, y, xp, yp);
		if (dp < field.at<cv::Vec3b>(x, y)[2]) {
			field.at<cv::Vec3b>(y, x)[0] = xp;
			field.at<cv::Vec3b>(y, x)[1] = yp;
			field.at<cv::Vec3b>(y, x)[2] = dp;
		}
	}

	//Propagation Up/Down
	if (y - dir > 0 && y - dir < input.size().height) {
		xp = field.at<cv::Vec3b>(x, y - dir)[0];
		yp = field.at<cv::Vec3b>(x, y - dir)[1] + dir;
		dp = distance(x, y, xp, yp);
		if (dp < field.at<cv::Vec3b>(x,y)[2] ) {
			field.at<cv::Vec3b>(y, x)[0] = xp;
			field.at<cv::Vec3b>(y, x)[1] = yp;
			field.at<cv::Vec3b>(y, x)[2] = dp;
		}
	}

	//Random search
	int wi = output.size().width, xpi = field.at<cv::Vec3b>(y, x)[0], ypi = field.at<cv::Vec3b>(y, x)[1];
	while (wi > 0) {
		
		xp = xpi + rand() % (2 * wi) - wi;
		yp = ypi + rand() % (2 * wi) - wi;
		xp = max(0, min(output.size().width - 1, xp));
		yp = max(0, min(output.size().height - 1, yp));

		dp = distance(x, y, xp, yp);
		if (dp < field.at<cv::Vec3b>(x, y)[2]) {
			field.at<cv::Vec3b>(y, x)[0] = xp;
			field.at<cv::Vec3b>(y, x)[1] = yp;
			field.at<cv::Vec3b>(y, x)[2] = dp;
		}
		wi /= 2;
	}
}

int NNF::distance(int x, int y, int xp, int yp)
{
	long distance = 0, wsum = 0, ssdmax = 10 * 255 * 255;

	// for each pixel in the source patch
	for (int dy = -S; dy <= S; dy++) {
		for (int dx = -S; dx <= S; dx++) {
			wsum += ssdmax;

			int xks = x + dx, yks = y + dy;
			if (xks < 0 || xks >= input.size().width) { distance += ssdmax; continue; }
			if (yks < 0 || yks >= input.size().height) { distance += ssdmax; continue; }

			// cannot use masked pixels as a valid source of information
			if (input.at<cv::Vec3b>(xks, yks) == cv::Vec3b(0, 0, 0)) { distance += ssdmax; continue; }

			// corresponding pixel in the target patch
			int xkt = xp + dx, ykt = yp + dy;
			if (xkt < 0 || xkt >= output.size().width) { distance += ssdmax; continue; }
			if (ykt < 0 || ykt >= output.size().height) { distance += ssdmax; continue; }

			// cannot use masked pixels as a valid source of information
			if (output.at<cv::Vec3b>(xks, yks) == cv::Vec3b(0, 0, 0)) { distance += ssdmax; continue; }

			// SSD distance between pixels (each value is in [0,255^2])
			long ssd = 0;

			// value distance (weight for R/G/B components = 3/6/1)
			for (int band = 0; band < 3; band++) {
				int weight = (band == 0) ? 3 : (band == 1) ? 6 : 1;
				double diff2 = pow(input.at<cv::Vec3b>(xks, yks)[band] - output.at < cv::Vec3b>(xkt, ykt)[band], 2); // Value 
				ssd += weight * diff2;
			}

			// add pixel distance to global patch distance
			distance += ssd;
		}
	}

	return (int)(MAX_DIST * distance / wsum);
}

void NNF::minimize(int pass)
{
	{

		int min_x = 0, min_y = 0, max_x = input.size().width - 1, max_y = input.size().height - 1;

		// multi-pass minimization
		for (int i = 0; i < pass; i++) {

			// scanline order
			for (int y = min_y; y < max_y; y++)
				for (int x = min_x; x <= max_x; x++)
					if (field.at<cv::Vec3b>(y, x)[2] > 0) minimizeLink(x, y, +1);

			// reverse scanline order
			for (int y = max_y; y >= min_y; y--)
				for (int x = max_x; x >= min_x; x--)
					if (field.at<cv::Vec3b>(y, x)[2] > 0) minimizeLink(x, y, -1);
		}
	}
}
