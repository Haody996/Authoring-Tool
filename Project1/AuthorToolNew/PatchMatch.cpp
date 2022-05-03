#include "PatchMatch.h"

cv::Mat PatchMatch::nnf_approx(Mat A, Mat B, Mat nnf,
    int patch_size,
    int iterations,
    bool nnf_initliazed,
    bool reconstruct) {
    std::srand(std::time(nullptr));
    NNF nnf_applicator = NNF(A, B, patch_size); 

    if (!nnf_initliazed) {
        nnf_applicator.randomize();
    }
    for (int i = 0; i < iterations; i++) {
        std::cout << "iteration: " << i + 1 << '\n';
        nnf_applicator.minimize(iterations);
    }
    if (reconstruct) {
        return nnf_applicator.reconstruction(A, B, nnf, patch_size);
    }
    else
        return A;
}

