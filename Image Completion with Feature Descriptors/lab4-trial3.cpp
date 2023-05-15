/*
 * UNIVERSITY OF PADOVA
 * DEPARTMENT OF INFORMATION ENGINEERING
 * ICT FOR INTERNET AND MULTIMEDIA
 *
 * 2022-2023 COMPUTER VISION
 * HOMEWORK-2 // LAB-4
 *
 * AUTHOR: UMUT BERK CAKMAKCI // 2071408
 *
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;


Mat filter(Mat noisy_img) {

    Mat gray_img;
    cvtColor(noisy_img, gray_img, COLOR_BGR2GRAY);
    
    // Compute the power spectrum of the noisy image
    Mat noisy_spec;
    dft(gray_img, noisy_spec, DFT_COMPLEX_OUTPUT);
    
    // Compute the power spectrum of the noise
    Mat noise_spec;
    Mat noise_img = gray_img - noisy_img;
    dft(noise_img, noise_spec, DFT_COMPLEX_OUTPUT);

	// Compute the Wiener filter
    double snr = 10; // signal-to-noise ratio
    Mat wiener_filter = Mat::zeros(noisy_spec.size(), CV_32FC2);

    for (int i = 0; i < noisy_spec.rows; i++) {
        for (int j = 0; j < noisy_spec.cols; j++) {
            float psd = norm(noisy_spec.at<Vec2f>(i, j));
            float nsd = norm(noise_spec.at<Vec2f>(i, j));
            float h = psd / (psd + nsd / snr);
            wiener_filter.at<Vec2f>(i, j) = Vec2f(h, 0);
        }
    }

    // Apply the Wiener filter to the noisy image
    Mat filtered_spec = noisy_spec.mul(wiener_filter);
    Mat filtered_img;
    idft(filtered_spec, filtered_img, DFT_SCALE | DFT_REAL_OUTPUT);
    
    // Display the results
    imshow("Noisy Image", noisy_img);
    imshow("Filtered Image", filtered_img);
    return filtered_img;
}

Mat imageMatching(Mat& src, Mat& dst, int ratio, String winname) {
    //, std::vector<Point2f>& first_keypoints, std::vector<Point2f>& second_keypoints
    int max_keypoints = 500;

    Ptr<SIFT> detector = SIFT::create();
    Ptr<SIFT> extractor = SIFT::create();

	//--Step 1: Key point detection
    std::vector<KeyPoint> keypoints1, keypoints2;
    //-- Step 2: Calculate descriptors (feature vectors)
    Mat descriptors1, descriptors2;

    detector->detect(src, keypoints1);
    detector->detect(dst, keypoints2);

    extractor->compute(src, keypoints1, descriptors1);
    extractor->compute(dst, keypoints2, descriptors2);

	//FlannBasedMatcher matcher;
    //BFMatcher matcher;
    BFMatcher matcher(NORM_L2);

    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    double max_dist = 0; double min_dist = 999999;

    //-- Quick calculation of max and min distances between keypoints
    for (int i = 0; i < descriptors1.rows; i++)
    {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);
    //-- Draw only "good" matches (ratio*min_dist)
    std::vector< DMatch > good_matches;

    for (int i = 0; i < descriptors1.rows; i++)
    {
        if (matches[i].distance < ratio * min_dist)
        {
            good_matches.push_back(matches[i]);
        }
    }
    matches.clear();

    Mat img_matches;
    drawMatches(src, keypoints1, dst, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    descriptors1.release();
    descriptors2.release();

    //-- Localize the object
    std::vector<Point2f> first_keypoints;
    std::vector<Point2f> second_keypoints;

    for (int i = 0; i < good_matches.size(); i++)
    {
        //cout << i << " :";
        //-- Get the keypoints from the good matches
        if (keypoints1[good_matches[i].queryIdx].pt.x > 0 && keypoints1[good_matches[i].queryIdx].pt.y > 0
            && keypoints2[good_matches[i].trainIdx].pt.x > 0 && keypoints2[good_matches[i].trainIdx].pt.y > 0) {
            first_keypoints.push_back(keypoints1[good_matches[i].queryIdx].pt);
            //cout << "first point" << keypoints1[ good_matches[i].queryIdx ].pt << endl;

            second_keypoints.push_back(keypoints2[good_matches[i].trainIdx].pt);
            //cout << "second point" << keypoints2[ good_matches[i].trainIdx ].pt << endl;
        }
    }

	Mat h = findHomography(second_keypoints, first_keypoints, RANSAC);

    Mat out;
    src.copyTo(out);
    warpPerspective(dst, out, h, src.size(),INTER_LINEAR, BORDER_TRANSPARENT);

	keypoints1.clear();
    keypoints2.clear();
    good_matches.clear();

	//-- Show detected matches
    imshow(winname, img_matches);

    return out;
}

int main(int argc, char** argv) {

    // Load image
    Mat image = imread("pratodellavalle.jpg");
    Mat patch_0 = imread("patch_0.jpg");
    Mat patch_1 = imread("patch_1.jpg");
    Mat patch_2 = imread("patch_2.jpg");
    Mat patch_t_0 = imread("patch_t_0.jpg");
    Mat patch_t_1 = imread("patch_t_1.jpg");
    Mat patch_t_2 = imread("patch_t_2.jpg");
    Mat incompImage = imread("image_to_complete.jpg");

    //imshow("Original Image", image);

    Mat result,output;
    output = imageMatching(incompImage, patch_0, 3, "Incomplete and Patch_0 Matches");
    output = imageMatching(output, patch_1, 3, "Incomplete and Patch_1 Matches");
    output = imageMatching(output, patch_2, 3, "Incomplete and Patch_2 Matches");

    imshow("MERGED IMAGE", output);

    waitKey(0);
    return 0;
}

