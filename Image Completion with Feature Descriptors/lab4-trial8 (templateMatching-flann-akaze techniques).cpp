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
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace std;

Mat image = imread("scrovegni/scrovegni.jpg");
Mat patch_0 = imread("scrovegni/patch_0.jpg");
Mat patch_1 = imread("scrovegni/patch_1.jpg");
Mat patch_2 = imread("scrovegni/patch_2.jpg");
Mat patch_3 = imread("scrovegni/patch_3.jpg");
Mat patch_t_0 = imread("scrovegni/patch_t_0.jpg");
Mat patch_t_1 = imread("scrovegni/patch_t_1.jpg");
Mat patch_t_2 = imread("scrovegni/patch_t_2.jpg");
Mat patch_t_3 = imread("scrovegni/patch_t_3.jpg");
Mat incompImage = imread("scrovegni/image_to_complete.jpg");



bool use_mask;
Mat mask; Mat result;

/*
 *
 * Filter/mirror/rotate (transform) the image 
 * Add 3 patches at the same time
 * Blending/mixing techniques ???
 * ORB instead of SIFT
 * Manuel RANSAC and affine transform
 * Template matching
 * Alternative feature detection techniques: Fast, KAZE, AKAZE
 *
 */

 
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

Mat cannyEdge(Mat image, String winname)
 {
     Mat contours;
     Mat gray_image;
     vector<cv::Mat> channels;
     Mat hsv;
     cvtColor(image, hsv, COLOR_RGB2GRAY);
     split(hsv, channels);
     gray_image = channels[0];

     Canny(image, contours, 100, 300);

     //imshow(winname+" Image", image);
     //imshow(winname+" Gray Image", gray_image);
     imshow(winname+" Contours", contours);

     return contours;

 }
 
Mat imageMatching(Mat& src1, Mat& dst1, double ratio, double distanceThreshold, String winname)
{
    int max_keypoints = 500;
    double oran = 0.67;
    double oran2 = 0.75;
    Mat src2, dst2;
    resize(src1, src2, Size(src1.cols*oran, src1.rows*oran),0,0,INTER_LINEAR);
    resize(dst1, dst2, Size(dst1.cols*oran2, dst1.rows*oran2), 0, 0, INTER_LINEAR);
    //src1.copyTo(src2);
    //dst1.copyTo(dst2);

    Mat src_gray, dst_gray, dst_flip, dst, src;
    //flip(dst, dst_flip, 1);
    cvtColor(src2, src, COLOR_BGR2GRAY);
    cvtColor(dst2, dst, COLOR_BGR2GRAY);

    //cv::Ptr<cv::xfeatures2d::BEBLID> descriptor = cv::xfeatures2d::BEBLID::create(0.75);

    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
    Ptr<SIFT> extractor = SIFT::create();

    //Step 1: Key point detection
    std::vector<KeyPoint> keypoints1, keypoints2;
    detector->detect(src, keypoints1);
    detector->detect(dst, keypoints2);

    // Step 2: Calculate descriptors (feature vectors)
    Mat descriptors1, descriptors2;
    extractor->compute(src, keypoints1, descriptors1);
    extractor->compute(dst, keypoints2, descriptors2);

    // Step 3: Match detected keypoints between 2 images
    FlannBasedMatcher matcher;
    //BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    std::vector<std::vector<DMatch>> matches2;
    matcher.knnMatch(descriptors1, descriptors2, matches2, 2); // Set k=2 for ratio test

    double max_dist = 0; double min_dist = 999999;

    // Step 4: Draw only "good" matches (ratio*min_dist)
    std::vector< DMatch > good_matches2;

    // Apply ratio test
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (matches2[i][0].distance < ratio * matches2[i][1].distance)
        {
            good_matches2.push_back(matches2[i][0]);
        }
    }

    //const float distanceThreshold = 500.0f; // Set the distance threshold
    std::vector<DMatch> filteredMatches;
    for (const auto& match : good_matches2)
    {
        if (match.distance < distanceThreshold)
        {
            filteredMatches.push_back(match);
        }
    }

    matches.clear();
    matches2.clear();

    Mat img_matches;
    drawMatches(src2, keypoints1, dst2, keypoints2, filteredMatches, img_matches, Scalar::all(-1),
        Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    descriptors1.release();
    descriptors2.release();

    // Localize the object
    std::vector<Point2f> first_keypoints;
    std::vector<Point2f> second_keypoints;

    for (int i = 0; i < filteredMatches.size(); i++)
    {
        //cout << i << " :";
        //-- Get the keypoints from the good matches
        if (keypoints1[filteredMatches[i].queryIdx].pt.x > 0 && keypoints1[filteredMatches[i].queryIdx].pt.y > 0 &&
            keypoints2[filteredMatches[i].trainIdx].pt.x > 0 && keypoints2[filteredMatches[i].trainIdx].pt.y > 0)
        {
            first_keypoints.push_back(keypoints1[filteredMatches[i].queryIdx].pt);
            //cout << "first point" << keypoints1[ filteredMatches[i].queryIdx ].pt << endl;

            second_keypoints.push_back(keypoints2[filteredMatches[i].trainIdx].pt);
            //cout << "second point" << keypoints2[ filteredMatches[i].trainIdx ].pt << endl;
        }
    }

    imshow(winname, img_matches);

    Mat h = findHomography(second_keypoints, first_keypoints, RANSAC);
    //Mat h = findHomography(s_keypoints, f_keypoints, RANSAC);

    Mat out;
    src2.copyTo(out); // copy the incomplete image to out to wrap the patch directly onto incomplete image.
    warpPerspective(dst2, out, h, src.size(), INTER_LINEAR, BORDER_TRANSPARENT);

    keypoints1.clear();
    keypoints2.clear();
    good_matches2.clear();
    filteredMatches.clear();

    return out;
}

Mat TemplateMatching(int, void*, Mat& img, Mat& templ, String winname)
{
    Mat img_display;
    img.copyTo(img_display);
    int result_cols = img_display.cols;
    int result_rows = img_display.rows;
    result.create(img.rows, img.cols, CV_32FC1);
    img.copyTo(result);

    // Methods of Template Matching: "SQDIFF", "SQDIFF NORMED", "TM CCORR", "TM CCORR NORMED", "TM COEFF", "TM COEFF NORMED";
    matchTemplate(img, templ, result, TM_CCOEFF_NORMED);
    normalize(result, result, 1, 0, NORM_L2, -1, Mat());

    double minVal; double maxVal; Point minLoc; Point maxLoc;
    Point matchLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

    matchLoc = maxLoc;
    double alpha = 0;
    rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols-2, matchLoc.y + templ.rows-2), Scalar(255, 255, 255, alpha), cv::FILLED);
    //rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(255, 255, 255), 0, LINE_4, 0);
    templ.copyTo(img_display(cv::Rect(matchLoc.x, matchLoc.y, templ.cols, templ.rows)));

    //imshow(winname, img_display);
    //imshow("result_window4", result);

    return img_display;
}

Mat stretchContrast(const cv::Mat& inputImage, double lowPercentile, double highPercentile)
{
    // Convert the input image to Lab color space
    cv::Mat labImage;
    cv::cvtColor(inputImage, labImage, cv::COLOR_BGR2Lab);

    // Split the Lab image into separate channels
    std::vector<cv::Mat> labChannels;
    cv::split(labImage, labChannels);

    // Calculate the low and high intensity thresholds based on percentiles
    cv::Scalar lowVal, highVal;
    cv::Mat flatImage = labChannels[0].reshape(1, 1);  // Flatten the intensity channel
    cv::minMaxLoc(flatImage, &lowVal[0], &highVal[0], nullptr, nullptr);
    lowVal[0] += (highVal[0] - lowVal[0]) * lowPercentile;
    highVal[0] -= (highVal[0] - lowVal[0]) * (1.0 - highPercentile);

    cv::Mat stretchedImage;
    cv::normalize(labChannels[0], stretchedImage, lowVal[0], highVal[0], cv::NORM_MINMAX);

    labChannels[0] = stretchedImage;
    cv::Mat outputLabImage;
    cv::merge(labChannels, outputLabImage);

    cv::Mat outputImage;
    cv::cvtColor(outputLabImage, outputImage, cv::COLOR_Lab2BGR);

    return outputImage;
}

Mat denoiseColorChannel(const cv::Mat& inputImage, int channelIndex, int kernelSize, int sColor, int sSpace)
{
    // Convert the input image to YUV color space
    cv::Mat yuvImage;
    cv::cvtColor(inputImage, yuvImage, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> yuvChannels;
    cv::split(yuvImage, yuvChannels);
    cv::Mat denoisedChannel;
    cv::Mat denoisedChannel2;
    bilateralFilter(yuvChannels[channelIndex], denoisedChannel, kernelSize, sColor, sSpace);

    // Replace the original color channel with the denoised channel
    yuvChannels[channelIndex] = denoisedChannel;
    //yuvChannels[2] = denoisedChannel2;

    cv::Mat outputYUVImage;
    cv::merge(yuvChannels, outputYUVImage);

    cv::Mat outputImage;
    cv::cvtColor(outputYUVImage, outputImage, cv::COLOR_YCrCb2BGR);

    return outputImage;
}

int main(int argc, char** argv)
{
    //Mat imageCE = cannyEdge(image, "Original");
    //Mat incompImageCE = cannyEdge(incompImage, "Incomplete");
    //Mat patch_0_CE = cannyEdge(patch_0, "Patch_0");
    //Mat patch_1_CE = cannyEdge(patch_1, "Patch_1");
    //Mat patch_t_1_CE = cannyEdge(patch_t_1, "Patch_t_1");

    // Apply the noise filter on the specified color channel
    Mat f_incompImage = denoiseColorChannel(incompImage, 0, 5, 15, 20);
    Mat f_patch_t_0 = denoiseColorChannel(patch_t_0, 0, 7, 30, 50);

    Mat output;
    output = imageMatching(incompImage, patch_t_1, 0.8, 220, "Incomplete and Patch_0 Matches");
    //output = imageMatching(output, patch_t_1, 2.0, 150, "Incomplete and Patch_1 Matches");
    //output = imageMatching(output, patch_t_2, 2.0, 150, "Incomplete and Patch_2 Matches");
    //output = imageMatching(output, patch_t_3, 2.0, 150, "Incomplete and Patch_3 Matches");

    imshow("MERGED IMAGE", output);

    Mat output3;
    //output3 = TemplateMatching(0, 0, incompImage, patch_0, " Incomplete and Patch_0 Image");
    //output3 = TemplateMatching(0, 0, output3, patch_1, " Incomplete and Patch_0 Image");
    //output3 = TemplateMatching(0, 0, output3, patch_2, " Incomplete and Patch_0 Image");
    //imshow("MERGED IMAGE USING TEMPLATE MATCHING METHOD", output3);

    waitKey(0);
    return 0;
}

