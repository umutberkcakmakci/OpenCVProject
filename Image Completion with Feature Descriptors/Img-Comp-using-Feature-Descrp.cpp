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
 * Chosen Dataset for this code is PratoDellaValle.
 *
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

Mat image = imread("pratodellavalle.jpg");
Mat patch_0 = imread("patch_0.jpg");
Mat patch_1 = imread("patch_1.jpg");
Mat patch_2 = imread("patch_2.jpg");
Mat patch_3 = imread("patch_3.jpg");
Mat patch_t_0 = imread("patch_t_0.jpg");
Mat patch_t_1 = imread("patch_t_1.jpg");
Mat patch_t_2 = imread("patch_t_2.jpg");
Mat patch_t_3 = imread("patch_t_3.jpg");
Mat incompImage = imread("image_to_complete.jpg");

bool use_mask;
Mat mask; Mat result;

/*
 * Chosen Dataset: Prato Della Valle
 *
 * Filter/mirror/rotate (transform) the images
 * ORB instead of SIFT
 * Template matching
 * FlanBasedMatcher + knnMatch
 * Alternative feature detection techniques: Fast, GFTT, AKAZE
 *
 */

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
 
Mat imageMatching(Mat& src_color, Mat& dst_color, double ratio, double distanceThreshold, String winname)
{
    int max_keypoints = 500;
    
    Mat src_gray, dst_gray;
    cvtColor(src_color, src_gray, COLOR_BGR2GRAY);
    cvtColor(dst_color, dst_gray, COLOR_BGR2GRAY);

    //cv::Ptr<cv::xfeatures2d::BEBLID> descriptor = cv::xfeatures2d::BEBLID::create(0.75);

    //Ptr<SIFT> detector = SIFT::create(0,3,0.08,10,2);
    //Ptr<ORB> detector = ORB::create();
    //Ptr<FastFeatureDetector> detector = FastFeatureDetector::create();
    Ptr<AKAZE> detector = AKAZE::create(AKAZE::DESCRIPTOR_KAZE, 1, 3, 0.001, 4, 4, KAZE::DIFF_PM_G2);
    //Ptr<GFTTDetector> detector = GFTTDetector::create(100, 0.02, 1, 3, true, 0.04);
    Ptr<SIFT> extractor = SIFT::create();

    //Step 1: Key point detection
    vector<KeyPoint> keypoints1, keypoints2;
    detector->detect(src_gray, keypoints1);
    detector->detect(dst_gray, keypoints2);

    // Step 2: Calculate descriptors (feature vectors)
    Mat descriptors1, descriptors2;
    extractor->compute(src_gray, keypoints1, descriptors1);
    extractor->compute(dst_gray, keypoints2, descriptors2);

    // Step 3: Match detected keypoints between 2 images
    FlannBasedMatcher matcher;
    //BFMatcher matcher(NORM_L2);
    //BFMatcher matcher(NORM_HAMMING);

    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    vector<vector<DMatch>> matches2;
    matcher.knnMatch(descriptors1, descriptors2, matches2, 2); // Set k=2 for ratio test

    double max_dist = 0; double min_dist = 999999;

    // Step 4: Draw only "good" matches (ratio*min_dist)
    vector< DMatch > good_matches2;

	/************ Taken from Github ****************************************************************************************************/

    // Apply ratio test
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (matches2[i][0].distance < ratio * matches2[i][1].distance)
        {
            good_matches2.push_back(matches2[i][0]);
        }
    }

    // Elimination with Distance Threshold
    vector<DMatch> filteredMatches;
    for (const auto& match : good_matches2)
    {
        if (match.distance < distanceThreshold)
        {
            filteredMatches.push_back(match);
        }
    }

    /************ End of Github Code ****************************************************************************************************/

    matches.clear();
    matches2.clear();

    Mat img_matches;
    // Draw the matches found between two images
    drawMatches(src_color, keypoints1, dst_color, keypoints2, filteredMatches, img_matches, Scalar::all(-1),
        Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    descriptors1.release();
    descriptors2.release();
    
    /************ Taken from StackOverFlow  ****************************************************************************************************/

    // We are gonna use these keypoints to find the homography of the images
    vector<Point2f> first_keypoints;
    vector<Point2f> second_keypoints;

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
    /************ End of StackOverFlow code ****************************************************************************************************/

    imshow(winname, img_matches);

    // Calculate homography
    Mat h = findHomography(second_keypoints, first_keypoints, RANSAC);

    // Overlay 2 images onto one another
    Mat overlayed_image;
    src_color.copyTo(overlayed_image); // copy the incomplete image to 'out' to wrap the patch directly onto incomplete image.
    warpPerspective(dst_color, overlayed_image, h, src_color.size(), INTER_LINEAR, BORDER_TRANSPARENT);

    keypoints1.clear();
    keypoints2.clear();
    good_matches2.clear();
    filteredMatches.clear();

    return overlayed_image;
}

Mat TemplateMatching(int, void*, Mat& img, Mat& templ, String winname)
{
    Mat img_display;
    double minVal; double maxVal; Point minLoc; Point maxLoc;
    Point matchLoc;
    double alpha = 0;

    img.copyTo(img_display);
    img.copyTo(result);

    /************ Taken from OpenCV  ****************************************************************************************************/

    // Methods of Template Matching: "SQDIFF", "SQDIFF NORMED", "TM CCORR", "TM CCORR NORMED", "TM COEFF", "TM COEFF NORMED";
    matchTemplate(img, templ, result, TM_CCOEFF_NORMED);
    normalize(result, result, 1, 0, NORM_L2, -1, Mat());
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

    matchLoc = maxLoc;
    rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols-2, matchLoc.y + templ.rows-2), Scalar(255, 255, 255, alpha), cv::FILLED);
    //rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(255, 255, 255), 2, 8, 0);

	/************ End of OpenCV code ****************************************************************************************************/

    templ.copyTo(img_display(cv::Rect(matchLoc.x, matchLoc.y, templ.cols, templ.rows)));

    //imshow(winname, img_display);
    //imshow("result_window4", result);

    return img_display;
}

Mat denoiseColorChannel(const cv::Mat& inputImage, int channelIndex, int kernelSize, int sColor, int sSpace)
{
    Mat yuvImage;
    cvtColor(inputImage, yuvImage, COLOR_BGR2YCrCb);
    vector<Mat> yuvChannels;
    split(yuvImage, yuvChannels);
    Mat denoisedChannel;
    Mat denoisedChannel2;
    bilateralFilter(yuvChannels[channelIndex], denoisedChannel, kernelSize, sColor, sSpace);

    // Replace the original color channel with the denoised channel
    yuvChannels[channelIndex] = denoisedChannel;
    //yuvChannels[2] = denoisedChannel2;

    Mat outputYUVImage;
    merge(yuvChannels, outputYUVImage);

	Mat outputImage;
    cvtColor(outputYUVImage, outputImage, cv::COLOR_YCrCb2BGR);

    return outputImage;
}

int main(int argc, char** argv)
{
    //Mat imageCE = cannyEdge(image, "Original");
    //Mat incompImageCE = cannyEdge(incompImage, "Incomplete");
    //Mat patch_0_CE = cannyEdge(patch_0, "Patch_0");
    //Mat patch_1_CE = cannyEdge(patch_1, "Patch_1");
    //Mat patch_t_1_CE = cannyEdge(patch_t_1, "Patch_t_1");

    // Resize only the original image
    double oran = 0.9;
    Mat r_incompImage;
    resize(incompImage, r_incompImage, Size(incompImage.cols * oran, incompImage.rows * oran), 0, 0, INTER_LINEAR);

    // Flip only the mirror image, which is Patch_t_1
    Mat flippedt1;
	flip(patch_t_1, flippedt1, 1);

    // Apply the noise filter on the specified color channel
    Mat f_incompImage = denoiseColorChannel(r_incompImage, 0, 5, 15, 20);
    Mat f_patch_t_1 = denoiseColorChannel(flippedt1, 0, 8, 45, 15);
    Mat f_patch_t_0 = denoiseColorChannel(patch_t_0, 0, 9, 45, 25);
    Mat f_patch_t_2 = denoiseColorChannel(patch_t_2, 0, 8, 45, 45);

    Mat output;
    output = imageMatching(r_incompImage, f_patch_t_1, 0.8, 200, "Incomplete and Patch_t_1 Matches");
    output = imageMatching(output, f_patch_t_0, 0.8, 200, "Incomplete and Patch_t_0 Matches");
    output = imageMatching(output, f_patch_t_2, 0.8, 200, "Incomplete and Patch_t_2 Matches");
    //output = imageMatching(output, patch_t_3, 0.8, 200, "Incomplete and Patch_t_3 Matches");
    imshow("Merged Image with Transposed Images", output);

    output = imageMatching(r_incompImage, patch_0, 0.8, 120, "Incomplete and Patch_0 Matches");
    output = imageMatching(output, patch_1, 0.8, 120, "Incomplete and Patch_1 Matches");
    output = imageMatching(output, patch_2, 0.8, 120, "Incomplete and Patch_2 Matches");
    //output = imageMatching(output, patch_3, 0.8, 200, "Incomplete and Patch_3 Matches");
    imshow("Merged Image with Normal Images", output);

    Mat output3;
    output3 = TemplateMatching(0, 0, incompImage, patch_0, " Incomplete and Patch_0 Image");
    output3 = TemplateMatching(0, 0, output3, patch_1, " Incomplete and Patch_1 Image");
    output3 = TemplateMatching(0, 0, output3, patch_2, " Incomplete and Patch_2 Image");
    imshow("Merged Image using Template Matching with Normal Images", output3);

    waitKey(0);
    return 0;
}

