/*
 * UNIVERSITY OF PADOVA
 * DEPARTMENT OF INFORMATION ENGINEERING
 * ICT FOR INTERNET AND MULTIMEDIA
 *
 * 2022-2023 COMPUTER VISION
 * HOMEWORK-1 // LAB-2
 *
 * AUTHOR: UMUT BERK CAKMAKCI // 2071408
 *
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui_c.h>

using namespace cv;
using namespace std;


int kernel_length = 3;
Mat src;
Mat dst;
char window_name[] = "Smoothing Demo";
int display_caption(const char* caption);
int display_dst(int delay);

Mat image;
int blurMin = 1;
int blurMax = 50;
int sigma_min = 1;
int sigmaR_min = 1;
int sigmaS_min = 1;
int sigma_max = 100;

Mat draw_histogram(Mat& hist, Mat& image, Scalar color) {
    // Set the histogram parameters
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true, accumulate = false;

    // Compute the histogram
    Mat hist_image;
    calcHist(&hist, 1, 0, Mat(), hist_image, 1, &histSize, &histRange, uniform, accumulate);

    // Draw the histogram
    int hist_h = 200, hist_w = 256;
    int bin_w = cvRound((double)hist_w / histSize);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    normalize(hist_image, hist_image, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < histSize; i++) {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist_image.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(hist_image.at<float>(i))),
            color, 2, LINE_AA);
    }

    // Combine the histogram and image
    Mat combined_image;
    vconcat(histImage, combined_image);

    return combined_image;
}

void show_histogram(Mat& img) {
    // Separate the image into 3 channels
    vector<Mat> bgr_planes;
    split(img, bgr_planes);

    // Set the histogram parameters
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    bool uniform = true, accumulate = false;

    // Compute the histograms
    Mat b_hist, g_hist, r_hist;
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

    // Draw the histograms
    int hist_h = 400, hist_w = 512;
    int bin_w = cvRound((double)hist_w / histSize);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    for (int i = 1; i < histSize; i++) {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
            Scalar(255, 0, 0), 2, LINE_AA);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
            Scalar(0, 255, 0), 2, LINE_AA);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
            Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
            Scalar(0, 0, 255), 2, LINE_AA);
    }

    // Display the histogram
    namedWindow("Histogram", WINDOW_AUTOSIZE);
    imshow("Histogram", histImage);
}

Mat show_histograms(Mat& image, String window_name) {
    // Separate the image into 3 channels
    vector<Mat> bgr_planes;
    split(image, bgr_planes);


    // Define the colors for each channel
    Vec3b blue(255, 0, 0);
    Vec3b green(0, 255, 0);
    Vec3b red(0, 0, 255);

    // Draw the histograms
    Mat b_hist = draw_histogram(bgr_planes[0], image, blue);
    Mat g_hist = draw_histogram(bgr_planes[1], image, green);
    Mat r_hist = draw_histogram(bgr_planes[2], image, red);

    // Concatenate the histograms and show them
    Mat hist_combined;
    hconcat(b_hist, g_hist, hist_combined);
    hconcat(hist_combined, r_hist, hist_combined);

    return hist_combined;
	//imshow(window_name+" Histograms", hist_combined);
}

void Blur(int sliderValue, void*) // this is standart blurring operation, not asked in the tasks.
{
    if(sliderValue <= 0)
    {
        return;
    }
    Mat output;
    blur(src, output, Size(sliderValue, sliderValue));
    imshow("BlurWindow", output);
}

void MedianBlur(int sliderValue, void*)
{
    if (sliderValue <= 0)
    {
        return;
    }
    sliderValue = 2 * sliderValue - 1;
    Mat output;
    medianBlur(src, output, sliderValue);
    imshow("MedianBlurWindow", output);
}

void GaussianBlur(int sliderValue, void*)
{
    if (sliderValue <= 0)
    {
        return;
    }
    sliderValue = 2 * sliderValue - 1;
    Mat output;
    GaussianBlur(src, output, Size(sliderValue, sliderValue), sigma_min, sigma_min);
    imshow("GaussianBlurWindow", output);
}

void BilateralFilter(int sliderValue, void*)
{
    if (sliderValue <= 0)
    {
        return;
    }
    //sliderValue = 2 * sliderValue - 1;
    Mat output;
    bilateralFilter(src, output, sliderValue, sigmaR_min * 2, sigmaS_min / 2);
    imshow("BilateralFilterWindow", output);
}

void combineImages(Mat img2, Mat img1, String window_name)
{
    // Get dimension of final image
    int rows = img1.rows + img2.rows;
    int cols = max(img1.cols, img2.cols);

    // Create a black image
    Mat3b res(rows, cols, Vec3b(0, 0, 0));

    // Copy images in correct position
    img1.copyTo(res(Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(res(Rect(0, img1.rows, img2.cols, img2.rows)));

    // Show result
    imshow(window_name, res);
}


int main(int argc, char** argv) {

    /*    PART-1 of LAB-2    ****************************************************************************************************************/

	// Load image
    Mat image = imread("dei_noise.jpg");

    /*    ORIGINAL RGB IMAGE AND HISTOGRAM    **********************************************************************************************/
    // Show original image
    //imshow("Original Image", image);

    // Concatenate histograms of each channel of original image
    Mat hist_orig;
    vector<Mat> bgr_orig;
    split(image, bgr_orig);
    hist_orig = show_histograms(image, "Original Image");
    combineImages(image, hist_orig, "Original Image and Histogram");
    /****************************************************************************************************************************************/



	/*    EQUALIZED RGB IMAGE AND HISTOGRAM    **********************************************************************************************/
    // Equalize each channel separately
    Mat eq_image, hist_eq;
    vector<Mat> eq_channels;
    for (int i = 0; i < bgr_orig.size(); i++) {
        Mat eq_channel;
        equalizeHist(bgr_orig[i], eq_channel);
        eq_channels.push_back(eq_channel);
    }
    merge(eq_channels, eq_image);
    //imshow("Equalized Image (RGB)", eq_image);

    // Concatenate histograms of each channel of equalized image (RGB)
    vector<Mat> bgr_eq;
    split(eq_image, bgr_eq);
    hist_eq = show_histograms(eq_image, "Equalized Image (RGB)");
    combineImages(eq_image, hist_eq, "Equalized (RGB) Image and Histogram");
    /****************************************************************************************************************************************/



    /*    EQUALIZED LAB IMAGE AND HISTOGRAM    **********************************************************************************************/
    // Convert image to Lab color space and equalize only L channel
    Mat lab_image, hist_lab;
    cvtColor(image, lab_image, COLOR_BGR2Lab);
    vector<Mat> lab_channels;
    split(lab_image, lab_channels);
    equalizeHist(lab_channels[0], lab_channels[0]);
    merge(lab_channels, lab_image);
    cvtColor(lab_image, lab_image, COLOR_Lab2BGR);
    //imshow("Equalized Image (Lab)", lab_image);

    // Concatenate histograms of L channel of equalized image (Lab)
    hist_lab = show_histograms(lab_image, "Equalized Image (Lab)");
    combineImages(lab_image, hist_lab, "Equalized (Lab) Image and Histogram");
    /*****************************************************************************************************************************************/



    /*    PART-2 of LAB-2    ******************************************************************************************************************/

    src = image;
    Mat img_blur, img_gaus, img_median, img_bilat;

    //blur(src, img_blur, Size(kernel_length, kernel_length), Point(-1, -1));
    //GaussianBlur(src, img_gaus, Size(kernel_length, kernel_length), 0, 0);
    //medianBlur(src, img_median, kernel_length);
    //bilateralFilter(src, img_bilat, kernel_length, kernel_length * 2, kernel_length / 2);

    //imshow("Normal Blurred Image", img_blur);
    //imshow("Gaussian Blurred Image", img_gaus);
    //imshow("Median Blurred Image", img_median);
    //imshow("Bilateral Filtered Image", img_bilat);


    struct Userdata {int sliderId; std::string sliderName;};

    /*
    namedWindow("BlurWindow");
    createTrackbar("kernel Value", "BlurWindow", &blurMin, blurMax, Blur);
    imshow("BlurWindow", src);
    */
    
    namedWindow("MedianBlurWindow");
    createTrackbar("kernel Value", "MedianBlurWindow", &blurMin, blurMax, MedianBlur);
    imshow("MedianBlurWindow", src);


    Userdata userdata0{ 0, "kernelVal" };
    Userdata userdata1{ 1, "sigmaVal" };
    namedWindow("GaussianBlurWindow");
    createTrackbar("kernelVal", "GaussianBlurWindow", &blurMin, blurMax, GaussianBlur, &userdata0);
    createTrackbar("sigmaVal", "GaussianBlurWindow", &sigma_min, sigma_max, GaussianBlur, &userdata1);
    imshow("GaussianBlurWindow", src);


    Userdata userdata2{ 2, "kernelVal" };
    Userdata userdata3{ 3, "sigmaRVal" };
    Userdata userdata4{ 4, "sigmaSVal" };
    namedWindow("BilateralFilterWindow");
    createTrackbar("kernelVal", "BilateralFilterWindow", &blurMin, 10, BilateralFilter, &userdata2);
    createTrackbar("sigmaRVal", "BilateralFilterWindow", &sigmaR_min, sigma_max, BilateralFilter, &userdata3);
    createTrackbar("sigmaSVal", "BilateralFilterWindow", &sigmaS_min, sigma_max, BilateralFilter, &userdata4);
    imshow("BilateralFilterWindow", src);
    
    

    waitKey(0);
    return 0;
}


