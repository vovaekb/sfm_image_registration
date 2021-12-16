#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

#define NUM_FEATURES 10000
#define LOWE_RATIO 0.8
#define MIN_MATCHES 25
#define VISUALIZE true
#define RATIO 0.45 // 0.25
#define F_DIST 3.0
#define F_CONF 0.95


int main(int argc, char** argv) {
  string img1_file = argv[1];
  string img2_file = argv[2];

  // Reading image files
  Mat img1 = imread(img1_file, CV_LOAD_IMAGE_GRAYSCALE);
  Mat img2 = imread(img2_file, CV_LOAD_IMAGE_GRAYSCALE);

  // Optinally resize images
  resize(img1, img1, Size(), RATIO, RATIO);
  resize(img2, img2, Size(), RATIO, RATIO);
  
  // detect and compute keypoints and descriptors
  ORB orb(NUM_FEATURES);
  vector<KeyPoint> kp1, kp2;
  Mat des1, des2;

  orb.detect(img1, kp1);
  orb.compute(img1, kp1, des1);
  orb.detect(img2, kp2);
  orb.compute(img2, kp2, des2);
  
  // do keypoint matching
  BFMatcher matcher;
  vector< vector<DMatch> > matches;
  matcher.knnMatch(des1, des2, matches, 2);
  
  // apply Lowe's ratio test and filter out only good matches
  vector<DMatch> good_matches;
  for (int i = 0; i < matches.size(); i++) {
    vector<DMatch> matches_entry = matches[i];

    if (matches_entry.size() != 2)
      cerr << "match " << i << " does not have enough match candidates." << endl;
    // handle default case
    else {
      DMatch m1 = matches_entry[0];
      DMatch m2 = matches_entry[1];

      if (m1.distance < LOWE_RATIO * m2.distance) {
        good_matches.push_back(m1);
      }
    }

  }
  // Print number of good matches
  cout << "Good matches found " << good_matches.size() << " " << endl;
  
  // Skip if there are not enough good matches
  if (good_matches.size() < MIN_MATCHES) {
    cerr << "Not enough good matches." << endl;
  }
  // Estimate fundamental matrix F
  vector<Point2f> pts1(good_matches.size());
  vector<Point2f> pts2(good_matches.size());

  for (int i = 0; i < good_matches.size(); i++) {
    DMatch m = good_matches[i];
    pts1[i] = kp1[m.queryIdx].pt;
    pts2[i] = kp2[m.trainIdx].pt;
  }

  Mat mask;
  Mat F = findFundamentalMat(pts1, pts2, FM_RANSAC, F_DIST, F_CONF, mask);

  vector<DMatch> inliers;
  for (int i = 0; i < good_matches.size(); i++) {
    if (mask.at<bool>(i, 0)) inliers.push_back(good_matches[i]);
  }

  cout << "Inliers found: " << inliers.size() << endl;

  // Visualize matches
  if (VISUALIZE) {
    Mat visual;
    drawMatches(img1, kp1, img2, kp2, inliers, visual);
    imshow("Inliers preserved", visual);
    waitKey(0);
  }
  
  return 0;
}
