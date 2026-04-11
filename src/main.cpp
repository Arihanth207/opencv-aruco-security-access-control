/**
 * ============================================================
 * ArUco Marker-Based Security Access Control System
 * ============================================================
 * Library  : OpenCV 4.6 (ArUco contrib module)
 * Domain   : Security / Access Control
 * Language : C++17
 *
 * Team:
 * Arihanth — Core detection pipeline       (PR #1 - merged)
 * Adityan  — Pose estimation & overlay     (PR #2 - this branch)
 * Sakthi   — Tamper detection enhancement  (coming in PR #3)
 *
 * Core Method Analysed: cv::aruco::detectMarkers()
 * ============================================================
 */

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>

// ─── Access Level Database ─────────────────────────────────────────────────
// Maps each ArUco marker ID to an access level and zone
struct AccessRule {
    std::string level;   // GUEST | STAFF | ADMIN
    std::string zone;    // physical zone this badge grants entry to
    cv::Scalar  colour;  // BGR colour used for annotation overlay
};

static const std::map<int, AccessRule> ACCESS_DB = {
    {  0, {"GUEST", "Lobby",       cv::Scalar( 50, 200,  50)} },
    {  1, {"GUEST", "Lobby",       cv::Scalar( 50, 200,  50)} },
    {  5, {"STAFF", "Office",      cv::Scalar(255, 150,  50)} },
    {  6, {"STAFF", "Office",      cv::Scalar(255, 150,  50)} },
    { 10, {"ADMIN", "Server Room", cv::Scalar(  0,   0, 220)} },
    { 23, {"ADMIN", "Server Room", cv::Scalar(  0,   0, 220)} },
};

// ─── Utility: semi-transparent filled rectangle ────────────────────────────
void drawFilledRect(cv::Mat& img, cv::Rect r, cv::Scalar colour, double alpha = 0.75)
{
    cv::Mat overlay;
    img.copyTo(overlay);
    cv::rectangle(overlay, r, colour, cv::FILLED);
    cv::addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, img);
}

// ─── Utility: current timestamp string ────────────────────────────────────
std::string currentTime()
{
    auto t = std::chrono::system_clock::to_time_t(
                 std::chrono::system_clock::now());
    std::ostringstream ss;
    ss << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

// ─── Pose Estimation: Added by Adityan (PR #2) ────────────────────────────
/**
 * estimateAndDrawPose()
 *
 * Estimates the 3D pose of each detected ArUco marker using
 * cv::aruco::estimatePoseSingleMarkers() which internally calls solvePnP.
 *
 * solvePnP solves the Perspective-n-Point problem:
 * Given 4 known 3D corner points of a marker and their 2D projections
 * in the image, compute the rotation (rvec) and translation (tvec)
 * vectors that describe the marker's pose relative to the camera.
 *
 * Output:
 * - 3D coordinate axes drawn on each marker (X=red, Y=green, Z=blue)
 * - Distance from camera displayed next to each marker
 * - Rotation angles (roll, pitch, yaw) logged to console
 */
void estimateAndDrawPose(
    cv::Mat&                                     annotated,
    const std::vector<std::vector<cv::Point2f>>& corners,
    const std::vector<int>&                      ids,
    const cv::Mat&                               cameraMatrix,
    const cv::Mat&                               distCoeffs,
    float                                        markerLength)
{
    if (ids.empty()) return;

    // Estimate rotation and translation vectors for each marker
    // rvec: rotation vector (Rodrigues format)
    // tvec: translation vector (x, y, z in metres from camera)
    std::vector<cv::Vec3d> rvecs, tvecs;
    cv::aruco::estimatePoseSingleMarkers(
        corners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

    for (size_t i = 0; i < ids.size(); ++i) {

        // Draw XYZ coordinate axes on each marker
        // X axis = red, Y axis = green, Z axis = blue
        cv::drawFrameAxes(annotated, cameraMatrix, distCoeffs,
                          rvecs[i], tvecs[i], markerLength * 0.6f);

        // Calculate Euclidean distance from camera to marker (in metres)
        double distance = cv::norm(tvecs[i]);

        // Convert rotation vector to rotation matrix for angle extraction
        cv::Mat rotMatrix;
        cv::Rodrigues(rvecs[i], rotMatrix);

        // Extract roll, pitch, yaw from rotation matrix
        double roll  = atan2(rotMatrix.at<double>(2, 1),
                             rotMatrix.at<double>(2, 2)) * 180.0 / CV_PI;
        double pitch = atan2(-rotMatrix.at<double>(2, 0),
                             sqrt(pow(rotMatrix.at<double>(2, 1), 2) +
                                  pow(rotMatrix.at<double>(2, 2), 2))) * 180.0 / CV_PI;
        double yaw   = atan2(rotMatrix.at<double>(1, 0),
                             rotMatrix.at<double>(0, 0)) * 180.0 / CV_PI;

        // Display distance label near each marker
        cv::Point2f centre(0.f, 0.f);
        for (auto& p : corners[i]) centre += p;
        centre *= 0.25f;

        std::ostringstream distLabel;
        distLabel << std::fixed << std::setprecision(2) << distance << "m";
        cv::putText(annotated, distLabel.str(),
                    cv::Point((int)(centre.x - 20), (int)(centre.y + 30)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

        // Log pose data to console
        std::cout << "  Pose  ID " << std::setw(3) << ids[i]
                  << "  |  Dist: "  << std::fixed << std::setprecision(3)
                  << distance << " m"
                  << "  |  Roll: "  << std::setprecision(1) << roll  << "°"
                  << "  Pitch: "    << pitch << "°"
                  << "  Yaw: "      << yaw   << "°\n";
    }
}

// ─── Core: process one badge image ────────────────────────────────────────
void processBadge(const std::string& imgPath,
                  const std::string& outDir,
                  std::ofstream&     logFile)
{
    // Load image from disk
    cv::Mat img = cv::imread(imgPath);
    if (img.empty()) {
        std::cerr << "[WARN] Cannot read image: " << imgPath << "\n";
        return;
    }

    // ── Step 1: Setup ArUco dictionary ────────────────────────────────────
    cv::Ptr<cv::aruco::Dictionary> dict =
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);

    // ── Step 2: Configure detection parameters ────────────────────────────
    cv::Ptr<cv::aruco::DetectorParameters> params =
        cv::aruco::DetectorParameters::create();

    params->cornerRefinementMethod    = cv::aruco::CORNER_REFINE_SUBPIX;
    params->adaptiveThreshWinSizeMin  = 7;
    params->adaptiveThreshWinSizeMax  = 53;
    params->adaptiveThreshWinSizeStep = 10;
    params->minMarkerPerimeterRate    = 0.02;

    // ── Step 3: Detect ArUco markers ──────────────────────────────────────
    std::vector<int>                      ids;
    std::vector<std::vector<cv::Point2f>> corners, rejected;

    auto t0 = std::chrono::high_resolution_clock::now();
    cv::aruco::detectMarkers(img, dict, corners, ids, params, rejected);
    auto t1 = std::chrono::high_resolution_clock::now();

    double detMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    cv::Mat annotated = img.clone();
    if (!ids.empty())
        cv::aruco::drawDetectedMarkers(annotated, corners, ids);

    // ── Step 4: Camera intrinsics for pose estimation ─────────────────────
    float fx = img.cols * 1.2f;
    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3)
        << fx,  0,  img.cols / 2.0,
            0, fx,  img.rows / 2.0,
            0,  0,  1.0);
    cv::Mat distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
    float markerLength = 0.05f; 

    // ── Step 5: Pose estimation (Adityan — PR #2) ─────────────────────────
    estimateAndDrawPose(annotated, corners, ids,
                        cameraMatrix, distCoeffs, markerLength);

    // ── Step 6: Access level lookup per detected marker ───────────────────
    std::string overallLevel  = "DENIED";
    std::string overallZone   = "-";
    cv::Scalar  overallColour(0, 0, 180);
    bool        anyKnown = false;

    for (size_t i = 0; i < ids.size(); ++i) {
        int id = ids[i];

        auto it = ACCESS_DB.find(id);
        std::string level  = (it != ACCESS_DB.end()) ? it->second.level  : "UNKNOWN";
        std::string zone   = (it != ACCESS_DB.end()) ? it->second.zone   : "-";
        cv::Scalar  colour = (it != ACCESS_DB.end()) ? it->second.colour
                                                      : cv::Scalar(0, 0, 180);

        if (it != ACCESS_DB.end()) {
            anyKnown      = true;
            overallLevel  = level;
            overallZone   = zone;
            overallColour = colour;
        }

        cv::Point2f centre(0.f, 0.f);
        for (auto& p : corners[i]) centre += p;
        centre *= 0.25f;

        cv::putText(annotated,
                    "ID:" + std::to_string(id) + " [" + level + "]",
                    cv::Point((int)(centre.x - 50), (int)(centre.y - 15)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

        std::cout << "  Access ID " << std::setw(3) << id
                  << "  |  Level: " << std::setw(8) << level
                  << "  |  Zone: "  << zone << "\n";
    }

    // ── Step 7: Draw access decision banner ───────────────────────────────
    bool granted = anyKnown;

    std::string statusText;
    if (ids.empty())
        statusText = "NO BADGE DETECTED";
    else if (!anyKnown)
        statusText = "ACCESS DENIED — Unknown badge";
    else
        statusText = "ACCESS GRANTED  [" + overallLevel + "]  Zone: " + overallZone;

    cv::Scalar bannerColour = granted
        ? cv::Scalar(30, 130, 30)
        : cv::Scalar(25,  25, 150);

    drawFilledRect(annotated, cv::Rect(0, 0, annotated.cols, 58), bannerColour);
    cv::putText(annotated, statusText,
                cv::Point(12, 38),
                cv::FONT_HERSHEY_SIMPLEX, 0.70,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    cv::putText(annotated, currentTime(),
                cv::Point(annotated.cols - 245, annotated.rows - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(190, 190, 190), 1, cv::LINE_AA);

    std::ostringstream stats;
    stats << "Detect: " << std::fixed << std::setprecision(1)
          << detMs << " ms  |  Markers: " << ids.size();
    cv::putText(annotated, stats.str(),
                cv::Point(8, annotated.rows - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.42,
                cv::Scalar(190, 190, 190), 1, cv::LINE_AA);

    // ── Step 8: Save annotated result ─────────────────────────────────────
    std::string base = imgPath.substr(imgPath.find_last_of("/\\") + 1);
    base = base.substr(0, base.find_last_of('.'));
    std::string outPath = outDir + "/" + base + "_result.jpg";
    cv::imwrite(outPath, annotated);
    std::cout << "  Saved: " << outPath << "\n\n";

    // ── Step 9: Write JSON log entry ──────────────────────────────────────
    logFile << "  {\n"
            << "    \"image\": \""      << imgPath       << "\",\n"
            << "    \"timestamp\": \""  << currentTime() << "\",\n"
            << "    \"markers\": "      << ids.size()    << ",\n"
            << "    \"ids\": [";
    for (size_t i = 0; i < ids.size(); ++i)
        logFile << ids[i] << (i + 1 < ids.size() ? ", " : "");
    logFile << "],\n"
            << "    \"access_level\": \"" << overallLevel  << "\",\n"
            << "    \"zone\": \""          << overallZone   << "\",\n"
            << "    \"granted\": "         << (granted ? "true" : "false") << ",\n"
            << "    \"detection_ms\": "
            << std::fixed << std::setprecision(2) << detMs << "\n"
            << "  }";
}

// ─── Entry point ──────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════╗\n"
              << "║  ArUco Security Access Control System — OpenCV 4.6  ║\n"
              << "╚══════════════════════════════════════════════════════╝\n\n";

    std::vector<std::string> images;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i)
            images.push_back(argv[i]);
    } else {
        cv::glob("images/*.jpg", images, false);
        std::vector<std::string> pngs;
        cv::glob("images/*.png", pngs, false);
        images.insert(images.end(), pngs.begin(), pngs.end());
    }

    if (images.empty()) {
        std::cerr << "[ERROR] No images found.\n"
                  << "        Pass image paths as arguments or place JPGs in images/\n";
        return 1;
    }

    std::string outDir = "results";
    std::ofstream log(outDir + "/access_log.json");
    log << "[\n";

    std::cout << "Processing " << images.size() << " badge image(s)...\n\n";

    for (size_t i = 0; i < images.size(); ++i) {
        std::cout << "━━ " << images[i] << "\n";
        processBadge(images[i], outDir, log);
        log << (i + 1 < images.size() ? ",\n" : "\n");
    }

    log << "]\n";
    log.close();

    std::cout << "Access log saved → " << outDir << "/access_log.json\n\n";
    return 0;
}