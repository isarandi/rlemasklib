#pragma once
#include "basics.h"

struct ValidRegion {
    float *ru; // undistorted max radii
    float *tu; // undistorted thetas
    float ru2_max; // max(ru^2)
    float ru2_min; // min(ru^2)

    float *rd; // distorted max radii
    float *td; // distorted thetas
    float rd2_max; // max(rd^2)
    float rd2_min; // min(rd^2)
    siz n;
};

struct Camera {
    double f[2]; // focal length
    double c[2]; // principal point
    double s; // skew (K[0,1])
    double R[9]; // rotation
    double d[12]; // distortion coefficients
    // Sensor tilt (tau_x, tau_y of the 14-coefficient model) as the OpenCV-style
    // homography matTilt = matProjZ @ RotY(tau_y) @ RotX(tau_x), applied to normalized
    // coordinates after the 12-coefficient distortion, plus its inverse.
    double tilt[9];
    double tilt_inv[9];
    bool has_tilt;
    struct ValidRegion valid; // valid region for distortion (12-coefficient, pre-tilt domain)
};


struct CameraChange {
    double K1_inv[6]; // inverse of intrinsic matrix of old camera
    double d1[12]; // distortion coefficients of old camera
    double t1_inv[9]; // inverse sensor-tilt homography of old camera
    bool tilt1;
    struct ValidRegion valid1; // valid region of old camera
    double H[9]; // homography (rotation) from old to new camera
    double d2[12]; // distortion coefficients of new camera
    double t2[9]; // sensor-tilt homography of new camera
    bool tilt2;
    double K2[6]; // intrinsic matrix of new camera
    struct ValidRegion valid2; // valid region of new camera
};

// Returns false if the transform is degenerate (output is then an all-zeros mask).
bool rleWarpDistorted(
    const RLE *R, RLE *M, siz h_out, siz w_out, struct Camera* old_camera,
    struct Camera* new_camera);
