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
    struct ValidRegion valid; // valid region for distortion
};


struct CameraChange {
    double K1_inv[6]; // inverse of intrinsic matrix of old camera
    double d1[12]; // distortion coefficients of old camera
    struct ValidRegion valid1; // valid region of old camera
    double H[9]; // homography (rotation) from old to new camera
    double d2[12]; // distortion coefficients of new camera
    double K2[6]; // intrinsic matrix of new camera
    struct ValidRegion valid2; // valid region of new camera
};

void rleWarpDistorted(
    const RLE *R, RLE *M, siz h_out, siz w_out, struct Camera* old_camera,
    struct Camera* new_camera);
