/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#define _USE_MATH_DEFINES // Required for M_PI on Windows

#include <TinyAD/Scalar.hh>

/**
 * Compute derivatives of the angle between a pair of 3D
 * vectors (one variable, one constant). Active scalars are used inside
 * Eigen types and freely mixed with passive scalars. This gives access
 * to a large number of Eigen’s vector and matrix operations, which
 * are differentiated by TinyAD on a per-scalar level.
 * This corresponds to Figure 5 in the paper.
 */
int main()
{
    // Choose autodiff scalar type for 3 variables
    using ADouble = TinyAD::Double<3>;

    // Init a vector of active variables
    Eigen::Vector3<ADouble> x = ADouble::make_active({0.0, -1.0, 1.0});

    // Init a vector of passive variables
    Eigen::Vector3<double> y = {2.0, 3.0, 5.0};

    // Compute angle between the two vectors
    ADouble angle = acos(x.dot(y) / (x.norm() * y.norm()));

    // Retreive gradient and Hessian w.r.t. x
    Eigen::Vector3d g = angle.grad;
    Eigen::Matrix3d H = angle.Hess;

    TINYAD_INFO("angle: " << std::endl << angle.val);
    TINYAD_INFO("g: " << std::endl << g);
    TINYAD_INFO("H: " << std::endl << H);

    return 0;
}
