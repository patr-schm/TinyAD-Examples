/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#include <TinyAD/Scalar.hh>

/**
 * This snippet demonstrates the most basic TinyAD usage:
 * Choose the number of variables k, assign an index to each
 * variable, and perform any computations using operators
 * provided by TinyAD/Scalar.hh.
 * See angle.cc for more convenient initialization of active variables.
 */
int main()
{
    // Scalar type w.r.t. k = 2 variables
    using ADouble = TinyAD::Double<2>;

    ADouble x0(3.5, 0); // Variable with index 0
    ADouble x1(5.0, 1); // Variable with index 1
    ADouble z = -log(0.5 * sqrt(x0 * x0 + x1 * x1));

    TINYAD_INFO("z: " << std::endl << z.val);
    TINYAD_INFO("z.grad: " << std::endl << z.grad);
    TINYAD_INFO("z.Hess: " << std::endl << z.Hess);

    return 0;
}
