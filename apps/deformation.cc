/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 * Mesh and constraints are from libigl/tutorial/709_SLIM
 */
#define _USE_MATH_DEFINES // Required for M_PI on Windows

#include <igl/readOBJ.h>

#include <TinyAD/ScalarFunction.hh>
#include <TinyAD/Utils/NewtonDecrement.hh>
#include <TinyAD/Utils/LineSearch.hh>
#include <TinyAD/Utils/Timer.hh>
#include <Eigen/IterativeLinearSolvers>

#include <TinyAD-Examples/GlowViewerIGL.hh>

enum Energy { SD, EXP_SD, AMIPS, CONF_AMIPS };

/**
 * Compute 3D deformation of tet mesh w.r.t. soft position constraints.
 * Mesh and constraints are from libigl/tutorial/709_SLIM.
 */
void deformation(
        const Energy& _energy)
{
    // Objective settings
    const double w_penalty = 1e5;

    // Load mesh and constraints from libigl example (libigl/tutorial/709_SLIM)
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    igl::readOBJ((DATA_PATH / "cube_48k.obj").string(), V, F);
    Eigen::VectorXi b(84); b << 440, 881, 1322, 1763, 2204, 2645, 3086, 3527, 3968, 4409, 4850, 5291, 5732, 6173, 6614, 7055, 7496, 7937, 8378, 8819, 9260, 0, 441, 882, 1323, 1764, 2205, 2646, 3087, 3528, 3969, 4410, 4851, 5292, 5733, 6174, 6615, 7056, 7497, 7938, 8379, 8820, 20, 461, 902, 1343, 1784, 2225, 2666, 3107, 3548, 3989, 4430, 4871, 5312, 5753, 6194, 6635, 7076, 7517, 7958, 8399, 8840, 420, 861, 1302, 1743, 2184, 2625, 3066, 3507, 3948, 4389, 4830, 5271, 5712, 6153, 6594, 7035, 7476, 7917, 8358, 8799, 9240;
    Eigen::MatrixXd bc(84, 3); bc << -1.28603, -0.713969, -0.206107, -1.25743, -0.742572, -0.235497, -1.22882, -0.771175, -0.264886, -1.20022, -0.799779, -0.294275, -1.17162, -0.828382, -0.323664, -1.14302, -0.856985, -0.353054, -1.11441, -0.885588, -0.382443, -1.08581, -0.914191, -0.411832, -1.05721, -0.942794, -0.441221, -1.0286, -0.971397, -0.470611, -1, -1, -0.5, -0.971397, -1.0286, -0.529389, -0.942794, -1.05721, -0.558779, -0.914191, -1.08581, -0.588168, -0.885588, -1.11441, -0.617557, -0.856985, -1.14302, -0.646946, -0.828382, -1.17162, -0.676336, -0.799779, -1.20022, -0.705725, -0.771175, -1.22882, -0.735114, -0.742572, -1.25743, -0.764503, -0.713969, -1.28603, -0.793893, 0.286031, -0.286031, -0.206107, 0.257428, -0.257428, -0.235497, 0.228825, -0.228825, -0.264886, 0.200221, -0.200221, -0.294275, 0.171618, -0.171618, -0.323664, 0.143015, -0.143015, -0.353054, 0.114412, -0.114412, -0.382443, 0.0858092, -0.0858092, -0.411832, 0.0572061, -0.0572061, -0.441221, 0.0286031, -0.0286031, -0.470611, 0, 0, -0.5, -0.0286031, 0.0286031, -0.529389, -0.0572061, 0.0572061, -0.558779, -0.0858092, 0.0858092, -0.588168, -0.114412, 0.114412, -0.617557, -0.143015, 0.143015, -0.646946, -0.171618, 0.171618, -0.676336, -0.200221, 0.200221, -0.705725, -0.228825, 0.228825, -0.735114, -0.257428, 0.257428, -0.764503, -0.286031, 0.286031, -0.793893, -0, -0.75, -0, -0, -0.75, -0.0375, -0, -0.75, -0.075, -0, -0.75, -0.1125, -0, -0.75, -0.15, -0, -0.75, -0.1875, -0, -0.75, -0.225, -0, -0.75, -0.2625, -0, -0.75, -0.3, -0, -0.75, -0.3375, -0, -0.75, -0.375, -0, -0.75, -0.4125, -0, -0.75, -0.45, -0, -0.75, -0.4875, -0, -0.75, -0.525, -0, -0.75, -0.5625, -0, -0.75, -0.6, -0, -0.75, -0.6375, -0, -0.75, -0.675, -0, -0.75, -0.7125, -0, -0.75, -0.75, -0.75, -0, -0, -0.75, -0, -0.0375, -0.75, -0, -0.075, -0.75, -0, -0.1125, -0.75, -0, -0.15, -0.75, -0, -0.1875, -0.75, -0, -0.225, -0.75, -0, -0.2625, -0.75, -0, -0.3, -0.75, -0, -0.3375, -0.75, -0, -0.375, -0.75, -0, -0.4125, -0.75, -0, -0.45, -0.75, -0, -0.4875, -0.75, -0, -0.525, -0.75, -0, -0.5625, -0.75, -0, -0.6, -0.75, -0, -0.6375, -0.75, -0, -0.675, -0.75, -0, -0.7125, -0.75, -0, -0.75;

    // Pre-compute tetrahedral rest shapes
    std::vector<Eigen::Matrix3d> rest_shapes(F.rows());
    for (int t_idx = 0; t_idx < F.rows(); ++t_idx)
    {
        // Get 3D vertex positions
        Eigen::Vector3d ar = V.row(F(t_idx, 0));
        Eigen::Vector3d br = V.row(F(t_idx, 1));
        Eigen::Vector3d cr = V.row(F(t_idx, 2));
        Eigen::Vector3d dr = V.row(F(t_idx, 3));

        // Save 3-by-3 matrix with edge vectors as colums
        rest_shapes[t_idx] = TinyAD::col_mat(br - ar, cr - ar, dr - ar);
    };

    // Set up function with 3D vertex positions as variables.
    auto func = TinyAD::scalar_function<3>(TinyAD::range(V.rows()));

    // Add objective term per tetrahedron. Each connecting 4 vertices.
    func.add_elements<4>(TinyAD::range(F.rows()), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        // Get variable 3d vertex positions and check injectivity
        using T = TINYAD_SCALAR_TYPE(element);
        int t_idx = element.handle;
        Eigen::Vector3<T> a = element.variables(F(t_idx, 0));
        Eigen::Vector3<T> b = element.variables(F(t_idx, 1));
        Eigen::Vector3<T> c = element.variables(F(t_idx, 2));
        Eigen::Vector3<T> d = element.variables(F(t_idx, 3));
        Eigen::Matrix3<T> M = TinyAD::col_mat(b - a, c - a, d - a);
        if (M.determinant() <= 0.0)
            return (T)INFINITY;

        // Compute tet Jacobian and energy
        Eigen::Matrix3d Mr = rest_shapes[t_idx];
        Eigen::Matrix3<T> J = M * Mr.inverse();
        double vol = Mr.determinant() / 6.0;
        if (_energy == SD)
            return vol * (J.squaredNorm() + J.inverse().squaredNorm());
        else if (_energy == EXP_SD)
            return vol * exp((J.squaredNorm() + J.inverse().squaredNorm()));
        else if (_energy == AMIPS)
            return vol * exp(0.5 * (J.squaredNorm() / J.determinant()
                               + 0.5 * (J.determinant() + 1.0 / J.determinant())));
        else if (_energy == CONF_AMIPS)
            return vol * J.squaredNorm() / pow(J.determinant(), 2.0 / 3.0);
        else
            TINYAD_ERROR_throw("");
    });

    // Add penalty term per constrained vertex.
    func.add_elements<1>(TinyAD::range(b.size()), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        // Evaluate element using either double or TinyAD::Double
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Vector3<T> p = element.variables(b[element.handle]);
        Eigen::Vector3d p_target = bc.row(element.handle);
        return w_penalty * (p_target - p).squaredNorm();
    });

    // Assemble inital x vector using the data in V.
    // x_from_data(...) takes a lambda function that maps
    // each variable handle (int) to its initial 3D value (Eigen::Vector3d).
    Eigen::VectorXd x = func.x_from_data([&] (int v_idx) { return V.row(v_idx); });

    // Projected Newton with conjugate gradient solver
    int max_iters = 1000;
    double convergence_eps = 1e-4;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> cg_solver;
    for (int i = 0; i < max_iters; ++i)
    {
        auto [f, g, H_proj] = func.eval_with_hessian_proj(x);
        TINYAD_DEBUG_OUT("Energy in iteration " << i << ": " << f);
        Eigen::VectorXd d = cg_solver.compute(H_proj + 1e-4 * TinyAD::identity<double>(x.size())).solve(-g);
        if (TinyAD::newton_decrement(d, g) < convergence_eps)
            break;
        x = TinyAD::line_search(x, d, f, g, func);
    }
    TINYAD_DEBUG_OUT("Final energy: " << func.eval(x));

    // Write final x vector to V for visualization.
    // x_to_data(...) takes a lambda function that writes the value
    // of each variable (Eigen::Vector3d) back our data structure.
    func.x_to_data(x, [&] (int v_idx, const Eigen::Vector3d& p) { V.row(v_idx) = p; });

    // Open viewer
    glow_view_mesh(V, F, true);
}

int main()
{
    // Init glow viewer
    glow::glfw::GlfwContext ctx;

    deformation(SD);
    deformation(EXP_SD);
    deformation(AMIPS);
    deformation(CONF_AMIPS);

    return 0;
}
