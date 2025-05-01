/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#define _USE_MATH_DEFINES // Required for M_PI on Windows

#include <igl/readOFF.h>
#include <igl/local_basis.h>
#include <igl/edge_topology.h>
#include <directional/read_raw_field.h>

#include <TinyAD/VectorFunction.hh>
#include <TinyAD/Utils/GaussNewtonDirection.hh>
#include <TinyAD/Utils/LinearSolver.hh>
#include <TinyAD/Utils/LineSearch.hh>
#include <TinyAD/Utils/Timer.hh>

#include <TinyAD-Examples/Filesystem.hh>
#include <TinyAD-Examples/GlowFrameFieldViewer.hh>

Eigen::VectorXd read_initial_x(
        const fs::path& _path,
        const Eigen::MatrixXd& _B1,
        const Eigen::MatrixXd& _B2)
{
    // Read field from file.
    int n = 0;
    Eigen::MatrixXd vector_field_4; // #F by 12 matrix: 4 3D vectors per face.
    directional::read_raw_field(_path.string(), n, vector_field_4);
    TINYAD_ASSERT_GEQ(n, 4);
    TINYAD_ASSERT_EQ(vector_field_4.rows(), _B1.rows());
    TINYAD_ASSERT_EQ(vector_field_4.rows(), _B2.rows());

    // Keep only the first two vectors per face. (The others are just symmetries.)
    // Express 3D vectors in local 2D coordinate system and stack them in a single vector x
    Eigen::VectorXd x = Eigen::VectorXd::Zero(4 * vector_field_4.rows());
    for (int f_idx = 0; f_idx < vector_field_4.rows(); ++f_idx)
    {
        x(4 * f_idx + 0) = _B1.row(f_idx).dot(vector_field_4.row(f_idx).segment(0, 3));
        x(4 * f_idx + 1) = _B2.row(f_idx).dot(vector_field_4.row(f_idx).segment(0, 3));
        x(4 * f_idx + 2) = _B1.row(f_idx).dot(vector_field_4.row(f_idx).segment(3, 3));
        x(4 * f_idx + 3) = _B2.row(f_idx).dot(vector_field_4.row(f_idx).segment(3, 3));
    }

    return x;
}

/**
 * Implementation of the frame field optimization algorithm
 * described in Integrable PolyVector Fields [Diamanti 2015].
 * Reduces curl of a 2-vector field on a triangle mesh.
 */
int main()
{
    // Init glow viewer
    glow::glfw::GlfwContext ctx;

    // Load mesh
    Eigen::MatrixXd V; // #V by 3
    Eigen::MatrixXi F; // #F by 3
    igl::readOFF((DATA_PATH / "cheburashka.off").string(), V, F);

    // Compute local basis per face
    Eigen::MatrixXd B1; // #F by 3. First basis vector per face.
    Eigen::MatrixXd B2; // #F by 3. Second basis vector per face.
    Eigen::MatrixXd N; // #F by 3. Face normals. Not used.
    igl::local_basis(V, F, B1, B2, N);

    // Compute mesh connectivity information
    Eigen::MatrixXi EV; // #E by 2. Vertex indices per edge.
    Eigen::MatrixXi FE; // #F by 3. Edges per face.
    Eigen::MatrixXi EF; // #E by 2. Faces per edge.
    igl::edge_topology(V, F, EV, FE, EF);
    for (int i = 0; i < EF.rows(); ++i) // Assert closed mesh
        TINYAD_ASSERT(EF(i, 0) != -1 && EF(i, 1) != -1);

    // Read initial field from file.
    // We represent a field as a vector of size 4 * #F,
    // where four consecutive entries are two tangent vectors in a local basis.
    Eigen::VectorXd x_init = read_initial_x(DATA_PATH / "cheburashka.rawfield", B1, B2);
    Eigen::VectorXd x = x_init;
    Eigen::VectorXd x_prev = x_init; // Track x of previous iteration

    // Compute constant transport terms for polynomial coefficients per edge
    Eigen::VectorX<std::complex<double>> e_f_conj(EF.rows()); // Normalized edge vector in local coordinate system of f (conjugated)
    Eigen::VectorX<std::complex<double>> e_g_conj(EF.rows()); // Normalized edge vector in local coordinate system of g (conjugated)
    Eigen::VectorX<std::complex<double>> t_fg_4(EF.rows()); // Transport term for 0-order coefficient
    Eigen::VectorX<std::complex<double>> t_fg_2(EF.rows()); // Transport term for 2nd-order coefficient
    for (int e_idx = 0; e_idx < EF.rows(); ++e_idx)
    {
        int v_from_idx = EV(e_idx, 0);
        int v_to_idx = EV(e_idx, 1);
        int f_idx = EF(e_idx, 0); // face f in paper
        int g_idx = EF(e_idx, 1); // face g in paper

        // Express normalized edge vector in local coordinate systems [Diamanti 2015, Eq. 6]
        Eigen::Vector3d e = (V.row(v_to_idx) - V.row(v_from_idx)).normalized();
        e_f_conj[e_idx] = conj(std::complex<double>(B1.row(f_idx).dot(e), B2.row(f_idx).dot(e)));
        e_g_conj[e_idx] = conj(std::complex<double>(B1.row(g_idx).dot(e), B2.row(g_idx).dot(e)));

        // Compute transport terms for polynomial coefficients [Diamanti 2015, Eq. 18]
        std::complex<double> t_fg = e_f_conj[e_idx] / e_g_conj[e_idx];
        t_fg_2[e_idx] = sqr(t_fg);
        t_fg_4[e_idx] = sqr(t_fg_2[e_idx]);
    }

    // Set up soft constraints (here only in face 0)
    const Eigen::VectorXd x_constr = x;
    std::vector<bool> face_constrained(F.rows(), false);
    face_constrained[0] = true;

    // Algorithm settings
    double w_smooth = 1.0;
    const double w_smooth_decay = 0.8;
    const double w_polycurl = 100.0; // Should be 10, but reference implementation in lib Directional is missing a square root
    const double w_polyquotient = 10.0;
    const double w_close_unconstrained = 1e-3;
    const double w_close_constrained = 100.0;
    const double w_barrier = 0.1;
    const double s_barrier = 0.9;

    // Set up objective function.
    // 4 variables (2 2D vectors) per face.
    auto func = TinyAD::vector_function<4>(TinyAD::range(F.rows()));

    // Add per-edge residuals. Each connecting two faces. Notation follows [Diamanti 2015]
    func.add_elements<2, 7>(TinyAD::range(EF.rows()), [&] (auto& element) -> TINYAD_VECTOR_TYPE(element)
    {
        // Element is evaluated with either double or TinyAD::Double<8>
        // and returns 7 scalar values
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Vector<T, 7> residuals = Eigen::Vector<T, 7>::Zero();

        // Get 2D vectors (alpha, beta) in local bases of faces f and g
        int e_idx = element.handle;
        int f_idx = EF(e_idx, 0);
        int g_idx = EF(e_idx, 1);
        Eigen::Vector4<T> vars_f = element.variables(f_idx); // 4 variables in face f
        Eigen::Vector4<T> vars_g = element.variables(g_idx); // 4 variables in face g
        std::complex<T> alpha_f(vars_f[0], vars_f[1]);
        std::complex<T> beta_f(vars_f[2], vars_f[3]);
        std::complex<T> alpha_g(vars_g[0], vars_g[1]);
        std::complex<T> beta_g(vars_g[2], vars_g[3]);

        // Smoothness term:
        // Compare complex coefficients of smoothness polynomial across edge [Diamanti 2015, Eq. 17, 18]
        std::complex<T> C_f_0 = sqr(alpha_f) * sqr(beta_f);
        std::complex<T> C_g_0 = sqr(alpha_g) * sqr(beta_g);
        std::complex<T> C_f_2 = -(sqr(alpha_f) + sqr(beta_f));
        std::complex<T> C_g_2 = -(sqr(alpha_g) + sqr(beta_g));
        std::complex<T> C_0_residual = C_f_0 * t_fg_4[e_idx] - C_g_0;
        std::complex<T> C_2_residual = C_f_2 * t_fg_2[e_idx] - C_g_2;
        double w_smooth_sqrt = sqrt(w_smooth);
        residuals[0] = w_smooth_sqrt * C_0_residual.real();
        residuals[1] = w_smooth_sqrt * C_0_residual.imag();
        residuals[2] = w_smooth_sqrt * C_2_residual.real();
        residuals[3] = w_smooth_sqrt * C_2_residual.imag();

        // PolyCurl term:
        // Compare real coefficients of polycurl polynomial across edge [Diamanti 2015, Eq. 11, 19]
        T af_ef = (alpha_f * e_f_conj[e_idx]).real();
        T ag_eg = (alpha_g * e_g_conj[e_idx]).real();
        T bf_ef = (beta_f * e_f_conj[e_idx]).real();
        T bg_eg = (beta_g * e_g_conj[e_idx]).real();
        T c_f_0 = sqr(af_ef) * sqr(bf_ef);
        T c_g_0 = sqr(ag_eg) * sqr(bg_eg);
        T c_f_2 = -(sqr(af_ef) + sqr(bf_ef));
        T c_g_2 = -(sqr(ag_eg) + sqr(bg_eg));
        residuals[4] = w_polycurl * (c_f_0 - c_g_0);
        residuals[5] = sqrt(w_polycurl) * (c_f_2 - c_g_2);

        // PolyQuotient term:
        // Compare real coefficients of polyquotient terms across edge [Diamanti 2015, Eq. 21]
        // There are typos in Eq. 14 and 21, where + should be -.
        T q1 = (sqr(bf_ef) - sqr(af_ef)) * ag_eg * bg_eg;
        T q2 = (sqr(bg_eg) - sqr(ag_eg)) * af_ef * bf_ef;
        residuals[6] = sqrt(w_polyquotient) * (q1 - q2);

        return residuals;
    });

    // Add per-face residuals.
    func.add_elements<1, 5>(TinyAD::range(F.rows()), [&] (auto& element) -> TINYAD_VECTOR_TYPE(element)
    {
        // Element is evaluated with either double or TinyAD::Double<4>
        // and returns 5 scalar values
        using T = TINYAD_SCALAR_TYPE(element);
        Eigen::Vector<T, 5> residuals = Eigen::Vector<T, 5>::Zero();

        // Get 2D vectors (alpha, beta) in local basis of face f
        int f_idx = element.handle;
        Eigen::Vector4<T> vars = element.variables(f_idx); // 4 variables in face f
        std::complex<T> alpha(vars[0], vars[1]);
        std::complex<T> beta(vars[2], vars[3]);

        // Closeness term:
        // Either soft penalty towards constraint or towards previous iteration.
        // Get reference vectors (alpha_ref, beta_ref).
        // Either from x_constr or from x_prev.
        const Eigen::VectorXd& x_ref = face_constrained[f_idx] ? x_constr : x_prev;
        std::complex<double> alpha_ref(x_ref[element.idx_local_to_global[0]], x_ref[element.idx_local_to_global[1]]);
        std::complex<double> beta_ref(x_ref[element.idx_local_to_global[2]], x_ref[element.idx_local_to_global[3]]);
        const double w_close_sqrt = face_constrained[f_idx] ? sqrt(w_close_constrained) : sqrt(w_close_unconstrained);
        residuals[0] = w_close_sqrt * (alpha.real() - alpha_ref.real());
        residuals[1] = w_close_sqrt * (alpha.imag() - alpha_ref.imag());
        residuals[2] = w_close_sqrt * (beta.real() - beta_ref.real());
        residuals[3] = w_close_sqrt * (beta.imag() - beta_ref.imag());

        // Barrier term:
        // Ensure convex angle between alpha and beta
        T barrier_x = (beta * conj(alpha)).imag(); // Needs to stay positive
        T barrier = 0.0;
        if (barrier_x <= 0.0)
            barrier = INFINITY;
        else if (barrier_x < s_barrier)
        {
            T b = barrier_x * sqr(barrier_x) / (s_barrier * sqr(s_barrier))
                - 3.0 * sqr(barrier_x) / sqr(s_barrier)
                + 3.0 * barrier_x / s_barrier;
            barrier = 1.0 / b - 1.0;
        }
        residuals[4] = sqrt(w_barrier) * barrier;

        return residuals;
    });

    // Solve via Gauss-Newton
    double f; // Objective value
    Eigen::VectorXd g; // Gradient
    Eigen::VectorXd r; // Residuals vector
    Eigen::SparseMatrix<double> J; // Residuals Jacobian
    Eigen::VectorXd d; // Update direction
    TinyAD::LinearSolver solver; // Linear solver storing pre-factorization
    double step_size = 0.1; // Adaptive step size
    const int max_iters = 60;
    for (int iter = 0; iter < max_iters; ++iter)
    {
        // Compute derivatives and Gauss-Newton direction
        func.eval_sum_of_squares_with_derivatives(x, f, g, r, J);
        TINYAD_DEBUG_OUT("Energy in iteration " << iter << ": " << f);
        d = TinyAD::gauss_newton_direction(r, J, solver);

        // Line search (inspired by Directional/tutorial/303_PolyCurlReduction/main.cpp)
        Eigen::VectorXd x_new;
        double f_new;
        while (true)
        {
            x_new = x + step_size * d;
            f_new = func.eval_sum_of_squares(x_new);
            // TINYAD_DEBUG_OUT("Line search: s = " << step_size << ", F = " << f_new);

            if (f_new < f)
            {
                // Line search success. Increase step size in next iteration.
                step_size *= 2.0;
                break;
            }
            else
            {
                // Decrease step size and try again.
                step_size *= 0.5;
                if (step_size < 1e-30)
                    TINYAD_ERROR_throw("Line search failed.");
            }
        }

        x_prev = x;
        x = x_new;

        // Decay smoothness term after every 5 iters
        if ((iter + 1) % 5 == 0)
            w_smooth *= w_smooth_decay;
    }

    // View initial and optimized frame fields with polycurl heatmap
    auto grid = gv::grid();
    glow_view_frame_field(V, F, B1, B2, x_init, 3000, 0.2, 0.00225, false, true);
    glow_view_frame_field(V, F, B1, B2, x, 3000, 0.2, 0.00225, false, true);
    glow_view_polycurl(V, F, x_init, EV, EF, e_f_conj, e_g_conj, 1e-6, 0.02);
    glow_view_polycurl(V, F, x, EV, EF, e_f_conj, e_g_conj, 1e-6, 0.02);

    return 0;
}
