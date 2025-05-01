/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#define _USE_MATH_DEFINES // Required for M_PI on Windows

#include <igl/readOBJ.h>

#include <TinyAD/ScalarFunction.hh>
#include <TinyAD/Utils/NewtonDirection.hh>
#include <TinyAD/Utils/NewtonDecrement.hh>
#include <TinyAD/Utils/LineSearch.hh>
#include <TinyAD/Utils/Timer.hh>

#include <TinyAD-Examples/Filesystem.hh>
#include <TinyAD-Examples/GlowViewerIGL.hh>

/**
 * Compute an abitrary tangent vector of the sphere at position p.
 */
Eigen::Vector3d any_tangent(
        const Eigen::Vector3d& _p)
{
    // Find coordinate axis spanning the largest angle with _p.
    // Return cross product that of axis with _p
    Eigen::Vector3d tang;
    double min_abs_dot = INFINITY;
    for (const Eigen::Vector3d& ax : { Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Vector3d(0.0, 1.0, 0.0), Eigen::Vector3d(0.0, 0.0, 1.0) })
    {
        double abs_dot = fabs(_p.dot(ax));
        if (abs_dot < min_abs_dot)
        {
            min_abs_dot = abs_dot;
            tang = ax.cross(_p).normalized();
        }
    }

    return tang;
}

/**
 * Compute an orthonormal tangent space basis of the sphere at each vertex.
 */
void compute_local_bases(
        const Eigen::MatrixX3d& _S,
        Eigen::MatrixX3d& _B1,
        Eigen::MatrixX3d& _B2)
{
    for (int v_idx = 0; v_idx < _S.rows(); ++v_idx)
    {
        _B1.row(v_idx) = any_tangent(_S.row(v_idx));
        _B2.row(v_idx) = _S.row(v_idx).cross(_B1.row(v_idx));
    }
}

/**
 * Optimize embedding of genus 0 surface on the sphere.
 */
void manifold_optimization(
        const std::string& _name)
{
    // Read triangle mesh and initial sphere embedding
    Eigen::MatrixX3d V; // #V-by-3 3D vertex positions
    Eigen::MatrixX3i F; // #F-by-3 indices into V
    Eigen::MatrixX3d S; // #V-by-3 3D vertex positions on sphere
    Eigen::MatrixX3i F2; // #F-by-3 indices into V
    igl::readOBJ((DATA_PATH / "giraffe_refined.obj").string(), V, F);
    igl::readOBJ((DATA_PATH / "giraffe_embedding.obj").string(), S, F2);
    TINYAD_ASSERT_EQ(F.rows(), F2.rows());

    // Set up orthonormal tangent space basis at each vertex on sphere.
    Eigen::MatrixX3d B1(V.rows(), 3);
    Eigen::MatrixX3d B2(V.rows(), 3);
    compute_local_bases(S, B1, B2);

    // Variable vector x constains a 2d tangent vector per vertex.
    Eigen::VectorXd x = Eigen::VectorXd::Zero(2 * V.rows());

    // Retraction operator: map from a local tangent space to the sphere.
    auto retract = [&] (const auto& v_tang, int v_idx)
    {
        // Evaluate target point in 3D ambient space and project to sphere via normalization.
        return (S.row(v_idx) + v_tang[0] * B1.row(v_idx) + v_tang[1] * B2.row(v_idx)).normalized().eval();
    };

    // Set up function with a 2D tangent vector per vertex as variables.
    auto func = TinyAD::scalar_function<2>(TinyAD::range(V.rows()));

    // Add objective term per triangle, each connecting 3 vertices.
    func.add_elements<3>(TinyAD::range(F.rows()), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        // Get 2D tangent vectors at vertices a, b, c.
        using T = TINYAD_SCALAR_TYPE(element);
        int f_idx = element.handle;
        Eigen::Vector2<T> a_tang = element.variables(F(f_idx, 0));
        Eigen::Vector2<T> b_tang = element.variables(F(f_idx, 1));
        Eigen::Vector2<T> c_tang = element.variables(F(f_idx, 2));

        // Retract 2D tangent vectors to 3D points on the sphere.
        Eigen::Vector3<T> a_mani = retract(a_tang, F(f_idx, 0));
        Eigen::Vector3<T> b_mani = retract(b_tang, F(f_idx, 1));
        Eigen::Vector3<T> c_mani = retract(c_tang, F(f_idx, 2));

        // Objective: injectivity barrier + Dirichlet energy
        T volume = (1.0 / 6.0) * TinyAD::col_mat(a_mani, b_mani, c_mani).determinant();
        if (volume <= 0.0)
            return (T)INFINITY;
        T E = -0.1 * log(volume);
        E += (a_mani - b_mani).squaredNorm() + (b_mani - c_mani).squaredNorm() + (c_mani - a_mani).squaredNorm();

        if (_name == "equator")
            E += sqr(a_mani.y()) + sqr(b_mani.y()) + sqr(c_mani.y());
        else if (_name == "northpole")
            E += sqr(1.0 - a_mani.y()) + sqr(1.0 - b_mani.y()) + sqr(1.0 - c_mani.y());

        return E;
    });

    // Projected-Newton with re-centering
    int max_iters = 1000;
    double convergence_eps = 1e-1;
    TinyAD::LinearSolver solver; // Stores pre-factorization
    for (int iter = 0; iter < max_iters; ++iter)
    {
        auto [f, g, H_proj] = func.eval_with_hessian_proj(x);
        TINYAD_DEBUG_OUT("Energy in iteration " << iter << ": " << f);
        Eigen::VectorXd d = TinyAD::newton_direction(g, H_proj, solver, 1e-9);
        if (TinyAD::newton_decrement(d, g) < convergence_eps)
            break;
        x = TinyAD::line_search(x, d, f, g, func);

        // Re-center local bases
        for (int v_idx = 0; v_idx < S.rows(); ++v_idx)
            S.row(v_idx) = retract(x.segment(2 * v_idx, 2), v_idx);
        compute_local_bases(S, B1, B2);
        x = Eigen::VectorXd::Zero(2 * V.rows());
    }
    TINYAD_DEBUG_OUT("Final energy: " << func.eval(x));

    // View result
    auto g = gv::grid();
    glow_view_mesh(V, F, true);
    glow_view_sphere_embedding(S, F);
}

int main()
{
    // Init glow viewer
    glow::glfw::GlfwContext ctx;

    manifold_optimization("default");
    manifold_optimization("equator");
    manifold_optimization("northpole");

    return 0;
}
