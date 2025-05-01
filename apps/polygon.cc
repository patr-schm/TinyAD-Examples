/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#define _USE_MATH_DEFINES // Required for M_PI on Windows

#include <TinyAD/Scalar.hh>
#include <TinyAD/Utils/Helpers.hh>
#include <TinyAD/Utils/LineSearch.hh>
#include <TinyAD/Utils/HessianProjection.hh>

#include <TinyAD-Examples/GlowViewerCommon.hh>

/**
 * Polygon deformation with optimal triangulation. A distortion energy between
 * a rest state polygon and its deformed version is evaluated per triangle.
 * In each evaluation we choose from all possible triangulations the one that
 * causes least distortion.
 * This corresponds to Figure 6 in the paper.
 */

/**
 * Define distortion energy (symmetric Dirichlet) per triangle (a, b, c).
 * T is the scalar type (either double or TinyAD::Double).
 * k is the number of variables (2 * #vertices).
 */
template <typename T, int k>
T triangle_distortion(
        const Eigen::Vector<T, k>& _x,
        const Eigen::Vector<double, k>& _x_rest,
        const int a_idx, const int b_idx, const int c_idx)
{
    Eigen::Vector2<T> a = _x.segment(2 * a_idx, 2);
    Eigen::Vector2<T> b = _x.segment(2 * b_idx, 2);
    Eigen::Vector2<T> c = _x.segment(2 * c_idx, 2);
    Eigen::Vector2d ar = _x_rest.segment(2 * a_idx, 2);
    Eigen::Vector2d br = _x_rest.segment(2 * b_idx, 2);
    Eigen::Vector2d cr = _x_rest.segment(2 * c_idx, 2);

    Eigen::Matrix2<T> M = TinyAD::col_mat(b - a, c - a);
    Eigen::Matrix2d M_rest = TinyAD::col_mat(br - ar, cr - ar);

    if (M.determinant() <= 0)
        return INFINITY;

    Eigen::Matrix2<T> J = M * M_rest.inverse();
    const double A = 0.5 * M_rest.determinant();
    return A * J.squaredNorm() + A * J.inverse().squaredNorm();
}

/**
 * Helper function for optimal_triangulation().
 */
void gather_triangles(
        const int i_from, const int i_to, const int n_vertices,
        const Eigen::MatrixXi& k_chosen,
        Eigen::MatrixXi& _F)
{
    TINYAD_ASSERT_GEQ(i_from, 0);
    TINYAD_ASSERT_L(i_from, n_vertices);
    TINYAD_ASSERT_GEQ(i_to, 0);
    TINYAD_ASSERT_L(i_to, n_vertices);
    TINYAD_ASSERT_L(i_from, i_to);

    if (i_from + 1 >= i_to)
        return;

    const int k = k_chosen(i_from, i_to);
    TINYAD_ASSERT_G(k, i_from);
    TINYAD_ASSERT_L(k, i_to);

    _F.conservativeResize(_F.rows() + 1, 3);
    _F.row(_F.rows() - 1) = Eigen::Vector3i(i_from, k, i_to);

    gather_triangles(i_from, k, n_vertices, k_chosen, _F);
    gather_triangles(k, i_to, n_vertices, k_chosen, _F);
}

/**
 * Compute optimal triangulation w.r.t. distortion measure.
 * k is the number of variables (2 * #vertices).
 * x is of passive type because we don't want to differentiate
 *   this discrete algorithm.
 */
template <int k>
Eigen::MatrixXi optimal_triangulation(
        const Eigen::Vector<double, k>& _x,
        const Eigen::Vector<double, k>& _x_rest)
{
    // Optimal triangulation w.r.t. map distortion via dynamic programming.
    const int n_vertices = k / 2;
    TINYAD_ASSERT_GEQ(n_vertices, 3);
    TINYAD_ASSERT_LEQ(n_vertices, 6);
    if (n_vertices == 3)
        return Eigen::Vector3i(0, 1, 2);

    Eigen::MatrixXd cost = Eigen::MatrixXd::Zero(n_vertices, n_vertices);
    Eigen::MatrixXi chosen = Eigen::MatrixXi::Constant(n_vertices, n_vertices, -1);

    // Fill upper right block diagonal-by-diagonal.
    // Gap is offset from main diagonal.
    for (int gap = 2; gap < n_vertices; ++gap)
    {
        // Row i_from, col i_to
        for (int i_from = 0; i_from < n_vertices - gap; ++i_from)
        {
            // Find optimal triangulation between i_from and i_to.
            // Triangle (i_from, i_between, i_to) and optimal triangulations
            // i_from to i_between and i_between to i_to.
            const int i_to = i_from + gap;
            cost(i_from, i_to) = INFINITY;
            for (int i_between = i_from + 1; i_between < i_to; ++i_between)
            {
                const double c = cost(i_from, i_between) + triangle_distortion(_x, _x_rest, i_from, i_between, i_to) + cost(i_between, i_to);
                if (c <= cost(i_from, i_to)) // <= because cost can be infinite
                {
                    cost(i_from, i_to) = c;
                    chosen(i_from, i_to) = i_between;
                }
            }
        }
    }

    // Gather chosen triangles from table
    Eigen::MatrixXi F(0, 3);
    gather_triangles(0, n_vertices-1, n_vertices, chosen, F);

    return F;
}

/**
 * Penalty term with per-vertex deformation forces.
 * T is the scalar type (either double or TinyAD::Double).
 * k is the number of variables (2 * #vertices).
 */
template <typename T, int k>
T penalty(
        const Eigen::Vector<T, k>& _x)
{
    T f = 0.0;

    f += (_x.segment(0, 2) - Eigen::Vector2d(-2.0, -1.0)).squaredNorm();
    f += (_x.segment(2, 2) - Eigen::Vector2d(2.0, -1.0)).squaredNorm();
    f += (_x.segment(6, 2) - Eigen::Vector2d(0.0, -1.0)).squaredNorm();
    f += (_x.segment(8, 2) - Eigen::Vector2d(-4.0, 1.0)).squaredNorm();

    return f;
}

/**
 * Differentiable objective function.
 * First computes the optimal triangulation w.r.t. the distortion measure,
 * then evaluates the distortion measure in a differentiable way.
 * T is the scalar type (either double or TinyAD::Double).
 * k is the number of variables (2 * #vertices).
 */
template <typename T, int k>
T objective(
        const Eigen::Vector<T, k>& _x,
        const Eigen::Vector<double, k>& _x_rest)
{
    Eigen::MatrixXi F = optimal_triangulation(TinyAD::to_passive(_x), _x_rest);

    T f = 0.0;
    for (int i = 0; i < (int)F.rows(); ++i)
        f += triangle_distortion(_x, _x_rest, F(i, 0), F(i, 1), F(i, 2));

    f += penalty(_x);

    return f;
}

/**
 * Choose initial polygon shape.
 * k is the number of variables (2 * #vertices).
 */
template <int k>
Eigen::Vector<double, k> initial_x()
{
    Eigen::Vector<double, k> x;
    for (int i = 0; i < 2 * k; ++i)
    {
        const double angle = (double)i / (double)(k / 2) * 2.0 * M_PI;
        x.segment(2 * i, 2) = Eigen::Vector2d(cos(angle), sin(angle));
    }

    return x;
}

/**
 * Render the polygon in state _x (vertex positions) and triangulation _F.
 * k is the number of variables (2 * #vertices).
 */
template <int k>
void view_polygon(
    const Eigen::Vector<double, k>& _x,
    const Eigen::MatrixXi& _F)
{
    // Center mesh at origin
    const int n_vertices = k / 2;
    Eigen::MatrixXd V(n_vertices, 3);
    for (int i = 0; i < n_vertices; ++i)
        V.row(i) = _x.segment(2 * i, 2);
    Eigen::Vector2d cog = V.colwise().mean();
    V.rowwise() -= cog.transpose();

    // Render
    auto canvas = gv::canvas();
    canvas.set_point_size_world(0.04);
    canvas.set_line_width_world(0.04);
    for (int f_idx = 0; f_idx < _F.rows(); ++f_idx)
    {
        tg::dpos3 a(V(_F(f_idx, 0), 0), 0, -V(_F(f_idx, 0), 1));
        tg::dpos3 b(V(_F(f_idx, 1), 0), 0, -V(_F(f_idx, 1), 1));
        tg::dpos3 c(V(_F(f_idx, 2), 0), 0, -V(_F(f_idx, 2), 1));
        canvas.set_color(BLUE);
        canvas.add_point(a);
        canvas.add_point(b);
        canvas.add_point(c);
        canvas.add_line(a, b);
        canvas.add_line(b, c);
        canvas.add_line(c, a);
        canvas.set_color(tg::mix(BLACK, WHITE, 0.85));
        canvas.add_face(a, b, c);
    }
}

/**
 * Render a sequence of previously recorded optimization states.
 */
template <int k>
void view_iters(
    const Eigen::Vector<double, k>& _x_rest,
    const std::vector<Eigen::Vector<double, k>>& _x_history,
    const std::vector<Eigen::MatrixXi>& _F_history,
    const std::vector<int>& _iters)
{
    TINYAD_ASSERT_EQ(_x_history.size(), _F_history.size());

    auto style = glow_default_style();
    auto no_shadow = gv::config(gv::no_shadow);
    auto cam_pos = gv::config(glow::viewer::camera_orientation(0_deg, 90_deg, 4.0));
    auto grid_cols = gv::columns();

    for (const int iter : _iters)
    {
        const auto& x = _x_history[iter];
        const auto& F = _F_history[iter];

        auto grid_rows = gv::rows();
        {
            auto view = gv::view();
            view_polygon(_x_rest, F);
        }
        {
            auto view = gv::view();
            view_polygon(x, F);
        }
    }
}

int main()
{
    // Init glow viewer
    glow::glfw::GlfwContext ctx;

    // Choose number of polygon vertices at compile time
    constexpr int n_vertices = 6;
    constexpr int k = 2 * n_vertices;

    // Choose initial polygon shape
    Eigen::Vector<double, k> x_rest = initial_x<k>();
    Eigen::Vector<double, k> x = x_rest;
    Eigen::MatrixXi F = optimal_triangulation(x, x_rest);

    // Record optimization history for later visualization
    std::vector<Eigen::Vector<double, k>> x_history;
    std::vector<Eigen::MatrixXi> F_history;

    // Optimize polygon shape w.r.t. external penalty forces
    const int max_iters = 20;
    for (int iter = 0; iter < max_iters; ++iter)
    {
        x_history.push_back(x);
        F_history.push_back(F);

        TinyAD::Double<k> f = objective(TinyAD::Double<k>::make_active(x), x_rest);
        TINYAD_DEBUG_OUT("Energy in iteration " << iter << ": " << f.val);
        TinyAD::project_positive_definite(f.Hess, 1e-6);
        Eigen::Vector<double, k> d = -f.Hess.inverse() * f.grad;
        x = TinyAD::line_search(x, d, f.val, f.grad, [&] (const Eigen::Vector<double, k>& _x) { return objective(_x, x_rest); }, 0.1);
        F = optimal_triangulation(x, x_rest);
    }

    // Visualization
    view_iters(x_rest, x_history, F_history, {0, 4, 6, 9, 19});

    return 0;
}
