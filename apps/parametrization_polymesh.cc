/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#define _USE_MATH_DEFINES // Required for M_PI on Windows

#include <polymesh/Mesh.hh>
#include <polymesh/formats.hh>
#include <typed-geometry/tg-lean.hh>

#include <TinyAD/Support/Polymesh.hh>
#include <TinyAD/ScalarFunction.hh>
#include <TinyAD/Utils/NewtonDirection.hh>
#include <TinyAD/Utils/NewtonDecrement.hh>
#include <TinyAD/Utils/LineSearch.hh>

#include <TinyAD-Examples/Filesystem.hh>
#include <TinyAD-Examples/GlowViewerCommon.hh>
#include <TinyAD-Examples/TutteEmbeddingPolymesh.hh>

/**
 * Injectively map a disk-topology triangle mesh to the plane
 * and optimize the symmetric Dirichlet energy via projected Newton.
 */
int main()
{
    // Init glow viewer
    glow::glfw::GlfwContext ctx;
    auto g = gv::grid(); // Create grid. Viewer opens on destructor.

    // Read disk-topology mesh and compute Tutte embedding
    pm::Mesh m;
    auto pos = m.vertices().make_attribute<Eigen::Vector3d>();
    pm::load((DATA_PATH / "armadillo_cut_low.obj").string(), m, pos);
    auto param = tutte_embedding(pos);

    glow_view_mesh(pos, true, "Input Mesh"); // Add input mesh to viewer gird
    glow_view_param(param, "Initial Parametrization"); // Add initial param to viewer grid

    // Pre-compute triangle rest shapes in local coordinate systems
    auto rest_shapes = m.faces().make_attribute<Eigen::Matrix2d>();
    for (auto t : m.faces())
    {
        // Get 3D vertex positions
        Eigen::Vector3d ar_3d(pos[t.any_halfedge().vertex_to()]);
        Eigen::Vector3d br_3d(pos[t.any_halfedge().next().vertex_to()]);
        Eigen::Vector3d cr_3d(pos[t.any_halfedge().vertex_from()]);

        // Set up local 2D coordinate system
        Eigen::Vector3d n = (br_3d - ar_3d).cross(cr_3d - ar_3d);
        Eigen::Vector3d b1 = (br_3d - ar_3d).normalized();
        Eigen::Vector3d b2 = n.cross(b1).normalized();

        // Express a, b, c in local 2D coordiante system
        Eigen::Vector2d ar_2d(0.0, 0.0);
        Eigen::Vector2d br_2d((br_3d - ar_3d).dot(b1), 0.0);
        Eigen::Vector2d cr_2d((cr_3d - ar_3d).dot(b1), (cr_3d - ar_3d).dot(b2));

        // Save 2-by-2 matrix with edge vectors as colums
        rest_shapes[t] = TinyAD::col_mat(br_2d - ar_2d, cr_2d - ar_2d);
    };

    // Set up function with 2D vertex positions as variables.
    auto func = TinyAD::scalar_function<2>(m.vertices());

    // Add objective term per triangle. Each connecting 3 vertices.
    func.add_elements<3>(m.faces(), [&] (auto& element)  -> TINYAD_SCALAR_TYPE(element)
    {
        // Evaluate element using either double or TinyAD::Double
        using T = TINYAD_SCALAR_TYPE(element);

        // Get variable 2D vertex positions
        pm::face_handle t = element.handle;
        Eigen::Vector2<T> a = element.variables(t.any_halfedge().vertex_to());
        Eigen::Vector2<T> b = element.variables(t.any_halfedge().next().vertex_to());
        Eigen::Vector2<T> c = element.variables(t.any_halfedge().vertex_from());

        // Triangle flipped?
        Eigen::Matrix2<T> M = TinyAD::col_mat(b - a, c - a);
        if (M.determinant() <= 0.0)
            return (T)INFINITY;

        // Get constant 2D rest shape of t
        Eigen::Matrix2d Mr = rest_shapes[t];
        double A = 0.5 * Mr.determinant();

        // Compute symmetric Dirichlet energy
        Eigen::Matrix2<T> J = M * Mr.inverse();
        return A * (J.squaredNorm() + J.inverse().squaredNorm());
    });

    // Assemble inital x vector from parametrization attribute.
    // x_from_data(...) takes a lambda function that maps
    // each variable handle (pm::vertex_handle) to its initial 2D value (Eigen::Vector2d).
    Eigen::VectorXd x = func.x_from_data([&] (pm::vertex_handle v) {
        return param[v];
    });

    // Projected Newton
    TinyAD::LinearSolver solver;
    int max_iters = 1000;
    double convergence_eps = 1e-2;
    for (int i = 0; i < max_iters; ++i)
    {
        auto [f, g, H_proj] = func.eval_with_hessian_proj(x);
        TINYAD_DEBUG_OUT("Energy in iteration " << i << ": " << f);
        Eigen::VectorXd d = TinyAD::newton_direction(g, H_proj, solver);
        if (TinyAD::newton_decrement(d, g) < convergence_eps)
            break;
        x = TinyAD::line_search(x, d, f, g, func);
    }
    TINYAD_DEBUG_OUT("Final energy: " << func.eval(x));

    // Write final x vector to parametrization property.
    // x_to_data(...) takes a lambda function that writes the final value
    // of each variable (Eigen::Vector2d) back our parametrization attribute.
    func.x_to_data(x, [&] (pm::vertex_handle v, const Eigen::Vector2d& p) {
        param[v] = p;
    });

    // Add resulting param to viewer grid
    glow_view_param(param, "Optimized Parametrization");

	return 0;
}
