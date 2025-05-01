/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#define _USE_MATH_DEFINES // Required for M_PI on Windows

#include <geometrycentral/surface/meshio.h>
#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>

#include <TinyAD/Support/GeometryCentral.hh>
#include <TinyAD/ScalarFunction.hh>
#include <TinyAD/Utils/NewtonDirection.hh>
#include <TinyAD/Utils/NewtonDecrement.hh>
#include <TinyAD/Utils/LineSearch.hh>

#include <TinyAD-Examples/Filesystem.hh>
#include <TinyAD-Examples/TutteEmbeddingGeometryCentral.hh>

/**
 * Injectively map a disk-topology triangle mesh to the plane
 * and optimize the symmetric Dirichlet energy via projected Newton.
 */
int main()
{
    using namespace geometrycentral;
    using namespace geometrycentral::surface;

    // Initialize polyscope
    polyscope::init();

    // Read mesh and compute Tutte embedding
    std::unique_ptr<ManifoldSurfaceMesh> mesh;
    std::unique_ptr<VertexPositionGeometry> geometry;
    std::tie(mesh, geometry) = readManifoldSurfaceMesh((DATA_PATH / "armadillo_cut_low.obj").string());
    VertexData<Vector2> param = tutte_embedding(*mesh, *geometry);

    // Add mesh and initial param to viewer
    auto ps_mesh = polyscope::registerSurfaceMesh("Mesh", geometry->inputVertexPositions, mesh->getFaceVertexList());
    auto ps_param_init = polyscope::registerSurfaceMesh2D("Param Initial", param, mesh->getFaceVertexList());
    ps_mesh->setEnabled(false);
    ps_param_init->setEdgeWidth(1.0);
    ps_param_init->setEnabled(false);

    // Pre-compute triangle rest shapes in local coordinate systems
    FaceData<Eigen::Matrix2d> rest_shapes(*mesh);
    for (auto f : mesh->faces())
    {
        // Get 3D vertex positions
        Eigen::Vector3d ar_3d = to_eigen(geometry->vertexPositions[f.halfedge().vertex()]);
        Eigen::Vector3d br_3d = to_eigen(geometry->vertexPositions[f.halfedge().next().vertex()]);
        Eigen::Vector3d cr_3d = to_eigen(geometry->vertexPositions[f.halfedge().next().next().vertex()]);

        // Set up local 2D coordinate system
        Eigen::Vector3d n = (br_3d - ar_3d).cross(cr_3d - ar_3d);
        Eigen::Vector3d b1 = (br_3d - ar_3d).normalized();
        Eigen::Vector3d b2 = n.cross(b1).normalized();

        // Express a, b, c in local 2D coordiante system
        Eigen::Vector2d ar_2d(0.0, 0.0);
        Eigen::Vector2d br_2d((br_3d - ar_3d).dot(b1), 0.0);
        Eigen::Vector2d cr_2d((cr_3d - ar_3d).dot(b1), (cr_3d - ar_3d).dot(b2));

        // Save 2-by-2 matrix with edge vectors as colums
        rest_shapes[f] = TinyAD::col_mat(br_2d - ar_2d, cr_2d - ar_2d);
    };

    // Set up function with 2D vertex positions as variables.
    auto func = TinyAD::scalar_function<2>(mesh->vertices());

    // Add objective term per face. Each connecting 3 vertices.
    func.add_elements<3>(mesh->faces(), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        // Evaluate element using either double or TinyAD::Double
        using T = TINYAD_SCALAR_TYPE(element);

        // Get variable 2D vertex positions
        Face f = element.handle;
        Eigen::Vector2<T> a = element.variables(f.halfedge().vertex());
        Eigen::Vector2<T> b = element.variables(f.halfedge().next().vertex());
        Eigen::Vector2<T> c = element.variables(f.halfedge().next().next().vertex());

        // Triangle flipped?
        Eigen::Matrix2<T> M = TinyAD::col_mat(b - a, c - a);
        if (M.determinant() <= 0.0)
            return (T)INFINITY;

        // Get constant 2D rest shape of f
        Eigen::Matrix2d Mr = rest_shapes[f];
        double A = 0.5 * Mr.determinant();

        // Compute symmetric Dirichlet energy
        Eigen::Matrix2<T> J = M * Mr.inverse();
        return A * (J.squaredNorm() + J.inverse().squaredNorm());
    });

    // Assemble inital x vector from parametrization property.
    // x_from_data(...) takes a lambda function that maps
    // each variable handle to its initial 2D value (Eigen::Vector2d).
    Eigen::VectorXd x = func.x_from_data([&] (Vertex v) {
        return to_eigen(param[v]);
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
    // of each variable (Eigen::Vector2d) back our parametrization property.
    func.x_to_data(x, [&] (Vertex v, const Eigen::Vector2d& p) {
        param[v] = to_geometrycentral(p);
    });

    // Show resulting parametrization
    auto ps_param_result = polyscope::registerSurfaceMesh2D("Param Result", param, mesh->getFaceVertexList());
    ps_param_result->setEdgeWidth(1.0);
    polyscope::show();

    return 0;
}
