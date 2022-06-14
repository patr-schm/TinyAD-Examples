/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Anton Florey
 */
#include <TinyAD/Support/OpenMesh.hh>
#include <TinyAD/ScalarFunction.hh>
#include <TinyAD/Utils/LineSearch.hh>
#include <TinyAD/Utils/NewtonDirection.hh>
#include <TinyAD/Utils/NewtonDecrement.hh>

#include <TinyAD-Examples/Filesystem.hh>
#include <TinyAD-Examples/GlowViewerOpenMesh.hh>

/**
 * Planarity constraint penalty function.
 */
template <typename T>
T angle_cost(
        const Eigen::Vector3<T>& a,
        const Eigen::Vector3<T>& b,
        const Eigen::Vector3<T>& c,
        const Eigen::Vector3<T>& d)
{
    // Compute all four normalized quad edge vectors
    Eigen::Vector3<T> ab = (b - a).normalized();
    Eigen::Vector3<T> bc = (c - b).normalized();
    Eigen::Vector3<T> cd = (d - c).normalized();
    Eigen::Vector3<T> da = (a - d).normalized();

    // Compute cosine of all 4 angles (unsigned)
    T cos_angle_a = (-da).dot(ab);
    T cos_angle_b = (-ab).dot(bc);
    T cos_angle_c = (-bc).dot(cd);
    T cos_angle_d = (-cd).dot(da);

    // Check if all values are in the interval [-1, 1]
    if(abs(cos_angle_a) > 1 || abs(cos_angle_b) > 1 || abs(cos_angle_c) > 1 || abs(cos_angle_d) > 1)
        TINYAD_WARNING("cosine out of range!");

    // Return the squared angle sum deviation from 2pi
    T angle_a = acos(cos_angle_a);
    T angle_b = acos(cos_angle_b);
    T angle_c = acos(cos_angle_c);
    T angle_d = acos(cos_angle_d);
    return sqr(angle_a + angle_b + angle_c + angle_d - 2.0 * M_PI);
}

/**
 * Heat map function for mesh faces.
 */
double face_heat(
        const OpenMesh::PolyMesh& mesh,
        const OpenMesh::SmartFaceHandle& f)
{
    Eigen::Vector3d a = mesh.point(f.halfedge().from());
    Eigen::Vector3d b = mesh.point(f.halfedge().to());
    Eigen::Vector3d c = mesh.point(f.halfedge().next().to());
    Eigen::Vector3d d = mesh.point(f.halfedge().prev().from());

    // Logarithmic angle cost
    double angle = std::log(1.0 + std::sqrt(angle_cost<double>(a, b, c, d)));

    // Return a value between 0 and 1
    double max_val = 0.1 * M_PI;
    return std::min(1.0, angle / (std::log(1.0 + max_val)));
}

/**
 * Draw a given mesh with color-coded faces.
 */
auto draw_mesh_colored(
        const OpenMesh::PolyMesh& mesh)
{
    auto style = glow_default_style();
    auto v = gv::view();
    auto c = gv::canvas();

    // Color code the mesh faces
    for(auto f : mesh.faces())
    {
        tg::pos3 tg_a = tg::pos3(mesh.point(f.halfedge().from()));
        tg::pos3 tg_b = tg::pos3(mesh.point(f.halfedge().to()));
        tg::pos3 tg_c = tg::pos3(mesh.point(f.halfedge().next().to()));
        tg::pos3 tg_d = tg::pos3(mesh.point(f.halfedge().prev().from()));
        double t = face_heat(mesh, f);
        c.set_color(tg::mix(WHITE, RED, t));
        c.add_face(tg_a, tg_b, tg_c, tg_d);
    }

    // Draw the mesh edges
    for(auto e : mesh.edges())
    {
        c.set_color(0.2 * WHITE);
        c.set_line_width_px(1.0);
        c.add_line(tg::pos3(mesh.point(e.v0())), tg::pos3(mesh.point(e.v1())));
    }

    return v;
}

/**
 * Scale a given mesh to have a bounding box of volume 1.
 */
void normalize_mesh(
        OpenMesh::PolyMesh& _mesh)
{
    // Scale the meshes bounding box to unit volume
    double maxf = std::numeric_limits<double>::max();
    double minf = std::numeric_limits<double>::min();
    Eigen::Vector3d _min(maxf, maxf, maxf);
    Eigen::Vector3d _max(minf, minf, minf);
    for(auto vh : _mesh.vertices()){
        Eigen::Vector3d curr = _mesh.point(vh);
        if(curr.x() < _min.x()) _min.x() = curr.x();
        if(curr.x() > _max.x()) _max.x() = curr.x();
        if(curr.y() < _min.y()) _min.y() = curr.y();
        if(curr.y() > _max.y()) _max.y() = curr.y();
        if(curr.z() < _min.z()) _min.z() = curr.z();
        if(curr.z() > _max.z()) _max.z() = curr.z();
    }
    // Volume of bounding rect:
    Eigen::Vector3d diff = _max - _min;
    double mesh_volume_scale = std::cbrt(std::abs(diff.x() * diff.y() * diff.z()));
    for(auto vh : _mesh.vertices())
        _mesh.point(vh) /= mesh_volume_scale;
}

/**
 * Optimize 3D vertex positions of a quad mesh for planarity.
 * Implementation of one of the planarity terms (Eq. 1) from
 * Geometric Modeling with Conical Meshes and Developable Surfaces [Liu 2006].
 */
int main()
{
    // Init viewer
    glow::glfw::GlfwContext ctx;

    // Read a mesh
    OpenMesh::PolyMesh mesh;
    OpenMesh::IO::read_mesh(mesh, DATA_PATH / "bunny.obj");

    // Scale the mesh's bounding box to unit volume
    normalize_mesh(mesh);
    OpenMesh::PolyMesh mesh_orig = mesh;

    // Initialize the unconstrained cost function
    // with 3D vertex positions as variables
    auto func = TinyAD::scalar_function<3>(mesh.vertices());

    // Set weights for the closeness and barrier term
    const double closeness_weight = 1.0;
    const double edge_barrier_weight = 1e-1;

    // Add planarity-enforcing terms for each quad face
    func.add_elements<4>(mesh.faces(), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        // Evaluate element using either double or TinyAD::Double
        using T = TINYAD_SCALAR_TYPE(element);
        OpenMesh::SmartFaceHandle f = element.handle;
        Eigen::Vector3<T> a = element.variables(f.halfedge().from());
        Eigen::Vector3<T> b = element.variables(f.halfedge().to());
        Eigen::Vector3<T> c = element.variables(f.halfedge().next().to());
        Eigen::Vector3<T> d = element.variables(f.halfedge().prev().from());

        return angle_cost<T>(a, b, c, d) / mesh.n_faces();
    });

    // Add penalty for large edge deviations (avoid degenerating edges)
    func.add_elements<2>(mesh.edges(), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        // Evaluate element using either double or TinyAD::Double
        using T = TINYAD_SCALAR_TYPE(element);
        OpenMesh::SmartEdgeHandle e = element.handle;
        Eigen::Vector3<T> v0 = element.variables(e.v0());
        Eigen::Vector3<T> v1 = element.variables(e.v1());

        // Compute the current edge length and compare it with the original one
        T edge_length = (v0 - v1).norm();
        const double orig_edge_length = mesh.calc_edge_length(e);

        // Symmetric barrier
        return edge_barrier_weight * (0.5 * (orig_edge_length / edge_length + edge_length / orig_edge_length) - 1.0) / mesh.n_edges();
    });

    // Add penalty terms per mesh vertex
    func.add_elements<1>(mesh.vertices(), [&] (auto& element) -> TINYAD_SCALAR_TYPE(element)
    {
        // Evaluate element using either double or TinyAD::Double
        using T = TINYAD_SCALAR_TYPE(element);

        OpenMesh::SmartVertexHandle v = element.handle;
        Eigen::Vector3<T> p = element.variables(v);
        Eigen::Vector3d p_ref = mesh_orig.point(v);

        return closeness_weight * (p - p_ref).squaredNorm() / mesh_orig.n_vertices();
    });

    // Initialize x with the 3D vertex positions
    Eigen::VectorXd x = func.x_from_data([&] (OpenMesh::SmartVertexHandle v) {
        return mesh.point(v);
    });

    // Projected Newton
    const int max_iters = 200;
    const double convergence_eps = 1e-6;
    TinyAD::LinearSolver solver;
    for (int iter = 0; iter < max_iters; ++iter)
    {
        auto [f, g, H_proj] = func.eval_with_hessian_proj(x);
        Eigen::VectorXd d = TinyAD::newton_direction(g, H_proj, solver);
        double newton_decrement = TinyAD::newton_decrement<double>(d, g);
        TINYAD_DEBUG_OUT("Energy | Newton decrement in iteration " << iter << ": " << f << " | " << newton_decrement);
        if(newton_decrement < convergence_eps)
            break;
        x = TinyAD::line_search(x, d, f, g, func, 1.0, 0.8, 256);
    }
    TINYAD_DEBUG_OUT("Final energy: " << func.eval(x));

    // Write final vertex positions to mesh.
    func.x_to_data(x, [&] (OpenMesh::SmartVertexHandle v, const Eigen::Vector3d& _p) {
        mesh.point(v) = _p;
    });

    // Visualization
    auto g = gv::grid();
    draw_mesh_colored(mesh_orig);
    draw_mesh_colored(mesh);

    return 0;
}
