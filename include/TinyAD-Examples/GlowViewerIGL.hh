/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <Eigen/Core>
#include <igl/readOBJ.h>
#include <TinyAD/Utils/Helpers.hh>
#include <TinyAD-Examples/GlowViewerCommon.hh>
#include <TinyAD-Examples/IGLPolymeshConvert.hh>

/**
 * Show a triangle mesh or tetrahedral mesh (igl-style format) using glow viewer.
 * Requires a glow::glfw::GlfwContext to be in scope.
 * Keep the returned viewer in scope if you wish to add to it.
 * Opens window when the last viewer goes out of scope.
 *      _V: #V-by-3 vertex positions
 *      _F: #F-by-3 or #F-by-4 indices into _V
 */
inline gv::detail::raii_view_closer glow_view_mesh(
        const Eigen::MatrixXd& _V,
        const Eigen::MatrixXi& _F,
        const bool _wireframe = false,
        const std::string& _caption = "",
        const tg::color4& _color = glow_default_wireframe_color,
        const float _line_width_px = glow_default_wireframe_width_px,
        const tg::color4& _mesh_color = glow_default_mesh_color)
{
    // Convert to polymesh
    pm::Mesh m;
    auto pos = to_polymesh(_V, _F, m);

    return glow_view_mesh(pos, _wireframe, _caption, _color, _line_width_px, _mesh_color);
}

/**
 * Show a triangle mesh or tetrahedral mesh (igl-style format) using glow viewer.
 * Requires a glow::glfw::GlfwContext to be in scope.
 * Keep the returned viewer in scope if you wish to add to it.
 * Opens window when the last viewer goes out of scope.
 *      _V: #V-by-3 vertex positions
 *      _b: #constr indices into _V
 *      _bc #constr-by-3 constrained positions
 */
inline gv::detail::raii_view_closer glow_view_point_constraints(
        const Eigen::MatrixXd& _V,
        const Eigen::VectorXi& _b,
        const Eigen::MatrixXd& _bc,
        const float _constraint_point_size_px = glow_default_constraint_point_size_px,
        const float _constraint_line_width_px = glow_default_constraint_line_width_px)
{
    auto style = glow_default_style();
    auto v = gv::view();
    auto c = gv::canvas();

    c.set_color(RED);
    c.set_point_size_px(_constraint_point_size_px);
    c.set_line_width_px(_constraint_line_width_px);
    for (int i = 0; i < _b.rows(); ++i)
    {
        const Eigen::Vector3d p(_V.row(_b[i]));
        const Eigen::Vector3d p_constr(_bc.row(i));
        c.add_point(p);
        c.add_point(p_constr);
        c.add_line(tg::segment(tg::dpos3(p), tg::dpos3(p_constr)));
    }

    return v;
}

/**
 * Show a triangle mesh in the plane (igl-style format) using glow viewer.
 * Requires a glow::glfw::GlfwContext to be in scope.
 * Keep the returned viewer in scope if you wish to add to it.
 * Opens window when the last viewer goes out of scope.
 *      _P: #V-by-2 vertex positions
 *      _F: #F-by-3 indices into _P
 */
inline gv::detail::raii_view_closer glow_view_param(
        const Eigen::MatrixXd& _P,
        const Eigen::MatrixXi& _F,
        const std::string& _caption = "",
        const tg::color4& _color = glow_default_wireframe_color,
        const float _line_width_px = glow_default_wireframe_width_px,
        const tg::color4& _mesh_color = glow_default_mesh_color)
{
    // Convert to polymesh
    pm::Mesh m;
    auto pos_param = to_polymesh(_P, _F, m);

    return glow_view_mesh(pos_param, true, _caption, _color, _line_width_px, _mesh_color);
}

/**
 * Draws the geodesic arc between two points on the unit sphere.
 */
inline void glow_draw_spherical_arc(
        gv::canvas_t& _c,
        const Eigen::Vector3d& _from,
        const Eigen::Vector3d& _to,
        const tg::color4& _color = glow_default_wireframe_color,
        const float _line_width_px = glow_default_wireframe_width_px,
        const int _n_samples_per_180_deg = 150,
        const double _offset = 1e-4)
{
    TINYAD_ASSERT_EPS(_from.norm(), 1.0, 1e-6);
    TINYAD_ASSERT_EPS(_to.norm(), 1.0, 1e-6);
    const double dist = acos(_from.dot(_to));
    const int n_samples = std::max(2, (int)(dist / M_PI * _n_samples_per_180_deg));

    for (int i = 0; i < n_samples - 1; ++i)
    {
        const double lambda1 = (double)i / (double)(n_samples - 1);
        const double lambda2 = (double)(i + 1) / (double)(n_samples - 1);
        Eigen::Vector3d p1 = (1.0 - lambda1) * _from + lambda1 * _to;
        Eigen::Vector3d p2 = (1.0 - lambda2) * _from + lambda2 * _to;
        p1 = p1.normalized() * (1.0 + _offset);
        p2 = p2.normalized() * (1.0 + _offset);

        _c.set_color(_color);
        _c.set_line_width_px(_line_width_px);
        _c.add_line(tg::dpos3(p1), tg::dpos3(p2));
    }
}

/**
 * Show a triangle mesh with vertex positions on a sphere using glow viewer.
 * Requires a glow::glfw::GlfwContext to be in scope.
 * Keep the returned viewer in scope if you wish to add to it.
 * Opens window when the last viewer goes out of scope.
 *      _V: #V-by-3 vertex positions on sphere
 *      _F: #F-by-3 indices into _V
 */
inline gv::detail::raii_view_closer glow_view_sphere_embedding(
        const Eigen::MatrixXd& _V,
        const Eigen::MatrixXi& _F,
        const tg::color4& _color = glow_default_wireframe_color,
        const float _line_width_px = glow_default_wireframe_width_px,
        const int _n_samples_per_180_deg = 150,
        const double _offset = 1e-4,
        const tg::color4& _mesh_color = glow_default_mesh_color)
{
    auto v = gv::view();
    auto bb = gv::config(tg::aabb3(-1, 1));

    { // Show sphere
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        igl::readOBJ((DATA_PATH / "sphere.obj").string(), V, F);
        glow_view_mesh(V, F, false, "", glow_default_wireframe_color, glow_default_wireframe_width_px, _mesh_color);
    }

    auto canvas = gv::canvas();
    for (int i = 0; i < _F.rows(); ++i)
    {
        const Eigen::Vector3d a = _V.row(_F(i, 0));
        const Eigen::Vector3d b = _V.row(_F(i, 1));
        const Eigen::Vector3d c = _V.row(_F(i, 2));

        // Flipped?
        tg::color4 color = _color;
        float width = _line_width_px;
        if (TinyAD::col_mat(a, b, c).determinant() <= 0.0)
        {
            color = RED;
            width *= 1.5;
        }

        glow_draw_spherical_arc(canvas, a, b, color, width, _n_samples_per_180_deg, _offset);
        glow_draw_spherical_arc(canvas, b, c, color, width, _n_samples_per_180_deg, _offset);
        glow_draw_spherical_arc(canvas, c, a, color, width, _n_samples_per_180_deg, _offset);
    }

    return v;
}
