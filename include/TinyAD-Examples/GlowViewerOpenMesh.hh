/*
 * (c) 2021 Patrick Schmidt
 */
#pragma once

#include <TinyAD-Examples/OpenMeshTypes.hh>
#include <TinyAD-Examples/GlowViewerCommon.hh>
#include <TinyAD-Examples/OpenMeshPolymeshConvert.hh>

/**
 * Show an OpenMesh triangle mesh using glow viewer.
 * Requires a glow::glfw::GlfwContext to be in scope.
 * Keep the returned viewer in scope if you wish to add to it.
 * Opens window when the last viewer goes out of scope.
 */
inline gv::detail::raii_view_closer glow_view_mesh(
        const OpenMesh::TriMesh& _mesh,
        const bool _wireframe = false,
        const std::string& _caption = "",
        const tg::color4& _color = glow_default_wireframe_color,
        const float _line_width_px = glow_default_wireframe_width_px,
        const tg::color4& _mesh_color = glow_default_mesh_color)
{
    // Convert to polymesh
    pm::Mesh m;
    auto pos = to_polymesh(_mesh, m);

    return glow_view_mesh(pos, _wireframe, _caption, _color, _line_width_px, _mesh_color);
}

/**
 * Show an OpenMesh polygonal mesh using glow viewer.
 * Requires a glow::glfw::GlfwContext to be in scope.
 * Keep the returned viewer in scope if you wish to add to it.
 * Opens window when the last viewer goes out of scope.
 */
inline gv::detail::raii_view_closer glow_view_mesh(
        const OpenMesh::PolyMesh& _mesh,
        const bool _wireframe = false,
        const std::string& _caption = "",
        const tg::color4& _color = glow_default_wireframe_color,
        const float _line_width_px = glow_default_wireframe_width_px,
        const tg::color4& _mesh_color = glow_default_mesh_color)
{
    // Convert to polymesh
    pm::Mesh m;
    auto pos = to_polymesh(_mesh, m);

    return glow_view_mesh(pos, _wireframe, _caption, _color, _line_width_px, _mesh_color);
}

/**
 * Show an OpenMesh triangle mesh in the plane using glow viewer.
 * Requires a glow::glfw::GlfwContext to be in scope.
 * Keep the returned viewer in scope if you wish to add to it.
 * Opens window when the last viewer goes out of scope.
 */
inline gv::detail::raii_view_closer glow_view_param(
        const OpenMesh::TriMesh& _mesh,
        const OpenMesh::VPropHandleT<Eigen::Vector2d>& _ph_param,
        const std::string& _caption = "",
        const tg::color4& _color = glow_default_wireframe_color,
        const float _line_width_px = glow_default_wireframe_width_px,
        const tg::color4& _mesh_color = glow_default_mesh_color)
{
    // Convert to polymesh
    pm::Mesh m;
    auto pos_param = to_polymesh(_mesh, m, _ph_param);

    return glow_view_mesh(pos_param, true, _caption, _color, _line_width_px, _mesh_color);
}
