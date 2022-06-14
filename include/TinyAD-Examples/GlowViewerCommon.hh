/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <TinyAD/Utils/Out.hh>
#include <TinyAD-Examples/Filesystem.hh>

#include <Eigen/Core>
#include <typed-geometry/tg-lean.hh>
#include <glow-extras/viewer/view.hh>
#include <glow-extras/viewer/canvas.hh>
#include <glow-extras/glfw/GlfwContext.hh>

/// Colors
static tg::color4 WHITE        (255.0f / 255.0f, 255.0f / 255.0f, 255.0f / 255.0f, 1.0f);
static tg::color4 BLACK        (  0.0f / 255.0f,   0.0f / 255.0f,   0.0f / 255.0f, 1.0f);
static tg::color4 BLUE         (  0.0f / 255.0f,  84.0f / 255.0f, 159.0f / 255.0f, 1.0f);
static tg::color4 MAGENTA      (227.0f / 255.0f,   0.0f / 255.0f, 102.0f / 255.0f, 1.0f);
static tg::color4 YELLOW       (255.0f / 255.0f, 237.0f / 255.0f,   0.0f / 255.0f, 1.0f);
static tg::color4 PETROL       (  0.0f / 255.0f,  97.0f / 255.0f, 101.0f / 255.0f, 1.0f);
static tg::color4 TEAL         (  0.0f / 255.0f, 152.0f / 255.0f, 161.0f / 255.0f, 1.0f);
static tg::color4 GREEN        ( 87.0f / 255.0f, 171.0f / 255.0f,  39.0f / 255.0f, 1.0f);
static tg::color4 MAY_GREEN    (189.0f / 255.0f, 205.0f / 255.0f,   0.0f / 255.0f, 1.0f);
static tg::color4 ORANGE       (246.0f / 255.0f, 168.0f / 255.0f,   0.0f / 255.0f, 1.0f);
static tg::color4 RED          (204.0f / 255.0f,   7.0f / 255.0f,  30.0f / 255.0f, 1.0f);
static tg::color4 BORDEAUX     (161.0f / 255.0f,  16.0f / 255.0f,  53.0f / 255.0f, 1.0f);
static tg::color4 PURPLE       ( 97.0f / 255.0f,  33.0f / 255.0f,  88.0f / 255.0f, 1.0f);
static tg::color4 LILAC        (122.0f / 255.0f, 111.0f / 255.0f, 172.0f / 255.0f, 1.0f);

static tg::color4 glow_default_wireframe_color = BLUE;
static tg::color4 glow_default_constraint_color = RED;
static tg::color4 glow_default_mesh_color = WHITE;
static float glow_default_wireframe_width_px = 0.5;
static float glow_default_constraint_point_size_px =  4.0;
static float glow_default_constraint_line_width_px = 1.0;

static tg::ivec2 glow_default_screenshot_resolution = tg::ivec2(1920, 1080);

/**
 * Return glow viewer config object which is active while in scope.
 * Sets background to white, removes grid, etc...
 */
inline gv::detail::raii_config glow_default_style()
{
    return gv::config(
        gv::background_color(tg::color3::white),
        gv::no_grid,
        gv::no_outline,
        gv::ssao_power(0.5f),
        gv::shadow_strength(0.5f),
        gv::sun_scale_factor(3.0f),
        gv::shadow_world_fadeout_factor(0.3f, 0.6f)
    );
}

/**
 * Return glow viewer config object which is active while in scope.
 * View calls will write a screenshot instead of opening a window.
 */
inline gv::detail::raii_config glow_screenshot_config(
        const fs::path& _file_path,
        const tg::ivec2& _resolution = glow_default_screenshot_resolution,
        const bool _transparent = true)
{
    // Create screenshot folder if not existing
    fs::create_directories(fs::path(_file_path).parent_path());

    return gv::config(gv::headless_screenshot(
                _resolution,
                64,
                _file_path.string(),
                _transparent ? GL_RGBA8 : GL_RGB8));
}

/**
 * Return glow viewer config object which is active while in scope.
 * View calls will write a screenshot instead of opening a window.
 */
inline gv::detail::raii_config glow_screenshot_config(
        const fs::path& _file_path,
        const glow::viewer::camera_transform& _cam_pos,
        const tg::ivec2& _resolution = glow_default_screenshot_resolution,
        const bool _transparent = true)
{
    // Create screenshot folder if not existing
    fs::create_directories(fs::path(_file_path).parent_path());

    return gv::config(
                _cam_pos,
                gv::headless_screenshot(
                            _resolution,
                            64,
                            _file_path.string(),
                            _transparent ? GL_RGBA8 : GL_RGB8));
}

/**
 * Show a triangle mesh or tetrahedral mesh (Polymesh datatype) using glow viewer.
 * Requires a glow::glfw::GlfwContext to be in scope.
 * Keep the returned viewer in scope if you wish to add to it.
 * Opens window when the last viewer goes out of scope.
 */
inline gv::detail::raii_view_closer glow_view_mesh(
        const pm::vertex_attribute<Eigen::Vector3d>& _pos,
        const bool _wireframe = false,
        const std::string& _caption = "",
        const tg::color4& _line_color = glow_default_wireframe_color,
        const float _line_width_px = glow_default_wireframe_width_px,
        const tg::color4& _mesh_color = glow_default_mesh_color)
{
    auto style = glow_default_style();
    auto v = gv::view(_pos, _caption, _mesh_color);

    if (_wireframe)
    {
        auto c = gv::canvas();
        for (auto e : _pos.mesh().edges())
        {
            c.set_line_width_px(_line_width_px);
            c.add_line(tg::segment(tg::dpos3(_pos[e.vertexA()]), tg::dpos3(_pos[e.vertexB()])), _line_color);
        }
    }

    return v;
}

/**
 * Show a triangle mesh in the plane (Polymesh datatype) using glow viewer.
 * Requires a glow::glfw::GlfwContext to be in scope.
 * Keep the returned viewer in scope if you wish to add to it.
 * Opens window when the last viewer goes out of scope.
 */
inline gv::detail::raii_view_closer glow_view_param(
        const pm::vertex_attribute<Eigen::Vector2d>& _param,
        const std::string& _caption = "",
        const tg::color4& _color = glow_default_wireframe_color,
        const float _line_width_px = glow_default_wireframe_width_px)
{
    auto style = glow_default_style();
    auto v = gv::view();

    // Draw mesh
    auto pos_param = _param.mesh().vertices().make_attribute<Eigen::Vector3d>();
    for (auto v : _param.mesh().vertices())
        pos_param[v] = Eigen::Vector3d(_param[v].x(), _param[v].y(), 0.0);
    gv::view(pos_param, _caption);

    // Draw wireframe
    auto c = gv::canvas();
    for (auto e : _param.mesh().edges())
    {
        c.set_line_width_px(_line_width_px);
        c.add_line(tg::segment(tg::dpos3(pos_param[e.vertexA()]), tg::dpos3(pos_param[e.vertexB()])), _color);
    }

    return v;
}
