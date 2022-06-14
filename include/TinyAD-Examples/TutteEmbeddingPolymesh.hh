/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <igl/harmonic.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <TinyAD-Examples/IGLPolymeshConvert.hh>

/**
 * Compute tutte embedding with boundary on circle.
 * Per-vertex 2D coordinates are returned as polymesh attribute.
 */
pm::vertex_attribute<Eigen::Vector2d> tutte_embedding(
        const pm::vertex_attribute<Eigen::Vector3d>& _pos)
{
    Eigen::MatrixXd V; // 3D vertex positions
    Eigen::MatrixXi F; // Mesh faces
    Eigen::VectorXi b; // Boundary constraint indices
    Eigen::MatrixXd bc; // Boundary constraint 2D positions
    Eigen::MatrixXd P; // 2D parametrization positions
    to_igl(_pos, V, F); // Convert mesh to igl format
    igl::boundary_loop(F, b); // Identify boundary vertices
    igl::map_vertices_to_circle(V, b, bc); // Set boundary vertex positions
    igl::harmonic(F, b, bc, 1, P); // Compute interior vertex positions

    return param_to_polymesh(P, _pos.mesh());
}
