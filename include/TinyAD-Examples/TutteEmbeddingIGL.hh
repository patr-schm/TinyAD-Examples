/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <igl/harmonic.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>

/**
 * Compute tutte embedding with boundary on circle.
 * Per-vertex 2D coordinates returned as n_vertices-by-2 matrix.
 */
inline Eigen::MatrixXd tutte_embedding(
        const Eigen::MatrixXd& _V,
        const Eigen::MatrixXi& _F)
{
    Eigen::VectorXi b; // #constr boundary constraint indices
    Eigen::MatrixXd bc; // #constr-by-2 2D boundary constraint positions
    Eigen::MatrixXd P; // #V-by-2 2D vertex positions
    igl::boundary_loop(_F, b); // Identify boundary vertices
    igl::map_vertices_to_circle(_V, b, bc); // Set boundary vertex positions
    igl::harmonic(_F, b, bc, 1, P); // Compute interior vertex positions

    return P;
}
