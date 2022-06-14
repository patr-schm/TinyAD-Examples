/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <Eigen/Core>
#include <TinyAD/Utils/Out.hh>
#include <geometrycentral/surface/manifold_surface_mesh.h>
#include <geometrycentral/surface/vertex_position_geometry.h>

Eigen::Vector2d to_eigen(
        const geometrycentral::Vector2& _v)
{
    return Eigen::Vector2d(_v.x, _v.y);
}

Eigen::Vector3d to_eigen(
        const geometrycentral::Vector3& _v)
{
    return Eigen::Vector3d(_v.x, _v.y, _v.z);
}

geometrycentral::Vector2 to_geometrycentral(
        const Eigen::Vector2d& _v)
{
    return geometrycentral::Vector2 { _v.x(), _v.y() };
}

/**
 * Convert a triangle mesh from Geometry Central to igl-style.
 *      _V: #V-by-3 vertex positions
 *      _F: #F-by-3 indices into _V
 */
inline void to_igl(
        geometrycentral::surface::ManifoldSurfaceMesh& _mesh,
        geometrycentral::surface::VertexPositionGeometry& _geometry,
        Eigen::MatrixXd& _V,
        Eigen::MatrixXi& _F)
{
    TINYAD_ASSERT(_mesh.isCompressed());

    _V.resize(_mesh.nVertices(), 3);
    _geometry.requireVertexPositions();
    for (const auto v : _mesh.vertices())
        _V.row(v.getIndex()) = to_eigen(_geometry.vertexPositions[v]);

    _F.resize(_mesh.nFaces(), 3);
    for (const auto f : _mesh.faces())
    {
        _F(f.getIndex(), 0) = f.halfedge().vertex().getIndex();
        _F(f.getIndex(), 1) = f.halfedge().next().vertex().getIndex();
        _F(f.getIndex(), 2) = f.halfedge().next().next().vertex().getIndex();
    }
}

/**
 * Convert a triangle mesh parametrization (igl-style) to Geometry Central.
 *      _P: #V-by-2 vertex positions in the plane
 */
inline geometrycentral::surface::VertexData<geometrycentral::Vector2> param_to_geometrycentral(
        const Eigen::MatrixXd& _P,
        geometrycentral::surface::ManifoldSurfaceMesh& _mesh,
        geometrycentral::surface::VertexPositionGeometry& _geometry)
{
    TINYAD_ASSERT(_mesh.isCompressed());

    geometrycentral::surface::VertexData<geometrycentral::Vector2> param(_mesh);
    for (auto v : _mesh.vertices())
        param[v] = geometrycentral::Vector2({ _P(v.getIndex(), 0), _P(v.getIndex(), 1) });

    return param;
}
