/*
 * (c) 2021 Patrick Schmidt
 */
#pragma once

#include <Eigen/Core>
#include <TinyAD/Utils/Out.hh>
#include <TinyAD-Examples/OpenMeshTypes.hh>

/**
 * Convert a triangle mesh
 * from igl-style format to OpenMesh.
 *      _V: #V-by-3 vertex positions
 *      _F: #F-by-3 indices into _V
 */
inline OpenMesh::TriMesh to_openmesh(
        const Eigen::MatrixXd& _V,
        const Eigen::MatrixXi& _F)
{
    TINYAD_ASSERT(_V.cols() == 3);
    TINYAD_ASSERT(_F.cols() == 3);

    OpenMesh::TriMesh mesh;

    for (int i = 0; i < _V.rows(); ++i)
        mesh.add_vertex(_V.row(i));

    using VH = OpenMesh::VertexHandle;
    for (int i = 0; i < _F.rows(); ++i)
        mesh.add_face(VH(_F(i, 0)), VH(_F(i, 1)), VH(_F(i, 2)));

    return mesh;
}

/**
 * Convert a triangle mesh from OpenMesh to igl-style.
 *      _V: #V-by-3 vertex positions
 *      _F: #F-by-3 indices into _V
 */
inline void to_igl(
        const OpenMesh::TriMesh& _mesh,
        Eigen::MatrixXd& _V,
        Eigen::MatrixXi& _F)
{
    _V.resize(_mesh.n_vertices(), 3);
    for (const auto v : _mesh.vertices())
        _V.row(v.idx()) = _mesh.point(v);

    _F.resize(_mesh.n_faces(), 3);
    for (const auto f : _mesh.faces())
    {
        _F(f.idx(), 0) = f.halfedge().to().idx();
        _F(f.idx(), 1) = f.halfedge().next().to().idx();
        _F(f.idx(), 2) = f.halfedge().from().idx();
    }
}

/**
 * Convert a triangle mesh parametrization (igl-style) to OpenMesh.
 *      _P: #V-by-2 vertex positions in the plane
 */
inline OpenMesh::VPropHandleT<Eigen::Vector2d> param_to_openmesh(
        const Eigen::MatrixXd& _P,
        OpenMesh::TriMesh& _mesh)
{
    TINYAD_ASSERT_EQ(_P.rows(), _mesh.n_vertices());
    TINYAD_ASSERT_EQ(_P.cols(), 2);

    OpenMesh::VPropHandleT<Eigen::Vector2d> ph_param;
    _mesh.add_property(ph_param);

    for (auto v : _mesh.vertices())
        _mesh.property(ph_param, v) = _P.row(v.idx());

    return ph_param;
}
