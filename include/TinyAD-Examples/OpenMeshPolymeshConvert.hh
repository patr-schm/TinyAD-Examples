/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <polymesh/Mesh.hh>
#include <typed-geometry/tg-lean.hh>
#include <TinyAD-Examples/OpenMeshTypes.hh>

/**
 * Convert a triangle mesh from OpenMesh to polymesh.
 * Optionally uses vertex positions from property _ph_pos.
 */
inline pm::vertex_attribute<Eigen::Vector3d> to_polymesh(
        const OpenMesh::TriMesh& _mesh,
        pm::Mesh& _m,
        const OpenMesh::VPropHandleT<Eigen::Vector2d>& _ph_pos = OpenMesh::VPropHandleT<Eigen::Vector2d>())
{
    auto pos_3d = [&] (const OpenMesh::VertexHandle _v)
    {
        if (_ph_pos.is_valid())
            return Eigen::Vector3d(_mesh.property(_ph_pos, _v)[0], _mesh.property(_ph_pos, _v)[1], 0.0);
        else
            return Eigen::Vector3d(_mesh.point(_v));
    };

    for (auto v : _mesh.vertices())
        _m.vertices().add();

    for (auto f : _mesh.faces())
    {
        _m.faces().add(
                    _m.vertices()[f.halfedge().next().to().idx()],
                    _m.vertices()[f.halfedge().from().idx()],
                    _m.vertices()[f.halfedge().to().idx()]);
    }

    auto pos = _m.vertices().make_attribute<Eigen::Vector3d>();
    for (auto v : _m.vertices())
        pos[v] = pos_3d(OpenMesh::VertexHandle(v.idx.value));

    return pos;
}

/**
 * Convert a polygonal mesh from OpenMesh to polymesh.
 * Optionally uses vertex positions from property _ph_pos.
 */
inline pm::vertex_attribute<Eigen::Vector3d> to_polymesh(
        const OpenMesh::PolyMesh& _mesh,
        pm::Mesh& _m,
        const OpenMesh::VPropHandleT<Eigen::Vector2d>& _ph_pos = OpenMesh::VPropHandleT<Eigen::Vector2d>())
{
    auto pos_3d = [&] (const OpenMesh::VertexHandle _v)
    {
        if (_ph_pos.is_valid())
            return Eigen::Vector3d(_mesh.property(_ph_pos, _v)[0], _mesh.property(_ph_pos, _v)[1], 0.0);
        else
            return Eigen::Vector3d(_mesh.point(_v));
    };

    for (auto v : _mesh.vertices())
        _m.vertices().add();

    for (auto f : _mesh.faces())
    {
        std::vector<pm::vertex_handle> vhs;
        for (auto v : f.vertices())
        {
            vhs.push_back(_m.vertices()[v.idx()]);
        }
        _m.faces().add(vhs);
    }

    auto pos = _m.vertices().make_attribute<Eigen::Vector3d>();
    for (auto v : _m.vertices())
        pos[v] = pos_3d(OpenMesh::VertexHandle(v.idx.value));

    return pos;
}
