/*
 * (c) 2021 Patrick Schmidt
 */
#pragma once

#include <Eigen/Core>
#include <polymesh/Mesh.hh>
#include <TinyAD/Utils/Out.hh>
#include <typed-geometry/tg-lean.hh>

/**
 * Convert a triangle or tetrahedral mesh
 * from igl-style format to polymesh.
 *      _V: #V-by-2 or #V-by-3 vertex positions
 *      _F: #F-by-3 or #F-by-4 indices into _V
 */
inline pm::vertex_attribute<Eigen::Vector3d> to_polymesh(
        const Eigen::MatrixXd& _V,
        const Eigen::MatrixXi& _F,
        pm::Mesh& _m)
{
    TINYAD_ASSERT(_V.cols() == 2 || _V.cols() == 3);
    TINYAD_ASSERT(_F.cols() == 3 || _F.cols() == 4);

    auto pos_3d = [&_V] (const int _v_idx)
    {
        return _V.cols() == 2 ?
                    Eigen::Vector3d(_V(_v_idx, 0), _V(_v_idx, 1), 0.0)
                  : Eigen::Vector3d(_V.row(_v_idx));
    };

    _m.clear();
    auto pos = _m.vertices().make_attribute<Eigen::Vector3d>();

    if (_F.cols() == 3)
    {
        // Triangle mesh
        for (int i = 0; i < _V.rows(); ++i)
        {
            _m.vertices().add();
            pos[_m.vertices()[i]] = pos_3d(i);
        }

        for (int i = 0; i < _F.rows(); ++i)
            _m.faces().add(_m.vertices()[_F(i, 0)], _m.vertices()[_F(i, 1)], _m.vertices()[_F(i, 2)]);
    }
    else if (_F.cols() == 4)
    {
        // Tet mesh: Convert to triangle soup
        for (int i = 0; i < _F.rows(); ++i)
        {
            auto v0 = _m.vertices().add();
            auto v1 = _m.vertices().add();
            auto v2 = _m.vertices().add();
            auto v3 = _m.vertices().add();
            pos[v0] = pos_3d(_F(i, 0));
            pos[v1] = pos_3d(_F(i, 1));
            pos[v2] = pos_3d(_F(i, 2));
            pos[v3] = pos_3d(_F(i, 3));

            _m.faces().add(v0, v2, v1);
            _m.faces().add(v0, v1, v3);
            _m.faces().add(v1, v2, v3);
            _m.faces().add(v2, v0, v3);
        }
    }
    else
        TINYAD_ERROR_throw("");

    return pos;
}

/**
 * Convert a triangle mesh from polymesh to igl-style.
 *      _V: #V-by-3 vertex positions
 *      _F: #F-by-3 indices into _V
 */
inline void to_igl(
        const pm::vertex_attribute<Eigen::Vector3d>& _pos,
        Eigen::MatrixXd& _V,
        Eigen::MatrixXi& _F)
{
    _V.resize(_pos.mesh().vertices().count(), 3);
    for (const auto v : _pos.mesh().vertices())
        _V.row(v.idx.value) = _pos[v];

    _F.resize(_pos.mesh().faces().count(), 3);
    for (const auto f : _pos.mesh().faces())
    {
        _F(f.idx.value, 0) = f.any_halfedge().vertex_to().idx.value;
        _F(f.idx.value, 1) = f.any_halfedge().next().vertex_to().idx.value;
        _F(f.idx.value, 2) = f.any_halfedge().vertex_from().idx.value;
    }
}

/**
 * Convert a triangle mesh parametrization (igl-style) to polymesh.
 *      _P: #V-by-2 vertex positions in the plane
 */
inline pm::vertex_attribute<Eigen::Vector2d> param_to_polymesh(
        const Eigen::MatrixXd& _P,
        const pm::Mesh& _m)
{
    TINYAD_ASSERT_EQ(_P.rows(), _m.vertices().size());
    TINYAD_ASSERT_EQ(_P.cols(), 2);

    auto param = _m.vertices().make_attribute<Eigen::Vector2d>();
    for (auto v : _m.vertices())
        param[v] = Eigen::Vector2d(_P.row(v.idx.value));

    return param;
}
