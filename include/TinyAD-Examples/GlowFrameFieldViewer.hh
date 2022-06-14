/*
 * This file is part of TinyAD and released under the MIT license.
 * Author: Patrick Schmidt
 */
#pragma once

#include <Eigen/Core>
#include <TinyAD/Utils/ToPassive.hh>
#include <TinyAD-Examples/GlowViewerIGL.hh>
#include <TinyAD-Examples/IGLOpenMeshConvert.hh>
#include <TinyAD-Examples/Extern/ExactPredicates.h>

enum class StreamlineFade
{
    Directed,
    Symmetric,
};

enum class StreamlineColors
{
    Black,
    RandomGray,
    RandomColorful,
};


/**
 * A point on a surface expressed via barycentric coordinates in a triangle.
 * The first halfedge always points to vertex a.
 */
struct BarycentricPoint
{

BarycentricPoint() :
    heh_to_a_(-1), alpha_(NAN), beta_(NAN), gamma_(NAN) { }

/// Applies permutation such that heh_to_a is first halfedge
BarycentricPoint(
        const OpenMesh::HalfedgeHandle _heh_to_a,
        const double _alpha,
        const double _beta,
        const OpenMesh::TriMesh& _mesh)
{
    const OpenMesh::HalfedgeHandle heh_first = _mesh.halfedge_handle(_mesh.face_handle(_heh_to_a));
    if (_heh_to_a == heh_first)
    {
        alpha_ = _alpha;
        beta_ = _beta;
    }
    else if (_heh_to_a == _mesh.next_halfedge_handle(heh_first))
    {
        alpha_ = 1.0 - _alpha - _beta;
        beta_ = _alpha;
    }
    else
    {
        alpha_ = _beta;
        beta_ = 1.0 - _alpha - _beta;
    }

    heh_to_a_ = heh_first;
    gamma_ = 1.0 - alpha_ - beta_;
}

/// Applies permutation such that heh_to_a is first halfedge
BarycentricPoint(
        const OpenMesh::HalfedgeHandle _heh_to_a,
        const double _alpha,
        const double _beta,
        const double _gamma,
        const OpenMesh::TriMesh& _mesh)
{
    const OpenMesh::HalfedgeHandle heh_first = _mesh.halfedge_handle(_mesh.face_handle(_heh_to_a));
    if (_heh_to_a == heh_first)
    {
        alpha_ = _alpha;
        beta_ = _beta;
        gamma_ = _gamma;
    }
    else if (_heh_to_a == _mesh.next_halfedge_handle(heh_first))
    {
        alpha_ = _gamma;
        beta_ = _alpha;
        gamma_ = _beta;
    }
    else
    {
        alpha_ = _beta;
        beta_ = _gamma;
        gamma_ = _alpha;
    }

    heh_to_a_ = heh_first;
}

BarycentricPoint(
        const OpenMesh::FaceHandle _fh,
        const double _alpha,
        const double _beta,
        const OpenMesh::TriMesh& _mesh) :
    heh_to_a_(_mesh.halfedge_handle(_fh)),
    alpha_(_alpha),
    beta_(_beta),
    gamma_(1.0 - _alpha - _beta)
{ }

BarycentricPoint(
        const OpenMesh::FaceHandle _fh,
        const double _alpha,
        const double _beta,
        const double _gamma,
        const OpenMesh::TriMesh& _mesh) :
    heh_to_a_(_mesh.halfedge_handle(_fh)),
    alpha_(_alpha),
    beta_(_beta),
    gamma_(_gamma)
{ }

BarycentricPoint(
        const Eigen::Vector2d _p,
        const Eigen::Vector2d _a,
        const Eigen::Vector2d _b,
        const Eigen::Vector2d _c,
        const OpenMesh::HalfedgeHandle _heh,
        const OpenMesh::TriMesh& _mesh)
{
    TINYAD_ASSERT(_heh.is_valid());
    heh_to_a_ = _heh;
    compute(_p, _a, _b, _c, heh_to_a_, _mesh, alpha_, beta_, gamma_);

    assert_first_halfedge(_mesh);
}

BarycentricPoint(
        const Eigen::Vector2d _p,
        const Eigen::Vector2d _a,
        const Eigen::Vector2d _b,
        const Eigen::Vector2d _c,
        const OpenMesh::FaceHandle _fh,
        const OpenMesh::TriMesh& _mesh)
{
    TINYAD_ASSERT(_fh.is_valid());
    heh_to_a_ = _mesh.halfedge_handle(_fh);
    compute(_p, _a, _b, _c, heh_to_a_, _mesh, alpha_, beta_, gamma_);
}

BarycentricPoint(
        const Eigen::Vector3d _p,
        const Eigen::Vector3d _a,
        const Eigen::Vector3d _b,
        const Eigen::Vector3d _c,
        const OpenMesh::HalfedgeHandle _heh,
        const OpenMesh::TriMesh& _mesh)
{
    TINYAD_ASSERT(_heh.is_valid());
    heh_to_a_ = _heh;
    compute(_p, _a, _b, _c, heh_to_a_, _mesh, alpha_, beta_, gamma_);

    assert_first_halfedge(_mesh);
}

BarycentricPoint(
        const Eigen::Vector3d _p,
        const OpenMesh::FaceHandle _fh,
        const OpenMesh::TriMesh& _mesh)
{
    TINYAD_ASSERT(_fh.is_valid());

    heh_to_a_ = _mesh.halfedge_handle(_fh);
    const Eigen::Vector3d a = _mesh.point(vh_a(_mesh));
    const Eigen::Vector3d b = _mesh.point(vh_b(_mesh));
    const Eigen::Vector3d c = _mesh.point(vh_c(_mesh));

    compute(_p, a, b, c, heh_to_a_, _mesh, alpha_, beta_, gamma_);
}

BarycentricPoint(
        const OpenMesh::VertexHandle _vh,
        const OpenMesh::TriMesh& _mesh)
{
    TINYAD_ASSERT(_vh.is_valid());

    const OpenMesh::FaceHandle fh = *_mesh.cvf_begin(_vh);
    heh_to_a_ = _mesh.halfedge_handle(fh);

    if (_vh == _mesh.to_vertex_handle(heh_to_a_))
    {
        alpha_ = 1.0;
        beta_ = 0.0;
        gamma_ = 0.0;
    }
    else if (_vh == _mesh.to_vertex_handle(_mesh.next_halfedge_handle(heh_to_a_)))
    {
        alpha_ = 0.0;
        beta_ = 1.0;
        gamma_ = 0.0;
    }
    else if (_vh == _mesh.from_vertex_handle(heh_to_a_))
    {
        alpha_ = 0.0;
        beta_ = 0.0;
        gamma_ = 1.0;
    }
    else
        TINYAD_ERROR_throw("");
}

static void compute(
        const Eigen::Vector2d& _p,
        const Eigen::Vector2d& _a,
        const Eigen::Vector2d& _b,
        const Eigen::Vector2d& _c,
        const OpenMesh::HalfedgeHandle _heh_to_a,
        const OpenMesh::TriMesh& _mesh,
        double& _alpha,
        double& _beta,
        double& _gamma)
{
    const OpenMesh::HalfedgeHandle heh_first = _mesh.halfedge_handle(_mesh.face_handle(_heh_to_a));
    TINYAD_ASSERT_EQ(_heh_to_a, heh_first);

    // Compute barycentric coordinates of 2d point
    const auto va = _a - _c;
    const auto vb = _b - _c;
    const auto vp = _p - _c;

    const double d00 = va.dot(va);
    const double d01 = va.dot(vb);
    const double d02 = va.dot(vp);
    const double d11 = vb.dot(vb);
    const double d12 = vb.dot(vp);

    const double denom = d00 * d11 - d01 * d01;

    _alpha = (d02 * d11 - d01 * d12) / denom;
    _beta = (d00 * d12 - d01 * d02) / denom;
    _gamma = 1.0 - _alpha - _beta;
}

template <typename T>
static void compute(
        const Eigen::Vector2<T>& _p,
        const Eigen::Vector2<T>& _a,
        const Eigen::Vector2<T>& _b,
        const Eigen::Vector2<T>& _c,
        const OpenMesh::HalfedgeHandle _heh_to_a,
        const OpenMesh::TriMesh& _mesh,
        double& _alpha,
        double& _beta,
        double& _gamma)
{
    const OpenMesh::HalfedgeHandle heh_first = _mesh.halfedge_handle(_mesh.face_handle(_heh_to_a));
    TINYAD_ASSERT_EQ(_heh_to_a, heh_first);

    // Compute barycentric coordinates of 2d point
    const auto va = _a - _c;
    const auto vb = _b - _c;
    const auto vp = _p - _c;

    const double d00 = va.dot(va);
    const double d01 = va.dot(vb);
    const double d02 = va.dot(vp);
    const double d11 = vb.dot(vb);
    const double d12 = vb.dot(vp);

    const double denom = d00 * d11 - d01 * d01;

    _alpha = (d02 * d11 - d01 * d12) / denom;
    _beta = (d00 * d12 - d01 * d02) / denom;
    _gamma = 1.0 - _alpha - _beta;
}

static void compute(
        const Eigen::Vector3d _p,
        const Eigen::Vector3d _a,
        const Eigen::Vector3d _b,
        const Eigen::Vector3d _c,
        const OpenMesh::HalfedgeHandle _heh_to_a,
        const OpenMesh::TriMesh& _mesh,
        double& _alpha,
        double& _beta,
        double& _gamma)
{
    const OpenMesh::HalfedgeHandle heh_first = _mesh.halfedge_handle(_mesh.face_handle(_heh_to_a));
    TINYAD_ASSERT_EQ(_heh_to_a, heh_first);

    // Compute barycentric coordinates of projected point
    const auto va = _a - _c;
    const auto vb = _b - _c;
    const auto vp = _p - _c;

    const double d00 = va.dot(va);
    const double d01 = va.dot(vb);
    const double d02 = va.dot(vp);
    const double d11 = vb.dot(vb);
    const double d12 = vb.dot(vp);

    const double denom = d00 * d11 - d01 * d01;

    _alpha = (d02 * d11 - d01 * d12) / denom;
    _beta = (d00 * d12 - d01 * d02) / denom;
    _gamma = 1.0 - _alpha - _beta;
}

void assert_first_halfedge(
        const OpenMesh::TriMesh& _mesh) const
{
    const OpenMesh::HalfedgeHandle heh_first = _mesh.halfedge_handle(_mesh.face_handle(heh_to_a_));
    TINYAD_ASSERT_EQ(heh_to_a_, heh_first);
}

Eigen::Vector3d point(
        const OpenMesh::TriMesh& _mesh) const
{
    TINYAD_ASSERT(is_valid());

    const Eigen::Vector3d a = _mesh.point(vh_a(_mesh));
    const Eigen::Vector3d b = _mesh.point(vh_b(_mesh));
    const Eigen::Vector3d c = _mesh.point(vh_c(_mesh));

    return interpolate(a, b, c);
}

bool is_valid() const
{
    return heh_to_a_.is_valid();
}

const double& alpha() const
{
    return alpha_;
}

const double& beta() const
{
    return beta_;
}

const double& gamma() const
{
    return gamma_;
}

bool is_inside_inclusive() const
{
    return alpha_ >= 0.0 && beta_ >= 0.0 && gamma_ >= 0.0;
}

bool is_inside_exclusive() const
{
    return alpha_ > 0.0 && beta_ > 0.0 && gamma_ > 0.0;
}

OpenMesh::FaceHandle fh(
        const OpenMesh::TriMesh& _mesh) const
{
    TINYAD_ASSERT(is_valid());
    return _mesh.face_handle(heh_to_a_);
}

void set_fh(
        const OpenMesh::FaceHandle _fh, const OpenMesh::TriMesh& _mesh)
{
    TINYAD_ASSERT(_mesh.is_valid_handle(_fh));
    set_heh(_mesh.halfedge_handle(_fh));
}

const OpenMesh::HalfedgeHandle& heh() const
{
    return heh_to_a_;
}

void set_heh(
        const OpenMesh::HalfedgeHandle _heh)
{
    heh_to_a_ = _heh;
}

void set_alpha_beta(
        const double& _alpha,
        const double& _beta)
{
    alpha_ = _alpha;
    beta_ = _beta;
    gamma_ = 1.0 - _alpha - _beta;
}

OpenMesh::VertexHandle vh_a(
        const OpenMesh::TriMesh& _mesh) const
{
    TINYAD_ASSERT(is_valid());
    return _mesh.to_vertex_handle(heh_to_a_);
}

OpenMesh::VertexHandle vh_b(
        const OpenMesh::TriMesh& _mesh) const
{
    TINYAD_ASSERT(is_valid());
    return _mesh.to_vertex_handle(_mesh.next_halfedge_handle(heh_to_a_));
}

OpenMesh::VertexHandle vh_c(
        const OpenMesh::TriMesh& _mesh) const
{
    TINYAD_ASSERT(is_valid());
    return _mesh.from_vertex_handle(heh_to_a_);
}

OpenMesh::HalfedgeHandle heh_to_a(
        const OpenMesh::TriMesh& _mesh) const
{
    return heh_to_a_;
}

OpenMesh::HalfedgeHandle heh_to_b(
        const OpenMesh::TriMesh& _mesh) const
{
    return _mesh.next_halfedge_handle(heh_to_a_);
}

OpenMesh::HalfedgeHandle heh_to_c(
        const OpenMesh::TriMesh& _mesh) const
{
    return _mesh.prev_halfedge_handle(heh_to_a_);
}

OpenMesh::VertexHandle vh_closest(
        const OpenMesh::TriMesh& _mesh) const
{
    TINYAD_ASSERT(is_valid());

    if (alpha_ >= beta_ && alpha_ >= gamma_)
        return vh_a(_mesh);

    if (beta_ >= alpha_ && beta_ >= gamma_)
        return vh_b(_mesh);

    return vh_c(_mesh);
}

// Returns vertex handle if exactly on vertex.
// Invalid otherwise
OpenMesh::VertexHandle vh_on(
        const OpenMesh::TriMesh& _mesh) const
{
    if (alpha_ == 1.0)
    {
        TINYAD_ASSERT_EQ(beta_, 0.0);
        TINYAD_ASSERT_EQ(gamma_, 0.0);
        return vh_a(_mesh);
    }
    else if (beta_ == 1.0)
    {
        TINYAD_ASSERT_EQ(alpha_, 0.0);
        TINYAD_ASSERT_EQ(gamma_, 0.0);
        return vh_b(_mesh);
    }
    else if (gamma_ == 1.0)
    {
        TINYAD_ASSERT_EQ(alpha_, 0.0);
        TINYAD_ASSERT_EQ(beta_, 0.0);
        return vh_c(_mesh);
    }
    else
        return OpenMesh::VertexHandle(-1);
}

// Returns halfedge handle if exactly on edge.
// Invalid otherwise
// This check excludes vertices!
OpenMesh::HalfedgeHandle heh_on(
        const OpenMesh::TriMesh& _mesh) const
{
    if (alpha_ == 0.0 && beta_ != 1.0 && gamma_ != 1.0)
    {
        TINYAD_ASSERT_G(beta_, 0.0);
        TINYAD_ASSERT_L(beta_, 1.0);
        TINYAD_ASSERT_G(gamma_, 0.0);
        TINYAD_ASSERT_L(gamma_, 1.0);
        return heh_to_c(_mesh);
    }
    else if (beta_ == 0.0 && gamma_ != 1.0 && alpha_ != 1.0)
    {
        TINYAD_ASSERT_G(gamma_, 0.0);
        TINYAD_ASSERT_L(gamma_, 1.0);
        TINYAD_ASSERT_G(alpha_, 0.0);
        TINYAD_ASSERT_L(alpha_, 1.0);
        return heh_to_a(_mesh);
    }
    else if (gamma_ == 0.0 && alpha_ != 1.0 && beta_ != 1.0)
    {
        TINYAD_ASSERT_G(alpha_, 0.0);
        TINYAD_ASSERT_L(alpha_, 1.0);
        TINYAD_ASSERT_G(beta_, 0.0);
        TINYAD_ASSERT_L(beta_, 1.0);
        return heh_to_b(_mesh);
    }
    else
        return OpenMesh::HalfedgeHandle(-1);
}

template <class T>
T interpolate(
        const T& _a,
        const T& _b,
        const T& _c) const
{
    if (alpha_ == 0.0)
    {
        if (beta_ == 0.0)
        {
            TINYAD_ASSERT_EQ(gamma_, 1.0);
            return _c;
        }
        else if (beta_ == 1.0)
        {
            TINYAD_ASSERT_EQ(gamma_, 0.0);
            return _b;
        }
        else
            return beta_ * _b + gamma_ * _c;
    }
    else if (beta_ == 0.0)
    {
        TINYAD_ASSERT_NEQ(alpha_, 0.0); // Already handled
        if (gamma_ == 0.0)
        {
            TINYAD_ASSERT_EQ(alpha_, 1.0);
            return _a;
        }
        else
            return alpha_ * _a + gamma_ * _c;
    }
    else if (gamma_ == 0.0)
    {
        TINYAD_ASSERT_NEQ(alpha_, 0.0); // Already handled
        TINYAD_ASSERT_NEQ(beta_, 0.0); // Already handled
        return alpha_ * _a + beta_ * _b;
    }
    else
        return alpha_ * _a + beta_ * _b + gamma_ * _c;
}

template <class T, class VPropT>
T interpolate(
        const VPropT& _prop,
        const OpenMesh::TriMesh& _mesh) const
{
    TINYAD_ASSERT(is_valid());

    // Get property values at vertices
    const T& a = _prop[vh_a(_mesh)];
    const T& b = _prop[vh_b(_mesh)];
    const T& c = _prop[vh_c(_mesh)];

    return interpolate(a, b, c);
}

private:

// This is always the first halfedge of the face.
OpenMesh::HalfedgeHandle heh_to_a_;

// Store all three barycentric coordinates to faithfully represent
// points on edges and vertices.
double alpha_;
double beta_;
double gamma_;

};

/**
 * A point on an edge expressed by a parameter in [0, 1].
 */
class PointOnEdge
{

public:
    PointOnEdge() :
        heh_(-1), lambda_(0) { }

    PointOnEdge(const OpenMesh::HalfedgeHandle _heh, const double _lambda) :
        heh_(_heh), lambda_(_lambda) { }

    PointOnEdge(const PointOnEdge& _other, const OpenMesh::HalfedgeHandle _heh) :
        heh_(_heh), lambda_(_other.lambda_) { }

    PointOnEdge(const Eigen::Vector2d _p, const OpenMesh::HalfedgeHandle _heh,
            const Eigen::Vector2d _from, const Eigen::Vector2d _to)
    {
        heh_ = _heh;
        lambda_ = compute_lambda(_p, _from, _to);
    }

    PointOnEdge(const Eigen::Vector3d _p, const OpenMesh::HalfedgeHandle _heh,
            const Eigen::Vector3d _from, const Eigen::Vector3d _to)
    {
        heh_ = _heh;
        lambda_ = compute_lambda(_p, _from, _to);
    }

    PointOnEdge(const Eigen::Vector3d _p, const OpenMesh::HalfedgeHandle _heh,
            const OpenMesh::TriMesh& _mesh)
        : PointOnEdge(_p,
                      _heh,
                      _mesh.point(vertex_from(_mesh, _heh)),
                      _mesh.point(vertex_to(_mesh, _heh))) { }

    PointOnEdge(const OpenMesh::VertexHandle _vh, const OpenMesh::HalfedgeHandle _heh, const OpenMesh::TriMesh& _mesh)
    {
        heh_ = _heh;
        if (_vh == vertex_from(_mesh, _heh))
            lambda_ = 0.0;
        else if (_vh == vertex_to(_mesh, _heh))
            lambda_ = 1.0;
        else
            TINYAD_ERROR_throw("Vertex " << _vh << " is not incident to halfedge " << _heh);
    }

    static double compute_lambda(const Eigen::Vector2d& _p, const Eigen::Vector2d& _from, const Eigen::Vector2d& _to)
    {
        return ((_p - _from).dot(_to - _from)) / (_to - _from).squaredNorm();
    }

    static double compute_lambda(const Eigen::Vector3d& _p, const Eigen::Vector3d& _from, const Eigen::Vector3d& _to)
    {
        return ((_p - _from).dot(_to - _from)) / (_to - _from).squaredNorm();
    }

    template <typename T>
    static double compute_lambda(const Eigen::Vector2<T>& _p, const Eigen::Vector2<T>& _from, const Eigen::Vector2<T>& _to)
    {
        return ((_p - _from).dot(_to - _from)) / (_to - _from).squaredNorm();
    }

    BarycentricPoint to_barycentric_point(const OpenMesh::TriMesh& _mesh, const OpenMesh::FaceHandle _fh) const
    {
        TINYAD_ASSERT(_mesh.is_valid_handle(_fh));
        TINYAD_ASSERT(_fh == _mesh.face_handle(heh_) || _fh == _mesh.opposite_face_handle(heh_));

        if (_fh == _mesh.face_handle(heh_))
        {
            BarycentricPoint bary;

            if (lambda_ == 0.0)
                bary = BarycentricPoint(heh_, 0.0, 0.0, 1.0, _mesh);
            else if (lambda_ == 1.0)
                bary = BarycentricPoint(heh_, 1.0, 0.0, 0.0, _mesh);
            else
                bary = BarycentricPoint(heh_, lambda_, 0.0, 1.0 - lambda_, _mesh);

            bary.assert_first_halfedge(_mesh);

            return bary;
        }
        else if (_fh == _mesh.opposite_face_handle(heh_))
        {
            const PointOnEdge opposite = to_opposite(_mesh);
            TINYAD_ASSERT(!_mesh.is_boundary(opposite.heh()));
            return opposite.to_barycentric_point(_mesh, _fh);
        }
        else
        {
            TINYAD_ERROR_throw("");
        }
    }

    BarycentricPoint to_barycentric_point(const OpenMesh::TriMesh& _mesh) const
    {
        if (_mesh.is_boundary(heh_))
        {
            const OpenMesh::FaceHandle fh_opp = _mesh.opposite_face_handle(heh_);
            return to_barycentric_point(_mesh, fh_opp);
        }
        else
        {
            const OpenMesh::FaceHandle fh = _mesh.face_handle(heh_);
            return to_barycentric_point(_mesh, fh);
        }
    }

    PointOnEdge to_opposite(const OpenMesh::TriMesh& _mesh) const
    {
        return PointOnEdge(_mesh.opposite_halfedge_handle(heh_), 1.0 - lambda_);
    }

    PointOnEdge to_halfedge(const OpenMesh::TriMesh& _mesh, const OpenMesh::HalfedgeHandle _heh) const
    {
        if (heh_ == _heh)
            return *this;
        else
        {
            TINYAD_ASSERT_EQ(_heh, _mesh.opposite_halfedge_handle(heh_));
            return to_opposite(_mesh);
        }
    }

    PointOnEdge to_halfedge(const OpenMesh::TriMesh& _mesh, const OpenMesh::FaceHandle _fh) const
    {
        if (_mesh.face_handle(heh_) == _fh)
            return *this;
        else
        {
            const PointOnEdge opp = to_opposite(_mesh);
            TINYAD_ASSERT_EQ(_mesh.face_handle(opp.heh_), _fh);
            return opp;
        }
    }

    void invalidate()
    {
        heh_.invalidate();
    }

    static PointOnEdge invalid()
    {
        return PointOnEdge();
    }

    bool is_valid() const
    {
        return heh_.is_valid();
    }

    bool is_inside() const
    {
        return lambda_ >= 0.0 && lambda_ <= 1.0;
    }

    const OpenMesh::HalfedgeHandle& heh() const
    {
        return heh_;
    }

    OpenMesh::HalfedgeHandle& heh()
    {
        return heh_;
    }

    const double& lambda() const
    {
        return lambda_;
    }

    // Deprecate this one for consistency with BarycentricPoint
    double& lambda()
    {
        return lambda_;
    }

    void set_lambda(const double _lambda)
    {
        lambda_ = _lambda;
    }

    double lambda(const OpenMesh::TriMesh& _mesh, const OpenMesh::HalfedgeHandle _heh) const
    {
        return to_halfedge(_mesh, _heh).lambda();
    }

    static OpenMesh::VertexHandle vertex_from(const OpenMesh::TriMesh& _mesh, const OpenMesh::HalfedgeHandle _heh)
    {
        // ASSERT(_fh.is_valid());

        return _mesh.from_vertex_handle(_heh);
    }

    static OpenMesh::VertexHandle vertex_to(const OpenMesh::TriMesh& _mesh, const OpenMesh::HalfedgeHandle _heh)
    {
        // ASSERT(_fh.is_valid());

        return _mesh.to_vertex_handle(_heh);
    }

    OpenMesh::VertexHandle vertex_from(const OpenMesh::TriMesh& _mesh) const
    {
        return vertex_from(_mesh, heh_);
    }

    OpenMesh::VertexHandle vertex_to(const OpenMesh::TriMesh& _mesh) const
    {
        return vertex_to(_mesh, heh_);
    }

    Eigen::Vector3d point(const OpenMesh::TriMesh& _mesh) const
    {
        assert(is_valid());

        Eigen::Vector3d a = _mesh.point(vertex_from(_mesh));
        Eigen::Vector3d b = _mesh.point(vertex_to(_mesh));

        return interpolate(a, b);
    }

    template <class T>
    T interpolate(const T& _from, const T& _to) const
    {
        return (1.0 - lambda_) * _from + lambda_ * _to;
    }

    template <class T>
    T interpolate(const std::vector<T>& _per_vertex, const OpenMesh::TriMesh& _mesh) const
    {
        const OpenMesh::VertexHandle vh_from = vertex_from(_mesh);
        const OpenMesh::VertexHandle vh_to = vertex_to(_mesh);

        // Get  values at vertices
        const T& a = _per_vertex[vh_from.idx()];
        const T& b = _per_vertex[vh_to.idx()];

        return interpolate(a, b);
    }

    template <class T, class VPropT>
    T interpolate(const VPropT& _prop, const OpenMesh::TriMesh& _mesh) const
    {
        const OpenMesh::VertexHandle vh_from = vertex_from(_mesh);
        const OpenMesh::VertexHandle vh_to = vertex_to(_mesh);

        // Get property values at vertices
        const T& a = _prop[vh_from];
        const T& b = _prop[vh_to];

        return interpolate(a, b);
    }

private:
    // The edge is parametrized with unit speed between 0 and 1
    // in direction of the halfedge.
    OpenMesh::HalfedgeHandle heh_;
    double lambda_;
};

inline void handles(
        const OpenMesh::TriMesh& _mesh, const OpenMesh::HalfedgeHandle _heh,
        OpenMesh::VertexHandle& _out_vh_a, OpenMesh::VertexHandle& _out_vh_b, OpenMesh::VertexHandle& _out_vh_c)
{
    auto heh = _heh;
    _out_vh_a = _mesh.to_vertex_handle(heh);
    heh = _mesh.next_halfedge_handle(heh);
    _out_vh_b = _mesh.to_vertex_handle(heh);
    heh = _mesh.next_halfedge_handle(heh);
    _out_vh_c = _mesh.to_vertex_handle(heh);
}

inline void handles(
        const OpenMesh::TriMesh& _mesh, const OpenMesh::FaceHandle _fh,
        OpenMesh::VertexHandle& _out_vh_a, OpenMesh::VertexHandle& _out_vh_b, OpenMesh::VertexHandle& _out_vh_c)
{
    const auto heh = _mesh.halfedge_handle(_fh);
    handles(_mesh, heh, _out_vh_a, _out_vh_b, _out_vh_c);
}

inline void points(
        const OpenMesh::TriMesh& _mesh, const OpenMesh::FaceHandle _fh,
        Eigen::Vector3d& _out_p_a, Eigen::Vector3d& _out_p_b, Eigen::Vector3d& _out_p_c)
{
    OpenMesh::VertexHandle vh_a, vh_b, vh_c;
    handles(_mesh, _fh, vh_a, vh_b, vh_c);

    _out_p_a = _mesh.point(vh_a);
    _out_p_b = _mesh.point(vh_b);
    _out_p_c = _mesh.point(vh_c);
}

/**
 * Local 2d coordinate system of a triangle.
 * Stores origin and basis vectors.
 * Provides convenience methods for conversion from/to the local CS.
 */
struct LocalCoordinateSystem
{

/**
 * Default constructor necessary to store this in a property.
 */
LocalCoordinateSystem()
{
    valid_ = false;
}

/**
 * Use this constructor for valid initialization.
 */
LocalCoordinateSystem(
        const OpenMesh::FaceHandle _fh,
        const OpenMesh::TriMesh& _mesh)
{
    origin_ = comp_origin(_mesh, _fh);
    comp_basis(_mesh, _fh, b0_, b1_);
    valid_ = true;
}

/**
 * Use this constructor for valid initialization.
 * Specify origin.
 */
LocalCoordinateSystem(
        const OpenMesh::FaceHandle _fh,
        const Eigen::Vector3d _origin,
        const OpenMesh::TriMesh& _mesh)
{
    origin_ = _origin;
    comp_basis(_mesh, _fh, b0_, b1_);
    valid_ = true;
}

/**
 * Use this constructor for valid initialization.
 * a, b, c are triangle vertex positions (ccw).
 */
LocalCoordinateSystem(
        const Eigen::Vector3d _a,
        const Eigen::Vector3d _b,
        const Eigen::Vector3d _c)
{
    origin_ = _a;
    comp_basis(_a, _b, _c, b0_, b1_);
    valid_ = true;
}

static LocalCoordinateSystem barycenter(
        const OpenMesh::FaceHandle _fh,
        const OpenMesh::TriMesh& _mesh)
{
    OpenMesh::VertexHandle vh_a, vh_b, vh_c;
    handles(_mesh, _fh, vh_a, vh_b, vh_c);

    const Eigen::Vector3d barycenter = (_mesh.point(vh_a) + _mesh.point(vh_b) + _mesh.point(vh_c)) / 3.0;
    return LocalCoordinateSystem(_fh, barycenter, _mesh);
}

/**
 * Returns origin of the coordinate system in model space (GCS)
 */
Eigen::Vector3d origin() const
{
    return origin_;
}

/**
 * Returns the origin of the coordinate system, i.e. the vertex the first
 * half-edge is pointing from.
 */
static Eigen::Vector3d comp_origin(
        const OpenMesh::TriMesh& _mesh,
        const OpenMesh::FaceHandle _fh)
{
    const OpenMesh::HalfedgeHandle heh0 = _mesh.halfedge_handle(_fh);
    const OpenMesh::VertexHandle vh = _mesh.from_vertex_handle(heh0);

    return _mesh.point(vh);
}

/**
 * Computes the orthonormal basis vectors of a 2d coordinate system in
 * which b0 corresponds to the direction of the first half-edge and
 * b1 is obtained by a 90° ccw rotation.
 */
static void comp_basis(
        const OpenMesh::TriMesh &_mesh,
        const OpenMesh::FaceHandle _fh,
        Eigen::Vector3d &_out_b0,
        Eigen::Vector3d &_out_b1)
{
    const auto normal = _mesh.calc_face_normal(_fh);
    const auto heh0 = _mesh.halfedge_handle(_fh);

    _out_b0 = _mesh.calc_edge_vector(heh0).normalized();
    _out_b1 = normal.cross(_out_b0);

    assert(fabs(normal.squaredNorm() - 1.0) < 1e-6);
    assert(fabs(_out_b1.squaredNorm() - 1.0) < 1e-6);
}

/**
 * Computes the orthonormal basis vectors of a 2d coordinate system in
 * which b0 corresponds to the direction of b-a and b1 is obtained
 * by a 90° ccw rotation.
 */
static void comp_basis(
        const Eigen::Vector3d _a,
        const Eigen::Vector3d _b,
        const Eigen::Vector3d _c,
        Eigen::Vector3d &_out_b0,
        Eigen::Vector3d &_out_b1)
{
    const auto normal = ((_b - _a).cross(_c - _a)).normalized();

    _out_b0 = (_b - _a).normalized();
    _out_b1 = normal.cross(_out_b0);
}

/**
 * Projects a vector from model space to the local coordinate system.
 */
Eigen::Vector2d vector_to_local(
        const Eigen::Vector3d& _vector) const
{
    return Eigen::Vector2d(_vector.dot(b0_), _vector.dot(b1_));
}

/**
 * Maps a vector from the local coordinate system to model space.
 */
Eigen::Vector3d vector_to_global(
        const Eigen::Vector2d& _vector_local) const
{
    return _vector_local[0] * b0_ + _vector_local[1] * b1_;
}

/**
 * Maps a point from model space to the local coordinate system.
 */
Eigen::Vector2d point_to_local(
        Eigen::Vector3d _point) const
{
    _point -= origin_;

    return Eigen::Vector2d(_point.dot(b0_), _point.dot(b1_));
}

/**
 * Maps 3 points from model space to the local coordinate system.
 */
void pointsToLocal(
        const Eigen::Vector3d _pa,
        const Eigen::Vector3d _pb,
        const Eigen::Vector3d _pc,
        Eigen::Vector2d& _out_a,
        Eigen::Vector2d& _out_b,
        Eigen::Vector2d& _out_c) const
{
    _out_a = point_to_local(_pa);
    _out_b = point_to_local(_pb);
    _out_c = point_to_local(_pc);
}

/**
 * Maps a point from the local coordinate system to model space.
 */
Eigen::Vector3d point_to_global(
        const Eigen::Vector2d &_point_local) const
{
    return origin_ + _point_local[0] * b0_ + _point_local[1] * b1_;
}

template <typename T>
Eigen::Vector3<T> point_to_global(
        const Eigen::Vector2<T>& _point_local) const
{
    return origin_ + _point_local[0] * b0_ + _point_local[1] * b1_;
}

/**
 * Maps the vertices of a triangle from model space to its local coordinate system
 */
static void vertices_to_local(
        OpenMesh::FaceHandle _fh,
        const OpenMesh::TriMesh& _mesh,
        Eigen::Vector2d& _out_a,
        Eigen::Vector2d& _out_b,
        Eigen::Vector2d& _out_c)
{
    OpenMesh::VertexHandle vh_a, vh_b, vh_c;
    handles(_mesh, _fh, vh_a, vh_b, vh_c);

    const auto a_3d = _mesh.point(vh_a);
    const auto b_3d = _mesh.point(vh_b);
    const auto c_3d = _mesh.point(vh_c);

    const LocalCoordinateSystem cs(_fh, _mesh);
    _out_a = cs.point_to_local(a_3d);
    _out_b = cs.point_to_local(b_3d);
    _out_c = cs.point_to_local(c_3d);
}

Eigen::Vector3d origin_;
Eigen::Vector3d b0_;
Eigen::Vector3d b1_;

bool valid_;

};

struct DualPath
{
    DualPath() = default;
    DualPath(
            const OpenMesh::FaceHandle _fh_start,
            const OpenMesh::FaceHandle _fh_end);

//    int length() const;

//    bool is_closed(
//            const OpenMesh::TriMesh& _mesh) const;

//    OpenMesh::SmartFaceHandle face(
//            const int _idx,
//            const OpenMesh::TriMesh& _mesh) const;

//    double embedded_length(
//            const OpenMesh::TriMesh& _mesh) const;

//    DualPath reversed(
//            const OpenMesh::TriMesh& _mesh) const;

//    void assert_start_end_valid(
//            const OpenMesh::TriMesh& _mesh) const;

//    void assert_valid(
//            const OpenMesh::TriMesh& _mesh) const;

    OpenMesh::FaceHandle fh_start = OpenMesh::FaceHandle(-1);
    OpenMesh::FaceHandle fh_end = OpenMesh::FaceHandle(-1);
    std::vector<OpenMesh::HalfedgeHandle> hehs; // Halfedges crossed, incident to previous face.
};

DualPath::DualPath(const OpenMesh::FaceHandle _fh_start, const OpenMesh::FaceHandle _fh_end) :
    fh_start(_fh_start),
    fh_end(_fh_end)
{
}

struct Snake
{
    Snake() { }

    Snake(const BarycentricPoint& _start, const BarycentricPoint& _end)
        : start(_start), end(_end) { }

    DualPath to_dual_path(const OpenMesh::TriMesh& _mesh) const
    {
        DualPath path(start.fh(_mesh), end.fh(_mesh));

        if (intersections.empty())
            return path;

        OpenMesh::FaceHandle fh_curr = path.fh_start;
        path.hehs.reserve(intersections.size());
        for (const PointOnEdge& intersection : intersections)
        {
            const OpenMesh::HalfedgeHandle heh_leave = intersection.to_halfedge(_mesh, fh_curr).heh();
            path.hehs.push_back(heh_leave);

            fh_curr = _mesh.opposite_face_handle(heh_leave);
        }

        TINYAD_ASSERT_EQ(fh_curr, path.fh_end);
        return path;
    }

    double length(const OpenMesh::TriMesh& _mesh) const
    {
        double l = 0;
        Eigen::Vector3d p_prev = start.point(_mesh);
        for (const auto& intersection : intersections)
        {
            const Eigen::Vector3d p_curr = intersection.point(_mesh);
            l += (p_curr - p_prev).norm();
            p_prev = p_curr;
        }

        l += (end.point(_mesh) - p_prev).norm();

        return l;
    }

    BarycentricPoint start;
    BarycentricPoint end;
    std::vector<PointOnEdge> intersections;
};

Eigen::Vector2d
tangent_to_bary_dir(
        const OpenMesh::TriMesh& _mesh,
        const OpenMesh::FaceHandle _fh,
        const Eigen::Vector3d& _tangent)
{
    // Get 3D vertex positions
    auto h = _mesh.halfedge_handle(_fh);
    Eigen::Vector3d a = _mesh.point(_mesh.to_vertex_handle(h));
    Eigen::Vector3d b = _mesh.point(_mesh.to_vertex_handle(_mesh.next_halfedge_handle(h)));
    Eigen::Vector3d c = _mesh.point(_mesh.from_vertex_handle(h));

//    // Set up local 2D coordinate system
//    Eigen::Vector3d n = (b - a).cross(c - a);
//    Eigen::Vector3d b1 = (b - a).normalized();
//    Eigen::Vector3d b2 = n.cross(b1).normalized();

//    // Express a-c, b-c
//    Eigen::Vector2d ca((a-c).dot(b1), (a-c).dot(b2));
//    Eigen::Vector2d cb((b-c).dot(b1), (b-c).dot(b2));

//    Eigen::Matrix2d J; // Maps from barycentric vectors to vectors in the LCS of the face.
//    J.col(0) = ca;
//    J.col(1) = cb;
//    const Eigen::Matrix2d J_inv = J.inverse(); // Maps from vectors in the LCS of the face to barycentric vectors.

    Eigen::Vector3d ca = a - c;
    Eigen::Vector3d cb = b - c;

    return Eigen::Vector2d(ca.dot(_tangent), cb.dot(_tangent));
}

//Eigen::Vector3d
//bary_to_tangent_dir(
//        const OpenMesh::TriMesh& _mesh,
//        const OpenMesh::FaceHandle _fh,
//        const Eigen::Vector2d& _bary_dir)
//{
//    Eigen::Vector3d pa, pb, pc;
//    points(_mesh, _fh, pa, pb, pc);
//    LocalCoordinateSystem lcs(_fh, _mesh);
//    Mat2d J; // Maps from barycentric vectors to vectors in the LCS of the face.
//    J.col(0) = lcs.vector_to_local(pa - pc);
//    J.col(1) = lcs.vector_to_local(pb - pc);

//    Eigen::Vector2d lcs_dir = J * _bary_dir;
//    Eigen::Vector3d tangent_dir = lcs.vector_to_global(lcs_dir);
//    return tangent_dir;
//}

Eigen::Vector2d
bary_dir_at_pos(
        const OpenMesh::TriMesh& _mesh,
        const Eigen::MatrixXd& B1, // #F by 3. First basis vector per face.
        const Eigen::MatrixXd& B2, // #F by 3. Second basis vector per face.
        const Eigen::VectorXd& _x,
        const BarycentricPoint& _pos)
{
    const int f_idx = _pos.fh(_mesh).idx();
    const Eigen::Vector3d tangent_dir = (B1.row(f_idx) * _x[4 * f_idx] + B2.row(f_idx) * _x[4 * f_idx + 1]).normalized();

//    const Eigen::Vector3d tangent_dir = whitney_interpolation(_mesh, _x, _pos);

    return tangent_to_bary_dir(_mesh, _pos.fh(_mesh), tangent_dir);
}

BarycentricPoint
random_barycentric_point(
        const OpenMesh::TriMesh& _mesh)
{
    const OpenMesh::FaceHandle fh = OpenMesh::FaceHandle(rand() % _mesh.n_faces());
    double alpha = static_cast<double>(rand()) / RAND_MAX;
    double beta  = static_cast<double>(rand()) / RAND_MAX;
    if (alpha + beta > 1)
    {
        alpha = 1 - alpha;
        beta = 1 - beta;
    }
    return BarycentricPoint(fh, alpha, beta, _mesh);
}

bool ccw_exact_exclusive_2d(
        const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Eigen::Vector2d& c)
{
    return orient2d(a.data(), b.data(), c.data()) > 0.0;
}

bool in_triangle_exact_inclusive_2d(
        const Eigen::Vector2d& p,
        const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Eigen::Vector2d& c)
{
    return orient2d(p.data(), a.data(), b.data()) >= 0 &&
           orient2d(p.data(), b.data(), c.data()) >= 0 &&
           orient2d(p.data(), c.data(), a.data()) >= 0;
}

bool in_triangle_exact_exclusive_2d(
        const Eigen::Vector2d& p,
        const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Eigen::Vector2d& c)
{
    return orient2d(p.data(), a.data(), b.data()) > 0 &&
           orient2d(p.data(), b.data(), c.data()) > 0 &&
           orient2d(p.data(), c.data(), a.data()) > 0;
}

bool intersects_exact_inclusive_2d(
        const Eigen::Vector2d& a, const Eigen::Vector2d& b,
        const Eigen::Vector2d& c, const Eigen::Vector2d& d)
{
    const auto sign1 = orient2d(a.data(), b.data(), c.data());
    const auto sign2 = orient2d(a.data(), d.data(), b.data());
    const auto sign3 = orient2d(a.data(), d.data(), c.data());
    const auto sign4 = orient2d(b.data(), c.data(), d.data());

    int n_negative = 0;
    int n_positive = 0;
    if (sign1 <= 0) ++n_negative;
    if (sign1 >= 0) ++n_positive;
    if (sign2 <= 0) ++n_negative;
    if (sign2 >= 0) ++n_positive;
    if (sign3 <= 0) ++n_negative;
    if (sign3 >= 0) ++n_positive;
    if (sign4 <= 0) ++n_negative;
    if (sign4 >= 0) ++n_positive;

    if (n_negative == 4 || n_positive == 4)
        return true;

    return false;
}

bool intersects_exact_exclusive_2d(
        const Eigen::Vector2d& a, const Eigen::Vector2d& b,
        const Eigen::Vector2d& c, const Eigen::Vector2d& d)
{
    const auto sign1 = orient2d(a.data(), b.data(), c.data());
    const auto sign2 = orient2d(a.data(), d.data(), b.data());
    const auto sign3 = orient2d(a.data(), d.data(), c.data());
    const auto sign4 = orient2d(b.data(), c.data(), d.data());

    int n_negative = 0;
    int n_positive = 0;
    if (sign1 < 0) ++n_negative;
    if (sign1 > 0) ++n_positive;
    if (sign2 < 0) ++n_negative;
    if (sign2 > 0) ++n_positive;
    if (sign3 < 0) ++n_negative;
    if (sign3 > 0) ++n_positive;
    if (sign4 < 0) ++n_negative;
    if (sign4 > 0) ++n_positive;

    if (n_negative == 4 || n_positive == 4)
        return true;

    return false;
}

/**
 * Intersects the line a to b with c to d and returns the
 * intersection parameter of the segment from a to b.
 * Does not perform any checks.
 */
template <typename T>
inline T line_line_intersection_parameter_2d(
        const Eigen::Vector2<T>& a, const Eigen::Vector2<T>& b,
        const Eigen::Vector2<T>& c, const Eigen::Vector2<T>& d)
{
    // Precondition: not collinear
    return ((a[0] - c[0]) * (c[1] - d[1]) - (a[1] - c[1]) * (c[0] - d[0])) /
           ((a[0] - b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] - d[0]));
}

Eigen::Matrix2d transport_matrix(
        const OpenMesh::TriMesh& _target_mesh,
        OpenMesh::HalfedgeHandle _heh_from)
{
//    const OpenMesh::HalfedgeHandle heh_to = _target_mesh.opposite_halfedge_handle(_heh_from);
//    const OpenMesh::FaceHandle fh_from = _target_mesh.face_handle(_heh_from);
//    const OpenMesh::FaceHandle fh_to = _target_mesh.opposite_face_handle(heh_to);
//    ISM_ASSERT(fh_from.is_valid());
//    ISM_ASSERT(fh_to.is_valid());

//    const Triangle3<double> t_from = embed_first_triangle_natural(fh_from, _target_mesh, Geometry::Planar);
//    const Triangle3<double> t_to = embed_next_triangle_natural(heh_to, t_from, _target_mesh, Geometry::Planar);

//    const auto triangle_basis = [](const Triangle3<double>& _t)
//    {
//        const Vec2d a = drop_z(_t.a);
//        const Vec2d b = drop_z(_t.b);
//        const Vec2d c = drop_z(_t.c);
//        Mat2d result;
//        result.col(0) = a - c;
//        result.col(1) = b - c;
//        return result;
//    };

//    const Mat2d basis_from = triangle_basis(t_from);
//    const Mat2d basis_to = triangle_basis(t_to);

//    return basis_to.inverse() * basis_from;

    const auto triangle_basis = [&](auto fh)
    {
        OpenMesh::VertexHandle vh_a, vh_b, vh_c;
        handles(_target_mesh, fh, vh_a, vh_b, vh_c);
        LocalCoordinateSystem lcs(fh, _target_mesh);

        const Eigen::Vector2d a = lcs.point_to_local(_target_mesh.point(vh_a));
        const Eigen::Vector2d b = lcs.point_to_local(_target_mesh.point(vh_b));
        const Eigen::Vector2d c = lcs.point_to_local(_target_mesh.point(vh_c));
        Eigen::Matrix2d result;
        result.col(0) = a - c;
        result.col(1) = b - c;
        return result;
    };

    Eigen::Matrix2d M_from = triangle_basis(_target_mesh.face_handle(_heh_from));
    Eigen::Matrix2d M_to = triangle_basis(_target_mesh.opposite_face_handle(_heh_from));

    return M_to.inverse() * M_from;
}

OpenMesh::HalfedgeHandle reference_halfedge(const OpenMesh::TriMesh& _mesh, OpenMesh::EdgeHandle _eh)
{
    return _mesh.halfedge_handle(_eh, 0);
}

OpenMesh::HalfedgeHandle reference_halfedge(const OpenMesh::TriMesh& _mesh, OpenMesh::FaceHandle _fh)
{
    return _mesh.halfedge_handle(_fh);
}

// Angle between the "reference direction" of face(_heh) and _heh
double theta(const OpenMesh::TriMesh& _mesh, OpenMesh::HalfedgeHandle _heh)
{
    const OpenMesh::FaceHandle fh = _mesh.face_handle(_heh);
    OpenMesh::HalfedgeHandle ref_heh = reference_halfedge(_mesh, fh);
    double result = 0.0;
    while (ref_heh != _heh) {
        result += M_PI - _mesh.calc_sector_angle(ref_heh);
        ref_heh = _mesh.next_halfedge_handle(ref_heh);
    }
    return result;
}

// When going from heh to opposite(heh)
double levi_civita_connection(const OpenMesh::TriMesh& _mesh, OpenMesh::HalfedgeHandle _heh)
{
    const OpenMesh::EdgeHandle eh = _mesh.edge_handle(_heh);
    const OpenMesh::HalfedgeHandle ref_heh = reference_halfedge(_mesh, eh);
    const OpenMesh::HalfedgeHandle ref_heh_opp = _mesh.opposite_halfedge_handle(ref_heh);
    const double sign = (ref_heh == _heh) ? 1 : -1;
    const double angle_diff = - theta(_mesh, ref_heh) - M_PI + theta(_mesh, ref_heh_opp);
    return sign * angle_diff;
}

Snake trace_snake_in_frame_field(
        const OpenMesh::TriMesh& _mesh,
        const Eigen::MatrixXd& _V,
        const Eigen::MatrixXi& _F,
        const Eigen::MatrixXd& B1, // #F by 3. First basis vector per face.
        const Eigen::MatrixXd& B2, // #F by 3. Second basis vector per face.
        const Eigen::VectorXd& _x,
        const double _length,
        const bool _normalize_lengths)
{
    Snake result;
    const OpenMesh::FaceHandle start_fh = OpenMesh::FaceHandle(rand() % _mesh.n_faces());
    double rand_alpha = static_cast<double>(rand()) / RAND_MAX;
    double rand_beta = static_cast<double>(rand()) / RAND_MAX;
    if (rand_alpha + rand_beta >= 1)
    {
        rand_alpha = 1 - rand_alpha;
        rand_beta = 1 - rand_beta;
    }
    result.start = BarycentricPoint(start_fh, rand_alpha, rand_beta, _mesh);

    Eigen::Vector3d pa_B, pb_B, pc_B;
    points(_mesh, start_fh, pa_B, pb_B, pc_B);
    LocalCoordinateSystem lcs_B(start_fh, _mesh);
    Eigen::Matrix2d J; // Maps from barycentric vectors to vectors in the LCS of the face.
    J.col(0) = lcs_B.vector_to_local(pa_B - pc_B);
    J.col(1) = lcs_B.vector_to_local(pb_B - pc_B);
    const Eigen::Matrix2d J_inv = J.inverse(); // Maps from vectors in the LCS of the face to barycentric vectors.

    const int vec_idx = rand() % 2;
    const int vec_flip = rand() % 2;

    Eigen::Vector3d igl_vector = _x.segment(4 * start_fh.idx() + (vec_idx == 0 ? 0 : 2), 2);
    if (vec_flip == 0)
        igl_vector *= -1.0;
    if (_normalize_lengths)
        igl_vector.normalize();
    const Eigen::Vector3d world_vector = B1.row(start_fh.idx()) * igl_vector.x() + B2.row(start_fh.idx()) * igl_vector.y();

    const double lcs_speed = _length;
    Eigen::Vector2d lcs_dir = lcs_speed * lcs_B.vector_to_local(world_vector);

    const Eigen::Vector2d bary_dir = J_inv * lcs_dir;

    // The following state is updated throughout the tracing:
    BarycentricPoint current = result.start; // The current face and barycentric position in it.
    OpenMesh::HalfedgeHandle heh_current_entrace = OpenMesh::HalfedgeHandle(-1); // The halfedge through which the current face was entered.
    Eigen::Vector2d bary_trace_dir = bary_dir; // The tracing direction and length in barycentric coords of the current triangle.

    int safeguard = 0;
    while (true)
    {
        const OpenMesh::FaceHandle current_fh = current.fh(_mesh);
        if (heh_current_entrace.is_valid())
        {
            TINYAD_ASSERT(_mesh.face_handle(heh_current_entrace) == current_fh);
        }
        const Eigen::Vector2d bary_start = Eigen::Vector2d(current.alpha(), current.beta());
        const Eigen::Vector2d bary_end = bary_start + bary_trace_dir;

        // The boundaries of the current triangle, in barycentric coordinates.
        const OpenMesh::HalfedgeHandle boundary_hehs[3] =
        {
            current.heh_to_a(_mesh),
            current.heh_to_b(_mesh),
            current.heh_to_c(_mesh),
        };
        const Eigen::Vector2d bary_corners_from[3] =
        {
            Eigen::Vector2d(0,0), // Corner c
            Eigen::Vector2d(1,0), // Corner a
            Eigen::Vector2d(0,1), // Corner b
        };
        const Eigen::Vector2d bary_corners_to[3] =
        {
            Eigen::Vector2d(1,0), // Corner a
            Eigen::Vector2d(0,1), // Corner b
            Eigen::Vector2d(0,0), // Corner c
        };
        if (!in_triangle_exact_inclusive_2d(bary_start, bary_corners_from[0], bary_corners_from[1], bary_corners_from[2]))
            return Snake();

        // Search for an intersection of the current trace segment with one of the boundary segments
        OpenMesh::HalfedgeHandle heh_intersected = OpenMesh::HalfedgeHandle(-1);
        Eigen::Vector2d bary_corner_from;
        Eigen::Vector2d bary_corner_to;
        for (int i = 0; i < 3; ++i)
        {
            const OpenMesh::HalfedgeHandle boundary_heh = boundary_hehs[i];
            if (boundary_heh == heh_current_entrace)
            {
                // Don't perform intersection tests against the halfedge we entered from.
                continue;
            }
            bary_corner_from = bary_corners_from[i];
            bary_corner_to = bary_corners_to[i];
            if (intersects_exact_inclusive_2d(bary_start, bary_end, bary_corner_from, bary_corner_to))
            {
                heh_intersected = boundary_heh;
                break;
            }
        }

        if (heh_intersected.is_valid())
        {
            // We have intersected a triangle edge.
            // Very unlikely, but the target point might lie exactly on the triangle boundary: This case is unsupported.
            if (in_triangle_exact_inclusive_2d(bary_end, bary_corners_from[0], bary_corners_from[1], bary_corners_from[2]))
                return Snake();

            // Compute the intersection parameter on the crossed triangle edge (inexact).
            const double edge_intersection_param = line_line_intersection_parameter_2d(bary_corner_from, bary_corner_to, bary_start, bary_end);
            const PointOnEdge edge_intersection(heh_intersected, edge_intersection_param);
            result.intersections.push_back(edge_intersection);

            // Compute the remaining trace length (inexact).
            const double trace_intersection_param = line_line_intersection_parameter_2d(bary_start, bary_end, bary_corner_from, bary_corner_to);
            const double remaining_trace_param = 1.0 - trace_intersection_param;
            if (remaining_trace_param < 0)
                return Snake();

            const OpenMesh::HalfedgeHandle heh_next_entrance = _mesh.opposite_halfedge_handle(heh_intersected);
            const OpenMesh::FaceHandle fh_next = _mesh.face_handle(heh_next_entrance);

            // Update the current state.
            const Eigen::Matrix2d transport = transport_matrix(_mesh, heh_intersected);
            bary_trace_dir *= remaining_trace_param;
            bary_trace_dir = transport * bary_trace_dir;

            {
                Eigen::Vector3d pa_B, pb_B, pc_B;
                points(_mesh, fh_next, pa_B, pb_B, pc_B);
                LocalCoordinateSystem lcs_B(fh_next, _mesh);
                Eigen::Matrix2d J; // Maps from barycentric vectors to vectors in the LCS of the face.
                J.col(0) = lcs_B.vector_to_local(pa_B - pc_B);
                J.col(1) = lcs_B.vector_to_local(pb_B - pc_B);
                const Eigen::Matrix2d J_inv = J.inverse(); // Maps from vectors in the LCS of the face to barycentric vectors.

                const double adjustment_angle = levi_civita_connection(_mesh, heh_next_entrance);

                Eigen::Vector2d lcs_dir = J * bary_trace_dir;
                lcs_dir = Eigen::Rotation2D<double>(-adjustment_angle) * lcs_dir.eval();

                // Find closest matching vector
                std::vector<Eigen::Vector2d> igl_dir_candidates =
                {
                    _x.segment(4 * fh_next.idx(), 2).normalized(),
                    _x.segment(4 * fh_next.idx() + 2, 2).normalized(),
                    -_x.segment(4 * fh_next.idx(), 2).normalized(),
                    -_x.segment(4 * fh_next.idx() + 2, 2).normalized(),
                };
                Eigen::Vector3d world_dir = lcs_B.vector_to_global(lcs_dir);
                Eigen::Vector2d igl_dir(B1.row(fh_next.idx()).dot(world_dir), B2.row(fh_next.idx()).dot(world_dir));
                igl_dir.normalize();

                int i_max = -1;
                double max_dot = -INFINITY;
                for (int i = 0; i < 4; ++i)
                {
                    if (igl_dir.dot(igl_dir_candidates[i]) > max_dot)
                    {
                        i_max = i;
                        max_dot = igl_dir.dot(igl_dir_candidates[i]);
                    }
                }
                igl_dir = igl_dir_candidates[i_max];

                Eigen::Vector3d world_dir_new = B1.row(fh_next.idx()) * igl_dir.x() + B2.row(fh_next.idx()) * igl_dir.y();

                const double length_remaining = lcs_dir.norm();
                lcs_dir = length_remaining * lcs_B.vector_to_local(world_dir_new);

                bary_trace_dir = J_inv * lcs_dir;
            }

            heh_current_entrace = heh_next_entrance;
            current = edge_intersection.to_barycentric_point(_mesh, fh_next);
        }
        else
        {
            // We have not intersected a triangle edge. The target point of the tracing lies inside the triangle.
            if (!in_triangle_exact_inclusive_2d(bary_end, bary_corners_from[0], bary_corners_from[1], bary_corners_from[2]))
                return Snake();
            current.set_alpha_beta(bary_end[0], bary_end[1]);
            result.end = current;
            break;
        }
        ++safeguard;
        if (safeguard > 1000)
            return Snake();
    }
    return result;
}

void view_hatched_frame_field(
        const Eigen::MatrixXd& _V,
        const Eigen::MatrixXi& _F,
        const Eigen::MatrixXd& B1, // #F by 3. First basis vector per face.
        const Eigen::MatrixXd& B2, // #F by 3. Second basis vector per face.
        const Eigen::VectorXd& _x,
        const int _n_strokes,
        const double _stroke_length,
        const double _stroke_width,
        const bool _normalize_lengths,
        const bool _random_colors)
{
    srand(0);

    OpenMesh::TriMesh mesh = to_openmesh(_V, _F)   ;

    pm::Mesh hatches;
    auto hatches_pos = hatches.vertices().make_attribute<Eigen::Vector3d>();
    auto hatches_width = hatches.vertices().make_attribute<float>();
    auto hatches_color = hatches.vertices().make_attribute<tg::color3>();

    std::vector<tg::color4> colors =
    {
        BLUE, MAGENTA, GREEN, RED, PURPLE, /*YELLOW,*/ PETROL, ORANGE, TEAL, MAY_GREEN, BORDEAUX, LILAC,
    };

    for (int i = 0; i < _n_strokes; ++i)
    {
        Snake snake = trace_snake_in_frame_field(mesh, _V, _F, B1, B2, _x, _stroke_length, _normalize_lengths);
        if (!snake.start.is_valid())
            continue;

        {
            std::vector<pm::vertex_handle> snake_vertices;
            snake_vertices.push_back(hatches.vertices().add());
            hatches_pos[snake_vertices.back()] = snake.start.point(mesh);
            for (const auto& intersection : snake.intersections)
            {
                snake_vertices.push_back(hatches.vertices().add());
                hatches_pos[snake_vertices.back()] = intersection.point(mesh);
            }
            snake_vertices.push_back(hatches.vertices().add());
            hatches_pos[snake_vertices.back()] = snake.end.point(mesh);

            std::vector<double> snake_t;
            snake_t.push_back(0.0);
            for (size_t i = 0; i < snake_vertices.size() - 1; ++i)
            {
                const auto v0 = snake_vertices[i];
                const auto v1 = snake_vertices[i+1];
                const auto e = hatches.edges().add_or_get(v0, v1);
                const double e_length = pm::edge_length(e, hatches_pos);
                snake_t.push_back(snake_t.back() + e_length);
            }
            TINYAD_ASSERT_EQ(snake_vertices.size(), snake_t.size());
            const double max_t = snake_t.back();

            auto c = colors[i % colors.size()];
            tg::color3 color_tg(c[0], c[1], c[2]);
            for (size_t i = 0; i < snake_vertices.size(); ++i)
            {
                const auto v = snake_vertices[i];
                const double t = snake_t[i] / max_t;
                const double alpha = tg::clamp(std::sin(t * M_PI), 0, 1);
                hatches_width[v] = alpha * _stroke_width;
                if (_random_colors)
                    hatches_color[v] = color_tg;
                else
                    hatches_color[v] = tg::color3(0.0);
            }
        }
    }
    gv::view(gv::lines(hatches_pos).line_width_world(hatches_width), hatches_color, gv::no_shading);
}

inline gv::detail::raii_view_closer glow_view_frame_field(
        const Eigen::MatrixXd& _V,
        const Eigen::MatrixXi& _F,
        const Eigen::MatrixXd& _B1, // #F by 3. First basis vector per face.
        const Eigen::MatrixXd& _B2, // #F by 3. Second basis vector per face.
        const Eigen::VectorXd& _x,
        const int _n_strokes = 3000,
        const double _stroke_length = 0.2,
        const double _stroke_width = 0.00225,
        const bool _normalize_lengths = true,
        const bool _random_colors = true)
{
    auto v = glow_view_mesh(_V, _F);
    view_hatched_frame_field(_V, _F, _B1, _B2, _x, _n_strokes, _stroke_length, _stroke_width, _normalize_lengths, _random_colors);

    return v;
}

inline gv::detail::raii_view_closer glow_view_polycurl(
        const Eigen::MatrixXd& _V,
        const Eigen::MatrixXi& _F,
        const Eigen::VectorXd& _x,
        const Eigen::MatrixXi& _EV,
        const Eigen::MatrixXi& _EF,
        const Eigen::VectorX<std::complex<double>>& _e_f_conj,
        const Eigen::VectorX<std::complex<double>>& _e_g_conj,
        const double _curl_min,
        const double _curl_max)
{
    auto style = glow_default_style();

    // Convert to polymesh
    pm::Mesh m;
    auto pos = to_polymesh(_V, _F, m);

    // Compute polycurl and average on vertices
    auto v_polycurl = m.vertices().make_attribute<double>(0.0);
    for (int e_idx = 0; e_idx < _EV.rows(); ++e_idx)
    {
        int f_idx = _EF(e_idx, 0);
        int g_idx = _EF(e_idx, 1);
        std::complex<double> alpha_f(_x[4 * f_idx], _x[4 * f_idx + 1]);
        std::complex<double> beta_f(_x[4 * f_idx + 2], _x[4 * f_idx + 3]);
        std::complex<double> alpha_g(_x[4 * g_idx], _x[4 * g_idx + 1]);
        std::complex<double> beta_g(_x[4 * g_idx + 2], _x[4 * g_idx + 3]);

        double af_ef = (alpha_f * _e_f_conj[e_idx]).real();
        double ag_eg = (alpha_g * _e_g_conj[e_idx]).real();
        double bf_ef = (beta_f * _e_f_conj[e_idx]).real();
        double bg_eg = (beta_g * _e_g_conj[e_idx]).real();
        double c_f_0 = sqr(af_ef) * sqr(bf_ef);
        double c_g_0 = sqr(ag_eg) * sqr(bg_eg);
        double c_f_2 = -(sqr(af_ef) + sqr(bf_ef));
        double c_g_2 = -(sqr(ag_eg) + sqr(bg_eg));

        v_polycurl[m.vertices()[_EV(e_idx, 0)]] += sqr(c_f_0 - c_g_0) + sqr(c_f_2 - c_g_2);
        v_polycurl[m.vertices()[_EV(e_idx, 1)]] += sqr(c_f_0 - c_g_0) + sqr(c_f_2 - c_g_2);
    }

    for (auto v : m.vertices())
        v_polycurl[v] /= v.edges().size();

    // Color vertices
    auto color = m.vertices().make_attribute<tg::color4>(WHITE);
    for (auto v : m.vertices())
    {
        TINYAD_ASSERT_G(v_polycurl[v], 0.0);
        double lambda = (log(v_polycurl[v]) - log(_curl_min)) / (log(_curl_max) - log(_curl_min));
        lambda = tg::clamp(lambda, 0.0, 1.0);
        color[v] = tg::mix(WHITE, MAGENTA, lambda);
    }

    return gv::view(pos, color);
}
