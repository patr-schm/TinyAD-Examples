/*
 * (c) 2021 Patrick Schmidt
 */
#pragma once

#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Geometry/EigenVectorT.hh>
#include <OpenMesh/Core/IO/MeshIO.hh>

namespace OpenMesh
{

// Eigen traits
struct EigenTraits : public OpenMesh::DefaultTraits
{
  typedef Eigen::Vector3d Point;
  typedef Eigen::Vector3d Normal;
  typedef double TexCoord1D;
  typedef Eigen::Vector2d TexCoord2D;
  typedef Eigen::Vector3d TexCoord3D;
  typedef Color Color;
};

// Mesh type
using TriMesh = OpenMesh::TriMesh_ArrayKernelT<EigenTraits>;
using PolyMesh = OpenMesh::PolyMesh_ArrayKernelT<EigenTraits>;

}


