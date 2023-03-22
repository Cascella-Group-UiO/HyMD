.. _topology-label:

============================
Topology and structure file
============================
The conformational setup, bond network, molecules, particle types, names,
masses, charges, and velocities are specified in the structure/topology input
file in HDF5 format. The file is structured as a flag collection of datasets
attached to the root :code:`/` group.

The `HyMD-tutorial`_ has multiple examples of structure files for various levels
of complexity.

.. _HyMD-tutorial:
   https://github.com/Cascella-Group-UiO/HyMD-tutorial

Required datasets
=================
:/coordinates:
   :code:`array` shape [:code:`T`, :code:`N`, :code:`D`,]

   :code:`datatype` [:code:`float32` or :code:`float64`]

   The shape represents :code:`T` number of frames, :code:`N` number of particles, in :code:`D` dimensions. The data type may be either four or eight byte reals. If multiple frames are provided :code:`T > 1`, then the last frame is used as starting point for the new simulation by default.

:/indices:
   :code:`array` shape [:code:`N`,]

   :code:`datatype` [:code:`int32` or :code:`int64`]

   The shape represents one particle index per :code:`N` particles. The data type may be either 32 or 64 bit integers.

:/names:
   :code:`array` shape [:code:`N`,]

   :code:`datatype` [:code:`string` (:code:`length` between :code:`1` and :code:`16`)]

   The shape represents one particle name per :code:`N` particles. The datatype may be a string of length :code:`=<16`.



Optional datasets
=================
:/velocities:
   :code:`array` shape [:code:`T`, :code:`N`, :code:`D`,]

   :code:`datatype` [:code:`float32` or :code:`float64`]

   The shape represents :code:`T` number of frames, :code:`N` number of particles, in :code:`D` dimensions. The data type may be either four or eight byte reals. If multiple frames are provided :code:`T > 1`, then the last frame is used as starting point for the new simulation by default.

:/types:
   :code:`array` shape [:code:`N`,]

   :code:`datatype` [:code:`int32` or :code:`int64`]

   The shape represents one particle type index per :code:`N` particles. The data type may be either 32 or 64 bit integers.

:/molecules:
   :code:`array` shape [:code:`N`,]

   :code:`datatype` [:code:`int32` or :code:`int64`]

   The shape represents one molecule index per :code:`N` particles. The data type may be either 32 or 64 bit integers.

:/bonds:
   :code:`array` shape [:code:`N`, :code:`B`,]

   :code:`datatype` [:code:`int32` or :code:`int64`]

   The shape represents :code:`B` bond specifications (indices of different particles) per :code:`N` particles. The data type may be either 32 or 64 bit integers.

:/charge:
   :code:`array` shape [:code:`N`,]

   :code:`datatype` [:code:`float32` or :code:`float64`]

   The shape represents one particle charge per :code:`N` particles. The data type may be either four or eight byte reals.

:/box:
   :code:`array` shape [:code:`3`,]

   :code:`datatype` [:code:`float32` or :code:`float64`]

   The shape represents the initial box dimensions in Cartesian coordinates.
