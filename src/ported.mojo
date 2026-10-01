"""Mojo port of the compute-bound numeric kernels of `genesis-world` 1.4.2.

Upstream is the Genesis physics engine, whose kernels are Taichi/quadrants
`@qd.func`s over `qd.Vector` / `qd.Matrix`. The functions below appear in the
same order, with the same names, the same argument order and the same branch
structure as the upstream file each one came from; the section banners name
that file. `src/capi.mojo` holds the `@export` wrappers.

Two dialects, three divergences, all of them forced by the FFI and by Mojo's
type system:

  * `qd.Vector(3)` becomes `Vec3` and `qd.Matrix(3, 3)` becomes `Mat3`, defined
    at the top of this file. They are plain structs of `Float64` so that `v[0]`,
    `v.dot(w)`, `A @ B`, `A.transpose()` and `A.determinant()` mean exactly
    what they mean upstream. `qd.Vector.zero(gs.qd_float, 3)` is `Vec3()`.
  * `gs.qd_float` is `Float64` and `gs.qd_int` is `Int32` throughout.
  * Upstream reaches engine state through `array_class` dataclasses
    (`dyn_info.geoms.data[i_g][0]`). Mojo cannot, and neither can the C ABI, so
    every field a function reads arrives as an explicit pointer argument, named
    after the field it replaces.

A `qd.static(range(3))` unrolled scan is written out where it is on a hot path. Mojo does not
unroll `for i in range(3)`, and a runtime index into `Vec3` / `Mat3` is a branch tree, so the
loop form costs several times what the straight-line form does on `Mat3` products. The
runtime-indexed accessors are still correct on CPU, so the Metal FIXME in
`func_triangle_affine_coords` does not apply.
"""

from std.math import sqrt, log, acos, atan2, floor, ceil, exp, sin, cos, clamp
from max.algorithm import parallelize

# `activate` puts one direction in each SIMD lane. The direction table is interleaved, so a
# block is filled by a 3x4 register transpose rather than by one contiguous vector load, and
# that fixes the block at four lanes whatever the machine's native float64 width is.
comptime _DW: Int = 4
comptime SF64 = SIMD[DType.float64, _DW]
comptime SI64 = SIMD[DType.int64, _DW]

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int32, AnyOrigin[mut=True]]

# `activate` splits the support table over the workers the host asks for, and only once the
# table holds at least `_ACTIVATE_PARALLEL_MIN` dot products, which is well past the cost of
# a parallel launch. The host owns the count because a launch that cannot spawn its threads
# aborts the process rather than returning an error.
comptime _ACTIVATE_PARALLEL_MIN: Int = 1 << 24

# `qd.float64` and `qd.int32` are what the engine selects for a 64-bit CPU run.
comptime F64 = Float64
comptime I32 = Int32

# gs.constants
comptime GEOM_TYPE_PLANE: Int32 = 0
comptime GEOM_TYPE_SPHERE: Int32 = 1
comptime GEOM_TYPE_ELLIPSOID: Int32 = 2
comptime GEOM_TYPE_CYLINDER: Int32 = 3
comptime GEOM_TYPE_CAPSULE: Int32 = 4
comptime GEOM_TYPE_BOX: Int32 = 5
comptime GEOM_TYPE_MESH: Int32 = 6
comptime GEOM_TYPE_TERRAIN: Int32 = 7

# collider/constants.py :: RETURN_CODE
comptime RETURN_CODE_SUCCESS: Int32 = 0
comptime RETURN_CODE_FAIL: Int32 = 1

# gs.EPS
comptime EPS: Float64 = 1e-15

# collider_info.gjk
comptime FLOAT_MAX: Float64 = 3.4028234663852886e38
comptime FLOAT_MIN: Float64 = 1e-15
comptime FLOAT_MIN_SQ: Float64 = 1e-30


# =====================================================================================
# ==================================== types ==========================================
# =====================================================================================


struct Vec3(ImplicitlyCopyable):
    var x: Float64
    var y: Float64
    var z: Float64

    def __init__(out self, x: Float64 = 0.0, y: Float64 = 0.0, z: Float64 = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def __getitem__(self, i: Int) -> Float64:
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        return self.z

    def __setitem__(mut self, i: Int, v: Float64):
        if i == 0:
            self.x = v
        elif i == 1:
            self.y = v
        else:
            self.z = v

    def dot(self, o: Vec3) -> Float64:
        return self.x * o.x + self.y * o.y + self.z * o.z

    def cross(self, o: Vec3) -> Vec3:
        return Vec3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )

    def norm_sqr(self) -> Float64:
        return self.dot(self)

    def norm(self) -> Float64:
        return sqrt(self.norm_sqr())

    def normalized(self) -> Vec3:
        return self * (1.0 / self.norm())

    def __add__(self, o: Vec3) -> Vec3:
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o: Vec3) -> Vec3:
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __sub__(self, s: Float64) -> Vec3:
        return Vec3(self.x - s, self.y - s, self.z - s)

    def __add__(self, s: Float64) -> Vec3:
        return Vec3(self.x + s, self.y + s, self.z + s)

    # `Mat3` is built from its rows, so row i of u (x) v is u[i] * v.
    def outer_product(self, o: Vec3) -> Mat3:
        return Mat3(self[0] * o, self[1] * o, self[2] * o)

    def __neg__(self) -> Vec3:
        return Vec3(-self.x, -self.y, -self.z)

    def __mul__(self, o: Vec3) -> Vec3:
        return Vec3(self.x * o.x, self.y * o.y, self.z * o.z)

    def __mul__(self, s: Float64) -> Vec3:
        return Vec3(self.x * s, self.y * s, self.z * s)

    def __rmul__(self, s: Float64) -> Vec3:
        return self * s

    def __truediv__(self, s: Float64) -> Vec3:
        return Vec3(self.x / s, self.y / s, self.z / s)


struct Quat(ImplicitlyCopyable):
    var w: Float64
    var x: Float64
    var y: Float64
    var z: Float64

    def __init__(out self, w: Float64 = 0.0, x: Float64 = 0.0, y: Float64 = 0.0, z: Float64 = 0.0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __getitem__(self, i: Int) -> Float64:
        if i == 0:
            return self.w
        elif i == 1:
            return self.x
        elif i == 2:
            return self.y
        return self.z

    def __setitem__(mut self, i: Int, v: Float64):
        if i == 0:
            self.w = v
        elif i == 1:
            self.x = v
        elif i == 2:
            self.y = v
        else:
            self.z = v

    def norm_sqr(self) -> Float64:
        return self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z

    def normalized(self) -> Quat:
        var n = sqrt(self.norm_sqr())
        return Quat(self.w / n, self.x / n, self.y / n, self.z / n)

    def __add__(self, o: Quat) -> Quat:
        return Quat(self.w + o.w, self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o: Quat) -> Quat:
        return Quat(self.w - o.w, self.x - o.x, self.y - o.y, self.z - o.z)

    def __neg__(self) -> Quat:
        return Quat(-self.w, -self.x, -self.y, -self.z)

    def __mul__(self, s: Float64) -> Quat:
        return Quat(self.w * s, self.x * s, self.y * s, self.z * s)

    def __rmul__(self, s: Float64) -> Quat:
        return self * s


struct Mat3(ImplicitlyCopyable):
    var r0: Vec3
    var r1: Vec3
    var r2: Vec3

    def __init__(out self, a: Vec3 = Vec3(), b: Vec3 = Vec3(), c: Vec3 = Vec3()):
        self.r0 = a
        self.r1 = b
        self.r2 = c

    @staticmethod
    def identity() -> Mat3:
        return Mat3(Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0), Vec3(0.0, 0.0, 1.0))

    def row(self, i: Int) -> Vec3:
        if i == 0:
            return self.r0
        elif i == 1:
            return self.r1
        return self.r2

    def at(self, i: Int, j: Int) -> Float64:
        return self.row(i)[j]

    def set(mut self, i: Int, j: Int, v: Float64):
        if i == 0:
            self.r0[j] = v
        elif i == 1:
            self.r1[j] = v
        else:
            self.r2[j] = v

    def transpose(self) -> Mat3:
        return Mat3(
            Vec3(self.r0[0], self.r1[0], self.r2[0]),
            Vec3(self.r0[1], self.r1[1], self.r2[1]),
            Vec3(self.r0[2], self.r1[2], self.r2[2]),
        )

    def __matmul__(self, o: Mat3) -> Mat3:
        return Mat3(
            self.r0[0] * o.r0 + self.r0[1] * o.r1 + self.r0[2] * o.r2,
            self.r1[0] * o.r0 + self.r1[1] * o.r1 + self.r1[2] * o.r2,
            self.r2[0] * o.r0 + self.r2[1] * o.r1 + self.r2[2] * o.r2,
        )

    def __matmul__(self, v: Vec3) -> Vec3:
        return Vec3(self.r0.dot(v), self.r1.dot(v), self.r2.dot(v))

    def __add__(self, o: Mat3) -> Mat3:
        return Mat3(self.r0 + o.r0, self.r1 + o.r1, self.r2 + o.r2)

    def __sub__(self, o: Mat3) -> Mat3:
        return Mat3(self.r0 - o.r0, self.r1 - o.r1, self.r2 - o.r2)

    def __mul__(self, s: Float64) -> Mat3:
        return Mat3(self.r0 * s, self.r1 * s, self.r2 * s)

    def __rmul__(self, s: Float64) -> Mat3:
        return self * s

    def __mul__(self, o: Mat3) -> Mat3:
        return Mat3(self.r0 * o.r0, self.r1 * o.r1, self.r2 * o.r2)

    def __truediv__(self, s: Float64) -> Mat3:
        return Mat3(self.r0 / s, self.r1 / s, self.r2 / s)

    def determinant(self) -> Float64:
        return (
            self.r0[0] * (self.r1[1] * self.r2[2] - self.r1[2] * self.r2[1])
            - self.r0[1] * (self.r1[0] * self.r2[2] - self.r1[2] * self.r2[0])
            + self.r0[2] * (self.r1[0] * self.r2[1] - self.r1[1] * self.r2[0])
        )


def _diag(v: Vec3) -> Mat3:
    """The matrix with `v` on its diagonal; every off-diagonal entry of a fresh `Mat3` is zero."""
    return Mat3(Vec3(v[0], 0.0, 0.0), Vec3(0.0, v[1], 0.0), Vec3(0.0, 0.0, v[2]))


def _clamp_diag(S: Mat3, lo: Float64, hi: Float64) -> Vec3:
    """Upstream's `for d in range(3)` over the diagonal, written out so `S` stays in registers."""
    return Vec3(min(max(S.r0[0], lo), hi), min(max(S.r1[1], lo), hi), min(max(S.r2[2], lo), hi))


def _max3(v: Vec3, s: Float64) -> Vec3:
    return Vec3(max(v[0], s), max(v[1], s), max(v[2], s))


# =====================================================================================
# ============================ genesis/utils/geom.py ===================================
# =====================================================================================


def qd_transform_by_quat_fast(v: Vec3, quat: Quat) -> Vec3:
    """Rotate a vector by a quaternion that is already normalized.
    Prefer this over the general form whenever the quaternion is known to be a unit one, for exactness rather than for
    speed: the general form divides by the squared norm, which is a no-op for a unit quaternion in exact arithmetic
    yet leaves a component that should be untouched a few units in the last place off, by an amount that depends on
    the rotation. A body resting flat on a plane then reports a height that varies with its orientation about the
    plane's normal, and the sign of that variation decides whether it is in contact at all. The two cross products
    here contribute exactly zero to a component the rotation preserves, so it comes back bit for bit.

    Assumptions:
    - quat must be normalized
    """
    var q_w = quat[0]
    var q_x = quat[1]
    var q_y = quat[2]
    var q_z = quat[3]
    var u = Vec3(q_x, q_y, q_z)
    var t = 2.0 * u.cross(v)
    return v + q_w * t + u.cross(t)


def qd_transform_by_trans_quat_fast(pos: Vec3, trans: Vec3, quat: Quat) -> Vec3:
    """Place a point by a translation and a quaternion that is already normalized. See qd_transform_by_quat_fast."""
    return qd_transform_by_quat_fast(pos, quat) + trans


def qd_quat_to_R(quat: Quat, eps: Float64) -> Mat3:
    """Converts quaternion to 3x3 rotation matrix."""
    var R = Mat3.identity()
    var d = quat.norm_sqr()
    if d > eps:
        var s = 2.0 / d
        var w = quat[0]
        var x = quat[1]
        var y = quat[2]
        var z = quat[3]
        var xs = x * s
        var ys = y * s
        var zs = z * s
        var wx = w * xs
        var wy = w * ys
        var wz = w * zs
        var xx = x * xs
        var xy = x * ys
        var xz = x * zs
        var yy = y * ys
        var yz = y * zs
        var zz = z * zs
        R = Mat3(
            Vec3(1.0 - (yy + zz), xy - wz, xz + wy),
            Vec3(xy + wz, 1.0 - (xx + zz), yz - wx),
            Vec3(xz - wy, yz + wx, 1.0 - (xx + yy)),
        )

    return R


def qd_inv_quat(quat: Quat) -> Quat:
    return Quat(quat[0], -quat[1], -quat[2], -quat[3])


def qd_quat_mul(q: Quat, u: Quat) -> Quat:
    return Quat(
        q.w * u.w - q.x * u.x - q.y * u.y - q.z * u.z,
        q.w * u.x + q.x * u.w + q.y * u.z - q.z * u.y,
        q.w * u.y - q.x * u.z + q.y * u.w + q.z * u.x,
        q.w * u.z + q.x * u.y - q.y * u.x + q.z * u.w,
    )


def qd_transform_quat_by_quat(v: Quat, u: Quat) -> Quat:
    """Transforms quat_v by quat_u.

    This is equivalent to quatmul(quat_u, quat_v) or R_u @ R_v
    """
    var vec = qd_quat_mul(u, v)
    return vec.normalized()


def qd_transform_by_quat(v: Vec3, quat: Quat) -> Vec3:
    var q_w = quat[0]
    var q_x = quat[1]
    var q_y = quat[2]
    var q_z = quat[3]
    var q_xx = q_x * q_x
    var q_xy = q_x * q_y
    var q_xz = q_x * q_z
    var q_wx = q_x * q_w
    var q_yy = q_y * q_y
    var q_yz = q_y * q_z
    var q_wy = q_y * q_w
    var q_zz = q_z * q_z
    var q_wz = q_z * q_w
    var q_ww = q_w * q_w

    return Vec3(
        v.x * (q_xx + q_ww - q_yy - q_zz) + v.y * (2.0 * q_xy - 2.0 * q_wz) + v.z * (2.0 * q_xz + 2.0 * q_wy),
        v.x * (2.0 * q_wz + 2.0 * q_xy) + v.y * (q_ww - q_xx + q_yy - q_zz) + v.z * (-2.0 * q_wx + 2.0 * q_yz),
        v.x * (-2.0 * q_wy + 2.0 * q_xz) + v.y * (2.0 * q_wx + 2.0 * q_yz) + v.z * (q_ww - q_xx - q_yy + q_zz),
    ) / (q_ww + q_xx + q_yy + q_zz)


def qd_inv_transform_by_quat(v: Vec3, quat: Quat) -> Vec3:
    return qd_transform_by_quat(v, qd_inv_quat(quat))


def qd_transform_by_trans_quat(pos: Vec3, trans: Vec3, quat: Quat) -> Vec3:
    return qd_transform_by_quat(pos, quat) + trans


def qd_inv_transform_by_trans_quat(pos: Vec3, trans: Vec3, quat: Quat) -> Vec3:
    return qd_transform_by_quat(pos - trans, qd_inv_quat(quat))


def qd_normalize(v: Vec3, eps: Float64) -> Vec3:
    # The guard floors the norm, not the radicand: under the square root it would compare against a squared length and
    # bias every vector shorter than its own square root. Flooring the norm leaves any vector longer than 'eps' exactly
    # unit and still keeps a vanishing one finite.
    return v / max(v.norm(), eps)


def qd_orthogonals(a: Vec3) -> Tuple[Vec3, Vec3]:
    """Returns orthogonal vectors `b` and `c`, given a normal vector `a`.

    Note that `a` is assumed to be normalized.
    """
    var b = Vec3()
    if abs(a[1]) < 0.5:
        b[0] = -a[0] * a[1]
        b[1] = 1.0 - a[1] ** 2
        b[2] = -a[2] * a[1]
    else:
        b[0] = -a[0] * a[2]
        b[1] = -a[1] * a[2]
        b[2] = 1.0 - a[2] ** 2
    b = b.normalized()
    return (b.cross(a), b)


# =====================================================================================
# ==== genesis/engine/solvers/rigid/collider/utils.py (func_det3) ====================
# =====================================================================================


def func_det3(v1: Vec3, v2: Vec3, v3: Vec3) -> Float64:
    """Compute the determinant of a 3x3 matrix M = [v1 | v2 | v3]."""
    return (
        v1[0] * (v2[1] * v3[2] - v2[2] * v3[1])
        - v1[1] * (v2[0] * v3[2] - v2[2] * v3[0])
        + v1[2] * (v2[0] * v3[1] - v2[1] * v3[0])
    )


# =====================================================================================
# ================== genesis/.../collider/support_field.py =============================
# =====================================================================================


def _polar_band(support_res: Int) -> Tuple[Int, Int]:
    """First stored polar row of a chart, and how many rows it stores."""
    return (support_res // 4 - 1, support_res // 2 + 3)


def get_direction_grid(dst: Ptr, support_res: Int) -> Int:
    """Unit direction of every cell of the two spherical charts, as [chart, azimuth, polar].

    A spherical chart cannot resolve the azimuth of a direction along its own polar axis, and which of the tied
    support vertices of such a direction wins then depends on the signs of zeros, hence on how the whole scene is
    oriented. The second chart carries its poles a quarter turn away, so every direction is at least 45 degrees
    off the poles of whichever chart answers for it, and no lookup is ever resolved by a chart degenerate there.
    """
    # The second chart is the first one turned a quarter turn about x, which carries its poles onto +/- y. A chart
    # answers only for the directions it is at least 45 degrees off its own poles for, so only that equatorial
    # band of rows is ever read and only it is stored, one guard row either side for the neighbours the lookup
    # reads. The two bands together hold about as many cells as the single full chart they replace.
    var band_start, band_rows = _polar_band(support_res)
    var n_cells = 0
    for chart in range(2):
        for i in range(support_res):
            var theta = Float64(i) / Float64(support_res) * 2.0 * 3.141592653589793 - 3.141592653589793
            for j in range(support_res):
                if j < band_start or j >= band_start + band_rows:
                    continue
                var phi = Float64(j) / Float64(support_res) * 3.141592653589793
                var x = sin(phi) * cos(theta)
                var y = sin(phi) * sin(theta)
                var z = cos(phi)
                var o = 3 * n_cells
                if chart == 0:
                    dst.unsafe_offset(o)[] = x
                    dst.unsafe_offset(o + 1)[] = y
                    dst.unsafe_offset(o + 2)[] = z
                else:
                    dst.unsafe_offset(o)[] = x
                    dst.unsafe_offset(o + 1)[] = z
                    dst.unsafe_offset(o + 2)[] = -y
                n_cells += 1
    return n_cells


def direction_grid_nums(support_res: Int) -> Int:
    """Number of directions `_get_direction_grid` emits (both charts)."""
    var _, band_rows = _polar_band(support_res)
    return 2 * support_res * band_rows


def _store_cell(support_v: Ptr, support_vid: IPtr, cell: Int, init_pos: Ptr, o: Int):
    """The winning vertex of one direction, written as a row of the support table."""
    var c = 3 * cell
    support_v.unsafe_offset(c)[] = init_pos.unsafe_load(o)
    support_v.unsafe_offset(c + 1)[] = init_pos.unsafe_load(o + 1)
    support_v.unsafe_offset(c + 2)[] = init_pos.unsafe_load(o + 2)
    support_vid.unsafe_offset(cell)[] = I32(o // 3)


def _activate_cell(
    init_pos: Ptr, vert_base: Int, n_verts: Int, v1: Ptr, i_v: Int, cell: Int,
    support_v: Ptr, support_vid: IPtr,
):
    """One direction against one geom, scanning the vertex table in order."""
    var d0 = v1.unsafe_load(3 * i_v)
    var d1 = v1.unsafe_load(3 * i_v + 1)
    var d2 = v1.unsafe_load(3 * i_v + 2)
    # numpy's argmax takes the first maximum, so a strict `>` keeps the same winner on a tie.
    var fmax = -FLOAT_MAX
    var imax = 0
    for i_w in range(n_verts):
        var o = vert_base + 3 * i_w
        var vdot = d0 * init_pos.unsafe_load(o) + d1 * init_pos.unsafe_load(o + 1) + d2 * init_pos.unsafe_load(o + 2)
        if vdot > fmax:
            fmax = vdot
            imax = i_w
    _store_cell(support_v, support_vid, cell, init_pos, vert_base + 3 * imax)


def _activate_block(
    init_pos: Ptr, vert_base: Int, n_verts: Int, v1: Ptr, i_v: Int, cell: Int,
    support_v: Ptr, support_vid: IPtr,
):
    """`_DW` directions at once, one per SIMD lane, so the vertex table is read once for `_DW` dots.

    Each lane scans the whole vertex table in order with a strict `>`, exactly as the scalar
    scan does, so a lane's winner is the vertex `np.argmax` would pick for its own direction.
    """
    var o3 = 3 * i_v
    var d0 = SIMD[DType.float64, _DW](
        v1.unsafe_load(o3), v1.unsafe_load(o3 + 3), v1.unsafe_load(o3 + 6), v1.unsafe_load(o3 + 9)
    )
    var d1 = SIMD[DType.float64, _DW](
        v1.unsafe_load(o3 + 1), v1.unsafe_load(o3 + 4), v1.unsafe_load(o3 + 7), v1.unsafe_load(o3 + 10)
    )
    var d2 = SIMD[DType.float64, _DW](
        v1.unsafe_load(o3 + 2), v1.unsafe_load(o3 + 5), v1.unsafe_load(o3 + 8), v1.unsafe_load(o3 + 11)
    )
    var best = SF64(-FLOAT_MAX)
    var bidx = SI64(0)
    for i_w in range(n_verts):
        var o = vert_base + 3 * i_w
        var vdot = d0 * init_pos.unsafe_load(o) + d1 * init_pos.unsafe_load(o + 1) + d2 * init_pos.unsafe_load(o + 2)
        var gt = vdot.gt(best)
        best = gt.select(vdot, best)
        bidx = gt.select(SI64(i_w), bidx)
    for k in range(_DW):
        _store_cell(support_v, support_vid, cell + k, init_pos, vert_base + 3 * Int(bidx[k]))


def _activate_geom(
    init_pos: Ptr, vert_base: Int, n_verts: Int, v1: Ptr, i_v0: Int, num_v: Int, cell0: Int,
    support_v: Ptr, support_vid: IPtr,
):
    """`num_v` directions of one geom from `i_v0`: the SIMD block, then the scalar `num_v % _DW` tail."""
    var i_v = i_v0
    while i_v + _DW <= i_v0 + num_v:
        _activate_block(init_pos, vert_base, n_verts, v1, i_v, cell0 + i_v - i_v0, support_v, support_vid)
        i_v += _DW
    while i_v < i_v0 + num_v:
        _activate_cell(init_pos, vert_base, n_verts, v1, i_v, cell0 + i_v - i_v0, support_v, support_vid)
        i_v += 1


def activate(
    init_pos: Ptr,
    geoms_vert_start: IPtr,
    geoms_vert_end: IPtr,
    n_geoms: Int,
    v1: Ptr,
    num_v: Int,
    support_v: Ptr,
    support_vid: IPtr,
    support_cell_start: IPtr,
    n_workers: Int = 1,
):
    """Precompute the support table: for every geom and every direction, the argmax vertex.

    This is the body of `SupportField.activate` with the qd kernel call replaced by its own
    writes, and it is the hottest pure-numeric loop in the support field: num_v x n_verts
    dot products per geom. Upstream accumulates the cell offset across geoms, but every geom
    contributes exactly `num_v` cells, so the offset of geom `i_g` is `i_g * num_v` and the
    table is a flat `(geom, direction)` grid whose cells are independent of each other, which
    is what the `parallelize` below divides. A work item is `_DW` consecutive directions of one
    geom, so it never straddles two geoms and never splits a SIMD block.
    """
    for i_g in range(n_geoms):
        support_cell_start.unsafe_offset(i_g)[] = I32(i_g * num_v)

    var n_verts_total = 0
    for i_g in range(n_geoms):
        n_verts_total += Int(geoms_vert_end.unsafe_load(i_g)) - Int(geoms_vert_start.unsafe_load(i_g))
    var chunks_per_geom = (num_v + _DW - 1) // _DW
    var n_chunks = n_geoms * chunks_per_geom

    def work(c: Int) {imm}:
        var i_g = c // chunks_per_geom
        var i_v0 = _DW * (c - i_g * chunks_per_geom)
        var vert_start = Int(geoms_vert_start.unsafe_load(i_g))
        _activate_geom(
            init_pos, 3 * vert_start, Int(geoms_vert_end.unsafe_load(i_g)) - vert_start, v1,
            i_v0, min(num_v - i_v0, _DW), i_g * num_v + i_v0, support_v, support_vid,
        )

    # One dot product per vertex per direction, over every geom: `n_verts_total` is
    # already the sum over geoms, so this is the true dot count, not a per-geom one.
    var workers = min(n_chunks, max(n_workers, 1))
    if workers > 1 and n_verts_total * num_v >= _ACTIVATE_PARALLEL_MIN:
        parallelize(work, n_chunks, workers)
    else:
        for c in range(n_chunks):
            work(c)


def _func_direction_grid_coords(d_mesh: Vec3, support_res: Int) -> Tuple[Int32, Float64, Float64]:
    """Chart of a mesh-frame direction and its continuous cell coordinates within that chart.

    The chart whose polar axis the direction is least aligned with answers, which leaves it at least 45 degrees off
    that chart's poles since the smaller of the two alignments never exceeds one over root two. The azimuth is then
    always well conditioned, so a direction along one of the geom's own axes - the normal of a face resting flat,
    whose support vertices tie - is resolved by a chart that has no degeneracy there, and a geom and any rotated copy
    of it pick the same vertex.
    """
    var i_chart: Int32 = 0
    var d_chart = d_mesh
    if abs(d_mesh[2]) > abs(d_mesh[1]):
        # The first chart's poles sit on +/- z, so a direction closer to them is handed to the second chart, whose
        # own coordinates see it a quarter turn away.
        i_chart = 1
        d_chart = Vec3(d_mesh[0], -d_mesh[2], d_mesh[1])

    var theta = atan2(d_chart[1], d_chart[0])  # [-pi, pi]
    var phi = acos(d_chart[2])  # [0, pi]

    return (i_chart, (theta + 3.141592653589793) / 3.141592653589793 / 2.0 * Float64(support_res), phi / 3.141592653589793 * Float64(support_res))


def _func_support_mesh(
    i_g: Int,
    d_mesh: Vec3,
    support_res: Int,
    support_cell_start: IPtr,
    support_vid: IPtr,
    support_v: Ptr,
) -> Tuple[Vec3, Int32]:
    """support point at mesh frame coordinate."""
    var dot_max = -1e20
    var v = Vec3(0.0, 0.0, 0.0)
    var vid: Int32 = 0

    var band_start, band_rows = _polar_band(support_res)
    var i_chart, ii, jj = _func_direction_grid_coords(d_mesh, support_res)
    var chart_start = Int32(Int(support_cell_start.unsafe_load(i_g)) + Int(i_chart) * support_res * band_rows)

    for i4 in range(4):
        var i: Int32
        var j: Int32
        if i4 % 2:
            i = Int32(Int(ceil(ii))) % I32(support_res)
        else:
            i = Int32(Int(floor(ii))) % I32(support_res)

        if i4 // 2 > 0:
            j = Int32(clamp(Int(ceil(jj)) - band_start, 0, band_rows - 1))
        else:
            j = Int32(clamp(Int(floor(jj)) - band_start, 0, band_rows - 1))

        var support_idx = Int32(Int(chart_start) + Int(i) * band_rows + Int(j))
        var _vid = support_vid.unsafe_load(support_idx)
        var o = 3 * Int(support_idx)
        var pos = Vec3(support_v.unsafe_load(o), support_v.unsafe_load(o + 1), support_v.unsafe_load(o + 2))
        var d = pos.dot(d_mesh)

        if d > dot_max:
            v = pos
            dot_max = d
            vid = _vid

    return (v, vid)


def _func_support_mesh_exhaustive(i_g: Int, d_mesh: Vec3, vert_start: Int, vert_end: Int, init_pos: Ptr) -> Tuple[Vec3, Int32]:
    """Support vertex by exhaustive scan in the mesh frame, the first maximum winning, like the scan the reference
    engine's fallback collision pipeline runs below its hill-climb threshold."""
    var dot_max = -1e20
    var i_max = vert_start
    for i_v in range(vert_start, vert_end):
        var o = 3 * i_v
        var pos = Vec3(init_pos.unsafe_load(o), init_pos.unsafe_load(o + 1), init_pos.unsafe_load(o + 2))
        var vdot = d_mesh.dot(pos)
        if vdot > dot_max:
            dot_max = vdot
            i_max = i_v
    var o = 3 * i_max
    return (Vec3(init_pos.unsafe_load(o), init_pos.unsafe_load(o + 1), init_pos.unsafe_load(o + 2)), I32(i_max))


def _func_support_sphere(
    i_g: Int, d: Vec3, pos: Vec3, quat: Quat, shrink: Bool, geoms_data: Ptr
) -> Tuple[Vec3, Vec3, Int32]:
    var sphere_center = pos
    var sphere_radius = geoms_data.unsafe_load(3 * i_g)

    # Shrink the sphere to a point
    var v = sphere_center
    var v_ = Vec3()
    var vid: Int32 = -1
    if not shrink:
        v = v + d * sphere_radius

        # Local position of the support point
        var local_d = qd_inv_transform_by_quat(d, quat)
        v_ = local_d * sphere_radius

    return (v, v_, vid)


def _func_support_ellipsoid(i_g: Int, d: Vec3, pos: Vec3, quat: Quat, geoms_data: Ptr) -> Vec3:
    var a = geoms_data.unsafe_load(3 * i_g)
    var b = geoms_data.unsafe_load(3 * i_g + 1)
    var c = geoms_data.unsafe_load(3 * i_g + 2)

    # Transform direction to ellipsoid local frame
    var d_local = qd_inv_transform_by_quat(d, quat)

    # Support point in local frame: diag(a², b², c²) · d_local / ‖diag(a, b, c) · d_local‖
    var sx = a * a * d_local[0]
    var sy = b * b * d_local[1]
    var sz = c * c * d_local[2]
    var norm = sqrt(a * a * d_local[0] ** 2 + b * b * d_local[1] ** 2 + c * c * d_local[2] ** 2)
    var s_local = Vec3(sx / norm, sy / norm, sz / norm)

    return pos + qd_transform_by_quat(s_local, quat)


def _func_support_capsule(
    i_g: Int, d: Vec3, pos: Vec3, quat: Quat, shrink: Bool, geoms_data: Ptr
) -> Vec3:
    """Support function for capsule geometry."""
    var res = Vec3(0.0, 0.0, 0.0)
    var capsule_center = pos
    var capsule_radius = geoms_data.unsafe_load(3 * i_g)
    var capsule_halflength = 0.5 * geoms_data.unsafe_load(3 * i_g + 1)

    if shrink:
        var local_dir = qd_transform_by_quat(d, qd_inv_quat(quat))
        res[2] = capsule_halflength if local_dir[2] >= 0.0 else -capsule_halflength
        res = qd_transform_by_trans_quat(res, capsule_center, quat)
    else:
        var capsule_axis = qd_transform_by_quat(Vec3(0.0, 0.0, 1.0), quat)
        var capsule_endpoint_side = -1.0 if d.dot(capsule_axis) < 0.0 else 1.0
        var capsule_endpoint = capsule_center + capsule_halflength * capsule_endpoint_side * capsule_axis
        res = capsule_endpoint + d * capsule_radius
    return res


def _func_support_cylinder(
    i_g: Int, d: Vec3, pos: Vec3, quat: Quat, shrink: Bool, geoms_data: Ptr
) -> Vec3:
    """Support function for cylinder geometry.

    Like the capsule, but with flat caps: the support point is on the rim of the cap selected by the sign of d along
    the axis, displaced radially by the radius along d projected onto the cap plane (a sphere/hemisphere cap would
    instead displace along d itself). When d is axial the radial part vanishes and the support is the cap centre.
    """
    var radius = geoms_data.unsafe_load(3 * i_g)
    var halflength = 0.5 * geoms_data.unsafe_load(3 * i_g + 1)
    var axis = qd_transform_by_quat(Vec3(0.0, 0.0, 1.0), quat)
    var endpoint_side = -1.0 if d.dot(axis) < 0.0 else 1.0
    var res = pos + halflength * endpoint_side * axis
    if not shrink:
        var d_radial = d - d.dot(axis) * axis
        var d_radial_norm = d_radial.norm()
        if d_radial_norm > 1e-9:
            res = res + (radius / d_radial_norm) * d_radial
    return res


def _func_support_prism(i_b: Int, d: Vec3, prism: Ptr) -> Tuple[Vec3, Int32]:
    var istart = 3
    if d[2] < 0.0:
        istart = 0

    var ibest = istart
    var best = _prism_corner(prism, istart, i_b).dot(d)
    for i in range(istart + 1, istart + 3):
        var dot_ = _prism_corner(prism, i, i_b).dot(d)
        if dot_ > best:
            ibest = i
            best = dot_

    return (_prism_corner(prism, ibest, i_b), I32(ibest))


def _prism_corner(prism: Ptr, i: Int, i_b: Int) -> Vec3:
    var o = 3 * (i * 8 + i_b)
    return Vec3(prism.unsafe_load(o), prism.unsafe_load(o + 1), prism.unsafe_load(o + 2))


def _func_support_box(
    i_g: Int, d: Vec3, pos: Vec3, quat: Quat, geoms_data: Ptr, vert_start: Int
) -> Tuple[Vec3, Vec3, Int32]:
    var d_box = qd_inv_transform_by_quat(d, quat)

    var v_ = Vec3(
        (-1.0 if d_box[0] < 0.0 else 1.0) * geoms_data.unsafe_load(3 * i_g) * 0.5,
        (-1.0 if d_box[1] < 0.0 else 1.0) * geoms_data.unsafe_load(3 * i_g + 1) * 0.5,
        (-1.0 if d_box[2] < 0.0 else 1.0) * geoms_data.unsafe_load(3 * i_g + 2) * 0.5,
    )
    var vid: Int32 = I32((1 if v_[0] > 0.0 else 0) + (2 if v_[1] > 0.0 else 0) + (4 if v_[2] > 0.0 else 0))
    vid += I32(vert_start)
    var v = qd_transform_by_trans_quat_fast(v_, pos, quat)
    return (v, v_, vid)


def _func_support_world(
    i_g: Int,
    d: Vec3,
    pos: Vec3,
    quat: Quat,
    vert_start: Int,
    vert_end: Int,
    init_pos: Ptr,
    support_res: Int,
    support_cell_start: IPtr,
    support_vid: IPtr,
    support_v: Ptr,
    exhaustive: Bool,
) -> Tuple[Vec3, Vec3, Int32]:
    """Support position for a world direction.

    The vertex comes from the precomputed support table, or from an exhaustive scan of the mesh vertices when
    'exhaustive' is set: the sampled table is approximate and may return a different vertex of a tied flat face,
    while the scan is exact at a cost linear in the vertex count.
    """
    var v = Vec3()
    var v_ = Vec3()
    var vid: Int32 = 0
    if exhaustive:
        var d_mesh = qd_transform_by_quat(d, qd_inv_quat(quat))
        var r = _func_support_mesh_exhaustive(i_g, d_mesh, vert_start, vert_end, init_pos)
        v_ = r[0]
        vid = r[1]
        v = qd_transform_by_trans_quat(v_, pos, quat)
    else:
        var d_mesh = qd_transform_by_quat_fast(d, qd_inv_quat(quat))
        var r = _func_support_mesh(i_g, d_mesh, support_res, support_cell_start, support_vid, support_v)
        v_ = r[0]
        vid = r[1]
        v = qd_transform_by_trans_quat_fast(v_, pos, quat)
    return (v, v_, vid)


def _func_count_supports_mesh(
    i_g: Int, d_mesh: Vec3, support_res: Int, support_cell_start: IPtr, support_v: Ptr
) -> Int32:
    """Count the number of distinct support vertices tied for the maximum dot product in the given direction.

    The support field is a spherical grid indexed by (theta, phi). We look up 4 neighboring cells using
    floor/ceil of the continuous grid coordinates (ii, jj). When ii or jj is an integer (common for
    axis-aligned directions), floor == ceil and multiple iterations map to the same cell. Without
    deduplication the same vertex is counted multiple times, inflating the count and triggering
    unnecessary perturbation in safe_gjk_support.
    """
    var dot_max = -1e20

    var band_start, band_rows = _polar_band(support_res)
    var i_chart, ii, jj = _func_direction_grid_coords(d_mesh, support_res)
    var chart_start = Int32(Int(support_cell_start.unsafe_load(i_g)) + Int(i_chart) * support_res * band_rows)

    # Collect unique cells, deduplicating floor == ceil collisions
    var cell_idx = SIMD[DType.int32, 4](-1, -1, -1, -1)
    var cell_dot = SIMD[DType.float64, 4](0.0, 0.0, 0.0, 0.0)
    var n_unique: Int32 = 0

    for i4 in range(4):
        var i: Int32
        var j: Int32
        if i4 % 2:
            i = Int32(Int(ceil(ii))) % I32(support_res)
        else:
            i = Int32(Int(floor(ii))) % I32(support_res)

        if i4 // 2 > 0:
            j = Int32(clamp(Int(ceil(jj)) - band_start, 0, band_rows - 1))
        else:
            j = Int32(clamp(Int(floor(jj)) - band_start, 0, band_rows - 1))

        var support_idx = Int32(Int(chart_start) + Int(i) * band_rows + Int(j))

        # Skip duplicate cells (from floor == ceil on integer indices).
        var is_dup = False
        for k in range(Int(n_unique)):
            if cell_idx[k] == support_idx:
                is_dup = True
        if is_dup:
            continue

        var o = 3 * Int(support_idx)
        var pos = Vec3(support_v.unsafe_load(o), support_v.unsafe_load(o + 1), support_v.unsafe_load(o + 2))
        cell_dot[Int(n_unique)] = pos.dot(d_mesh)
        cell_idx[Int(n_unique)] = support_idx
        n_unique += 1

    dot_max = -1e20
    var count: Int32 = 0
    for k in range(Int(n_unique)):
        if cell_dot[k] > dot_max:
            count = 1
            dot_max = cell_dot[k]
        elif cell_dot[k] == dot_max:
            count += 1

    return count


def _func_count_supports_world(
    i_g: Int, d: Vec3, quat: Quat, support_res: Int, support_cell_start: IPtr, support_v: Ptr
) -> Int32:
    """Count the number of valid support points for the given world direction.
    Only needs quat since counting doesn't depend on position.
    """
    var d_mesh = qd_transform_by_quat_fast(d, qd_inv_quat(quat))
    return _func_count_supports_mesh(i_g, d_mesh, support_res, support_cell_start, support_v)


def _func_count_supports_box(d: Vec3, quat: Quat) -> Int32:
    """Count the number of valid support points for a box in the given direction.

    Only needs quat since counting doesn't depend on position or geometry-specific data.

    If the direction has 1 zero component, there are 2 possible support points. If the direction has 2 zero
    components, there are 4 possible support points.
    """
    var d_box = qd_inv_transform_by_quat(d, quat)

    var n_zero = (1 if d_box[0] == 0.0 else 0) + (1 if d_box[1] == 0.0 else 0) + (1 if d_box[2] == 0.0 else 0)
    return I32(2 ** n_zero)


# =====================================================================================
# ================== genesis/.../collider/gjk_support.py ===============================
# =====================================================================================


def support_mesh(
    i_g: Int,
    direction: Vec3,
    pos: Vec3,
    quat: Quat,
    prev_vertex_id: IPtr,
    slot: Int,
    init_pos: Ptr,
    vert_start: Int,
    vert_end: Int,
) -> Tuple[Vec3, Int32]:
    """Find the support point on a mesh in the given direction."""
    var d_mesh = qd_transform_by_quat(direction, qd_inv_quat(quat))

    # Exhaustively search for the vertex with maximum dot product
    var fmax = -FLOAT_MAX
    var imax = 0

    # Use the previous maximum vertex if it is within the current range
    var prev_imax = Int(prev_vertex_id.unsafe_load(slot))
    if (prev_imax >= vert_start) and (prev_imax < vert_end):
        var o = 3 * prev_imax
        var pos_local = Vec3(init_pos.unsafe_load(o), init_pos.unsafe_load(o + 1), init_pos.unsafe_load(o + 2))
        fmax = d_mesh.dot(pos_local)
        imax = prev_imax

    for i in range(vert_start, vert_end):
        var o = 3 * i
        var pos_local = Vec3(init_pos.unsafe_load(o), init_pos.unsafe_load(o + 1), init_pos.unsafe_load(o + 2))
        var vdot = d_mesh.dot(pos_local)
        if vdot > fmax:
            fmax = vdot
            imax = i

    var ov = 3 * imax
    var v = Vec3(init_pos.unsafe_load(ov), init_pos.unsafe_load(ov + 1), init_pos.unsafe_load(ov + 2))
    var vid = I32(imax)

    prev_vertex_id.unsafe_offset(slot)[] = vid

    var v_world = qd_transform_by_trans_quat(v, pos, quat)
    return (v_world, vid)


def support_driver(
    i_g: Int,
    direction: Vec3,
    pos: Vec3,
    quat: Quat,
    shrink_sphere: Bool,
    geom_type: Int32,
    geoms_data: Ptr,
    vert_start: Int,
    vert_end: Int,
    init_pos: Ptr,
    support_res: Int,
    support_cell_start: IPtr,
    support_vid: IPtr,
    support_v: Ptr,
    prism: Ptr,
    i_b: Int,
    has_terrain: Bool,
    enable_mujoco_compatibility: Bool,
    prev_vertex_id: IPtr,
    slot: Int,
) -> Tuple[Vec3, Vec3, Int32]:
    """@ shrink_sphere: If True, use point and line support for sphere and capsule."""
    var v = Vec3()
    var v_ = Vec3()
    var vid: Int32 = -1

    if geom_type == GEOM_TYPE_SPHERE:
        var r = _func_support_sphere(i_g, direction, pos, quat, shrink_sphere, geoms_data)
        v = r[0]
        v_ = r[1]
        vid = r[2]
    elif geom_type == GEOM_TYPE_ELLIPSOID:
        v = _func_support_ellipsoid(i_g, direction, pos, quat, geoms_data)
    elif geom_type == GEOM_TYPE_CAPSULE:
        v = _func_support_capsule(i_g, direction, pos, quat, shrink_sphere, geoms_data)
    elif geom_type == GEOM_TYPE_CYLINDER:
        v = _func_support_cylinder(i_g, direction, pos, quat, shrink_sphere, geoms_data)
    elif geom_type == GEOM_TYPE_BOX:
        var r = _func_support_box(i_g, direction, pos, quat, geoms_data, vert_start)
        v = r[0]
        v_ = r[1]
        vid = r[2]
    elif geom_type == GEOM_TYPE_TERRAIN:
        if has_terrain:
            var r = _func_support_prism(i_b, direction, prism)
            v = r[0]
            vid = r[1]
    elif geom_type == GEOM_TYPE_MESH and enable_mujoco_compatibility:
        # If mujoco-compatible, do exhaustive search for the vertex
        var r = support_mesh(
            i_g, direction, pos, quat, prev_vertex_id, slot, init_pos, vert_start, vert_end
        )
        v = r[0]
        vid = r[1]
    else:
        var r = _func_support_world(
            i_g,
            direction,
            pos,
            quat,
            vert_start,
            vert_end,
            init_pos,
            support_res,
            support_cell_start,
            support_vid,
            support_v,
            enable_mujoco_compatibility,
        )
        v = r[0]
        v_ = r[1]
        vid = r[2]
    return (v, v_, vid)


def func_support(
    i_ga: Int,
    i_gb: Int,
    dir: Vec3,
    pos_a: Vec3,
    quat_a: Quat,
    pos_b: Vec3,
    quat_b: Quat,
    shrink_sphere: Bool,
    geom_type: IPtr,
    geoms_data: Ptr,
    geoms_vert_start: IPtr,
    geoms_vert_end: IPtr,
    init_pos: Ptr,
    support_res: Int,
    support_cell_start: IPtr,
    support_vid: IPtr,
    support_v: Ptr,
    prev_vertex_id: IPtr,
    prism: Ptr,
    i_b: Int,
    has_terrain: Bool,
    enable_mujoco_compatibility: Bool,
) -> Tuple[Vec3, Vec3, Vec3, Vec3, Int32, Int32, Vec3]:
    """Find support points on the two objects using [dir].

    Parameters:
    ----------
    dir: gs.qd_vec3
        The direction in which to find the support points, from [ga] (obj 1) to [gb] (obj 2).
    """
    var support_point_obj1 = Vec3(0.0, 0.0, 0.0)
    var support_point_obj2 = Vec3(0.0, 0.0, 0.0)
    var support_point_localpos1 = Vec3(0.0, 0.0, 0.0)
    var support_point_localpos2 = Vec3(0.0, 0.0, 0.0)
    var support_point_id_obj1: Int32 = -1
    var support_point_id_obj2: Int32 = -1

    for i in range(2):
        var d = dir if i == 0 else -dir
        var i_g = i_ga if i == 0 else i_gb
        var pos = pos_a if i == 0 else pos_b
        var quat = quat_a if i == 0 else quat_b

        var r = support_driver(
            i_g,
            d,
            pos,
            quat,
            shrink_sphere,
            geom_type.unsafe_load(i_g),
            geoms_data,
            Int(geoms_vert_start.unsafe_load(i_g)),
            Int(geoms_vert_end.unsafe_load(i_g)),
            init_pos,
            support_res,
            support_cell_start,
            support_vid,
            support_v,
            prism,
            i_b,
            has_terrain,
            enable_mujoco_compatibility,
            prev_vertex_id,
            i,
        )
        var sp = r[0]
        var sp_ = r[1]
        var si = r[2]

        if i == 0:
            support_point_obj1 = sp
            support_point_id_obj1 = si
            support_point_localpos1 = sp_
        else:
            support_point_obj2 = sp
            support_point_id_obj2 = si
            support_point_localpos2 = sp_

    var support_point_minkowski = support_point_obj1 - support_point_obj2

    return (
        support_point_obj1,
        support_point_obj2,
        support_point_localpos1,
        support_point_localpos2,
        support_point_id_obj1,
        support_point_id_obj2,
        support_point_minkowski,
    )


# =====================================================================================
# ================== genesis/.../collider/gjk_utils.py ================================
# =====================================================================================


def func_ray_triangle_intersection(ray_v1: Vec3, ray_v2: Vec3, tri_v1: Vec3, tri_v2: Vec3, tri_v3: Vec3) -> Int32:
    """Check if the ray intersects the triangle.

    Returns
    -------
    int
        True if the ray intersects the triangle, otherwise False.
    """
    var ray = ray_v2 - ray_v1

    # Signed volumes of the tetrahedrons formed by the ray and triangle edges
    var vols = Vec3(0.0, 0.0, 0.0)
    for i in range(3):
        var v1 = Vec3(0.0, 0.0, 0.0)
        var v2 = Vec3(0.0, 0.0, 0.0)
        if i == 0:
            v1 = tri_v1 - ray_v1
            v2 = tri_v2 - ray_v1
        if i == 1:
            v1 = tri_v2 - ray_v1
            v2 = tri_v3 - ray_v1
        elif i == 2:
            v1 = tri_v3 - ray_v1
            v2 = tri_v1 - ray_v1
        vols[i] = func_det3(v1, v2, ray)

    var all_ge = True
    var all_le = True
    for i in range(3):
        if vols[i] < 0.0:
            all_ge = False
        if vols[i] > 0.0:
            all_le = False
    return 1 if (all_ge or all_le) else 0


def func_triangle_affine_coords(point: Vec3, tri_v1: Vec3, tri_v2: Vec3, tri_v3: Vec3) -> Vec3:
    """Compute the affine coordinates of the point with respect to the triangle."""
    # Compute minors of the triangle vertices
    var ms = Vec3(0.0, 0.0, 0.0)
    for i in range(3):
        var i1 = (i + 1) % 3
        var i2 = (i + 2) % 3
        if i == 1:
            var t = i1
            i1 = i2
            i2 = t

        ms[i] = (
            tri_v2[i1] * tri_v3[i2]
            - tri_v2[i2] * tri_v3[i1]
            - tri_v1[i1] * tri_v3[i2]
            + tri_v1[i2] * tri_v3[i1]
            + tri_v1[i1] * tri_v2[i2]
            - tri_v1[i2] * tri_v2[i1]
        )

    # Exclude one of the axes with the largest projection using the minors of the above linear system. The scan is
    # unrolled so that every read of [ms] carries a compile-time index.
    var m_max = 0.0
    var i_x = 0
    var i_y = 0
    var absms = Vec3(abs(ms[0]), abs(ms[1]), abs(ms[2]))
    var is_found = False
    for i in range(3):
        var i_x_i = (i + 1) % 3
        var i_y_i = (i + 2) % 3
        if i == 1:
            var t = i_x_i
            i_x_i = i_y_i
            i_y_i = t
        if (
            not is_found
            and absms[i] >= absms[(i + 1) % 3]
            and absms[i] >= absms[(i + 2) % 3]
        ):
            # Remove the i-th row
            is_found = True
            m_max = ms[i]
            i_x = i_x_i
            i_y = i_y_i

    var cs = Vec3(0.0, 0.0, 0.0)
    for i in range(3):
        var tv1 = tri_v2
        var tv2 = tri_v3
        if i == 1:
            tv1 = tri_v3
            tv2 = tri_v1
        elif i == 2:
            tv1 = tri_v1
            tv2 = tri_v2

        # Corresponds to the signed area of 2-simplex (triangle): (point, tv1, tv2)
        cs[i] = (
            point[i_x] * tv1[i_y]
            + point[i_y] * tv2[i_x]
            + tv1[i_x] * tv2[i_y]
            - point[i_x] * tv2[i_y]
            - point[i_y] * tv1[i_x]
            - tv2[i_x] * tv1[i_y]
        )

    # Affine coordinates are computed as: [ l1, l2, l3 ] = [ C1 / m_max, C2 / m_max, C3 / m_max ]
    return cs / m_max


def func_point_triangle_intersection(point: Vec3, tri_v1: Vec3, tri_v2: Vec3, tri_v3: Vec3) -> Int32:
    """Check if the point is inside the triangle."""
    var is_inside = False
    # Compute the affine coordinates of the point with respect to the triangle
    var _lambda = func_triangle_affine_coords(point, tri_v1, tri_v2, tri_v3)

    # If any of the affine coordinates is negative, the point is outside the triangle
    var nonneg = True
    for i in range(3):
        if _lambda[i] < 0.0:
            nonneg = False
    if nonneg:
        # Check if the point predicted by the affine coordinates is equal to the point itself
        var pred = tri_v1 * _lambda[0] + tri_v2 * _lambda[1] + tri_v3 * _lambda[2]
        var diff = pred - point
        is_inside = diff.norm_sqr() < FLOAT_MIN_SQ

    return 1 if is_inside else 0


def func_point_plane_same_side(point: Vec3, plane_v1: Vec3, plane_v2: Vec3, plane_v3: Vec3) -> Int32:
    """Check if the point is on the same side of the plane as the origin."""
    # Compute the normal of the plane
    var edge1 = plane_v2 - plane_v1
    var edge2 = plane_v3 - plane_v1
    var normal = edge1.cross(edge2)

    var diff1 = point - plane_v1
    var dot1 = normal.dot(diff1)

    # origin - plane_v1
    var diff2 = -plane_v1
    var dot2 = normal.dot(diff2)

    return RETURN_CODE_SUCCESS if dot1 * dot2 > 0.0 else RETURN_CODE_FAIL


def func_origin_tetra_intersection(tet_v1: Vec3, tet_v2: Vec3, tet_v3: Vec3, tet_v4: Vec3) -> Int32:
    """Check if the origin is inside the tetrahedron."""
    var flag = RETURN_CODE_SUCCESS
    for i in range(4):
        var v1 = tet_v1
        var v2 = tet_v2
        var v3 = tet_v3
        var v4 = tet_v4
        if i == 1:
            v1 = tet_v2
            v2 = tet_v3
            v3 = tet_v4
            v4 = tet_v1
        elif i == 2:
            v1 = tet_v3
            v2 = tet_v4
            v3 = tet_v1
            v4 = tet_v2
        elif i == 3:
            v1 = tet_v4
            v2 = tet_v1
            v3 = tet_v2
            v4 = tet_v3
        flag = func_point_plane_same_side(v1, v2, v3, v4)
        if flag == RETURN_CODE_FAIL:
            break
    return flag


def func_project_origin_to_plane(v1: Vec3, v2: Vec3, v3: Vec3) -> Tuple[Vec3, Int32]:
    """Project the origin onto the plane defined by the simplex vertices."""
    var point = Vec3(0.0, 0.0, 0.0)
    var flag = RETURN_CODE_SUCCESS

    var d21 = v2 - v1
    var d31 = v3 - v1
    var d32 = v3 - v2

    for i in range(3):
        var n: Vec3
        var v: Vec3
        if i == 0:
            # Normal = (v1 - v2) x (v3 - v2)
            n = d32.cross(d21)
            v = v2
        elif i == 1:
            # Normal = (v2 - v1) x (v3 - v1)
            n = d21.cross(d31)
            v = v1
        else:
            # Normal = (v1 - v3) x (v2 - v3)
            n = d31.cross(d32)
            v = v3
        var nv = n.dot(v)
        var nn = n.norm_sqr()
        if nn == 0.0:
            # Zero normal, cannot project.
            flag = RETURN_CODE_FAIL
            break
        elif nn > FLOAT_MIN:
            point = n * (nv / nn)
            flag = RETURN_CODE_SUCCESS
            break

        # Last fallback if no valid normal was found
        if i == 2:
            # If the normal is still unreliable, cannot project.
            if nn < FLOAT_MIN:
                flag = RETURN_CODE_FAIL
            else:
                point = n * (nv / nn)
                flag = RETURN_CODE_SUCCESS

    return (point, flag)


# =====================================================================================
# ================== genesis/.../collider/contact.py =================================
# =====================================================================================


def _aabb_corner(geoms_init_AABB: Ptr, i_g: Int, k: Int) -> Vec3:
    # geoms_init_AABB is V_VEC(3) with shape (n_geoms, 8): the 8 corners of the
    # geom's rest-pose AABB, corner 0 = lower, corner 7 = upper.
    var o = 3 * (8 * i_g + k)
    return Vec3(geoms_init_AABB.unsafe_load(o), geoms_init_AABB.unsafe_load(o + 1), geoms_init_AABB.unsafe_load(o + 2))


def func_compute_geom_rbound(i_g: Int, geom_type: Int32, geoms_data: Ptr, geoms_init_AABB: Ptr) -> Float64:
    """Compute the bounding sphere radius for a geom, matching MuJoCo's geom_rbound."""
    var rbound = 0.0
    if geom_type == GEOM_TYPE_SPHERE:
        rbound = geoms_data.unsafe_load(3 * i_g)
    elif geom_type == GEOM_TYPE_CAPSULE:
        # radius + half_length (MuJoCo stores size as [radius, half_length])
        # Genesis stores data as [radius, full_length], so half_length = 0.5 * data[1]
        rbound = geoms_data.unsafe_load(3 * i_g) + 0.5 * geoms_data.unsafe_load(3 * i_g + 1)
    elif geom_type == GEOM_TYPE_ELLIPSOID:
        rbound = max(geoms_data.unsafe_load(3 * i_g), max(geoms_data.unsafe_load(3 * i_g + 1), geoms_data.unsafe_load(3 * i_g + 2)))
    elif geom_type == GEOM_TYPE_BOX:
        var d0 = geoms_data.unsafe_load(3 * i_g)
        var d1 = geoms_data.unsafe_load(3 * i_g + 1)
        var d2 = geoms_data.unsafe_load(3 * i_g + 2)
        rbound = sqrt(d0 * d0 + d1 * d1 + d2 * d2)
    else:
        # For mesh and other types, approximate as half AABB diagonal
        rbound = 0.5 * (_aabb_corner(geoms_init_AABB, i_g, 7) - _aabb_corner(geoms_init_AABB, i_g, 0)).norm()
    return rbound


def func_compute_geom_pair_scale(
    i_ga: Int, i_gb: Int, geom_type: Int32, geoms_init_AABB: Ptr
) -> Float64:
    # Intrinsic length scale of a geom pair: half the smaller geom's world-aligned bounding-box diagonal. The
    # original (rest-pose) AABB is used so the scale is a constant independent of the current orientation, which
    # makes sense since the size of the geometries is an intrinsic property. Multiply by a relative tolerance to
    # turn it into an absolute one.
    var aabb_size_b = (_aabb_corner(geoms_init_AABB, i_gb, 7) - _aabb_corner(geoms_init_AABB, i_gb, 0)).norm()
    var aabb_size = aabb_size_b
    if geom_type != GEOM_TYPE_PLANE:
        var aabb_size_a = (_aabb_corner(geoms_init_AABB, i_ga, 7) - _aabb_corner(geoms_init_AABB, i_ga, 0)).norm()
        aabb_size = min(aabb_size_a, aabb_size_b)

    return 0.5 * aabb_size


def func_compute_geom_pair_scale_mj(
    i_ga: Int, i_gb: Int, geom_type: IPtr, geoms_data: Ptr, geoms_init_AABB: Ptr
) -> Float64:
    """Geom-pair length scale matching MuJoCo's formula: min(rbound_g1, rbound_g2). Multiply by a relative tolerance
    to recover MuJoCo's absolute tolerance."""
    var rbound_a = func_compute_geom_rbound(i_ga, geom_type.unsafe_load(i_ga), geoms_data, geoms_init_AABB)
    var rbound_b = func_compute_geom_rbound(i_gb, geom_type.unsafe_load(i_gb), geoms_data, geoms_init_AABB)
    return min(rbound_a, rbound_b)


def func_compute_mc_tolerance(
    i_ga: Int,
    i_gb: Int,
    geom_type: IPtr,
    geoms_data: Ptr,
    geoms_init_AABB: Ptr,
    mc_tolerance: Float64,
    enable_mujoco_compatibility: Bool,
) -> Float64:
    """Absolute multi-contact acceptance tolerance of a geom pair.

    Shared by every convex narrowphase arm so the accepted contact set stays backend-independent. The relative
    tolerance scales with the reference engine's pair scale under MuJoCo compatibility and with the intrinsic pair
    scale otherwise."""
    var scale = func_compute_geom_pair_scale(i_ga, i_gb, geom_type.unsafe_load(i_ga), geoms_init_AABB)
    if enable_mujoco_compatibility:
        scale = func_compute_geom_pair_scale_mj(i_ga, i_gb, geom_type, geoms_data, geoms_init_AABB)
    return mc_tolerance * scale


def func_contact_orthogonals(normal: Vec3) -> Tuple[Vec3, Vec3]:
    """MuJoCo-compatible branch of `func_contact_orthogonals`: choose between world axes Y or Z
    to avoid colinearity issue."""
    var axis_0 = Vec3()
    var axis_1: Vec3

    # Choose between world axes Y or Z to avoid colinearity issue
    if abs(normal[1]) < 0.5:
        axis_0[1] = 1.0
    else:
        axis_0[2] = 1.0

    # Project axis on orthogonal plane to contact normal
    axis_0 = (axis_0 - normal.dot(axis_0) * normal).normalized()

    # Complete orthonormal frame (matching MuJoCo's mju_makeFrame)
    axis_1 = normal.cross(axis_0)
    axis_0 = axis_1.cross(normal)

    return (axis_0, axis_1)


def func_rotate_frame(
    pos: Vec3, quat: Quat, contact_pos: Vec3, qrot: Quat
) -> Tuple[Vec3, Quat]:
    """
    Instead of modifying geoms_state in place, this function takes thread-local
    pos/quat and returns the updated values.
    """
    var new_quat = qd_transform_quat_by_quat(quat, qrot)

    var rel = contact_pos - pos
    var vec = qd_transform_by_quat(rel, qrot)
    vec = vec - rel
    var new_pos = pos - vec

    return (new_pos, new_quat)


def func_contact_order_key(pos: Vec3) -> Float64:
    """Order a contact position along one generic direction, as a single scalar.

    Comparing components in turn tests each for equality, and in a frame attached to the geometry the contacts of one
    patch share components exactly - a box face puts two corners at the same local x - so the ordering follows the
    rounding of a mathematically tied quantity. Projecting on a direction no face of a box or regular prism is parallel
    to separates the points of a patch by a margin of their own spacing. The weights are successive powers of the
    golden ratio, as far from any rational direction as a pair of weights gets.
    """
    return pos[0] + 1.618033988749895 * pos[1] + 2.618033988749895 * pos[2]


# =====================================================================================
# ================== genesis/engine/materials/MPM/base.py ==============================
# =====================================================================================


def resolve_lame(E: Float64, nu: Float64) -> Tuple[Float64, Float64]:
    """Base.model_post_init: resolve the first and second Lame parameters from E and nu."""
    var mu = E / (2.0 * (1.0 + nu))
    var lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return (lam, mu)


def _update_F_S_Jp_noop(J: Float64, F_tmp: Mat3, U: Mat3, S: Mat3, V: Mat3, Jp: Float64) -> Tuple[Mat3, Mat3, Float64]:
    var F_new = F_tmp
    var S_new = S
    var Jp_new = Jp
    return (F_new, S_new, Jp_new)


def _update_stress_default(
    mu: Float64, lam: Float64, U: Mat3, S: Mat3, V: Mat3, F_tmp: Mat3, F_new: Mat3, J: Float64
) -> Mat3:
    var stress = 2.0 * mu * (F_new - U @ V.transpose()) @ F_new.transpose() + Mat3.identity() * (lam * J * (J - 1.0))
    return stress


# =====================================================================================
# ================== genesis/engine/materials/MPM/elastic.py ===========================
# =====================================================================================


def _update_F_S_Jp_elastic(J: Float64, F_tmp: Mat3, U: Mat3, S: Mat3, V: Mat3, Jp: Float64) -> Tuple[Mat3, Mat3, Float64]:
    var F_new = F_tmp
    var S_new = S
    var Jp_new = Jp
    return (F_new, S_new, Jp_new)


def _update_stress_corotation(
    mu: Float64, lam: Float64, U: Mat3, S: Mat3, V: Mat3, F_tmp: Mat3, F_new: Mat3, J: Float64
) -> Mat3:
    var stress = 2.0 * mu * (F_new - U @ V.transpose()) @ F_new.transpose() + Mat3.identity() * (lam * J * (J - 1.0))
    return stress


def _update_stress_neohooken(
    mu: Float64, lam: Float64, U: Mat3, S: Mat3, V: Mat3, F_tmp: Mat3, F_new: Mat3, J: Float64
) -> Mat3:
    var stress = mu * (F_tmp @ F_tmp.transpose()) + Mat3.identity() * (lam * log(J) - mu)
    return stress


# =====================================================================================
# ============ genesis/engine/materials/MPM/elasto_plastic.py ==========================
# =====================================================================================


def _update_F_S_Jp_elasto_plastic(
    mu: Float64, use_von_mises: Bool, von_mises_yield_stress: Float64, yield_lower: Float64, yield_higher: Float64,
    J: Float64, F_tmp: Mat3, U: Mat3, S: Mat3, V: Mat3, Jp: Float64,
) -> Tuple[Mat3, Mat3, Float64]:
    var F_new: Mat3
    var S_new: Mat3
    if use_von_mises:
        S_new = Mat3(_max3(S.r0, 0.05), _max3(S.r1, 0.05), _max3(S.r2, 0.05))  # to prevent NaN
        var epsilon = Vec3(log(S_new.r0[0]), log(S_new.r1[1]), log(S_new.r2[2]))
        var eps_sum = epsilon[0] + epsilon[1] + epsilon[2]
        var epsilon_hat = epsilon - (eps_sum / 3.0)
        var epsilon_hat_norm = epsilon_hat.norm()
        var delta_gamma = epsilon_hat_norm - von_mises_yield_stress / (2.0 * mu)

        if delta_gamma > 0.0:  # Yields
            epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
            S_new = _diag(Vec3(exp(epsilon[0]), exp(epsilon[1]), exp(epsilon[2])))
            F_new = U @ S_new @ V.transpose()
        else:
            F_new = F_tmp

    else:
        S_new = _diag(_clamp_diag(S, 1.0 - yield_lower, 1.0 + yield_higher))
        F_new = U @ S_new @ V.transpose()

    var Jp_new = Jp
    return (F_new, S_new, Jp_new)


# =====================================================================================
# ================== genesis/engine/materials/MPM/liquid.py ============================
# =====================================================================================


def _update_F_S_Jp_liquid(J: Float64, F_tmp: Mat3, U: Mat3, S: Mat3, V: Mat3, Jp: Float64) -> Tuple[Mat3, Mat3, Float64]:
    var F_new = Mat3.identity() * (J ** (1.0 / 3.0))
    var S_new = S
    var Jp_new = Jp
    return (F_new, S_new, Jp_new)


def _update_stress_liquid(
    mu: Float64, lam: Float64, U: Mat3, S: Mat3, V: Mat3, F_tmp: Mat3, F_new: Mat3, J: Float64
) -> Mat3:
    var stress = 2.0 * mu * (F_tmp - U @ V.transpose()) @ F_tmp.transpose() + Mat3.identity() * (lam * J * (J - 1.0))
    return stress


# =====================================================================================
# ================== genesis/engine/materials/MPM/sand.py ==============================
# =====================================================================================


def sand_alpha(friction_angle: Float64) -> Float64:
    """Sand.model_post_init: pressure-dependent plasticity coefficient."""
    var sin_phi = sin(friction_angle * 3.141592653589793 / 180.0)
    return sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)


def _sand_projection(lam: Float64, mu: Float64, alpha: Float64, S: Mat3, Jp: Float64) -> Tuple[Mat3, Float64]:
    var s = Vec3(S.r0[0], S.r1[1], S.r2[2])
    var epsilon = Vec3(
        log(max(abs(s[0]), 1e-4)), log(max(abs(s[1]), 1e-4)), log(max(abs(s[2]), 1e-4))
    )
    var S_out = _diag(Vec3(1.0, 1.0, 1.0))
    var tr = (epsilon[0] + epsilon[1] + epsilon[2]) + Jp
    var epsilon_hat = epsilon - tr / 3.0
    var epsilon_hat_norm = epsilon_hat.norm()

    var Jp_new: Float64
    if tr >= 0.0:
        Jp_new = tr
    else:
        Jp_new = 0.0
        var delta_gamma = epsilon_hat_norm + (3.0 * lam + 2.0 * mu) / (2.0 * mu) * tr * alpha
        var k = max(0.0, delta_gamma) / epsilon_hat_norm
        S_out = _diag(
            Vec3(
                exp(epsilon[0] - k * epsilon_hat[0]),
                exp(epsilon[1] - k * epsilon_hat[1]),
                exp(epsilon[2] - k * epsilon_hat[2]),
            )
        )

    return (S_out, Jp_new)


def _update_F_S_Jp_sand(
    lam: Float64, mu: Float64, alpha: Float64, J: Float64, F_tmp: Mat3, U: Mat3, S: Mat3, V: Mat3, Jp: Float64
) -> Tuple[Mat3, Mat3, Float64]:
    var r = _sand_projection(lam, mu, alpha, S, Jp)
    var S_new = r[0]
    var Jp_new = r[1]
    var F_new = U @ S_new @ V.transpose()
    return (F_new, S_new, Jp_new)


def _update_stress_sand(
    mu: Float64, lam: Float64, U: Mat3, S: Mat3, V: Mat3, F_tmp: Mat3, F_new: Mat3, J: Float64
) -> Mat3:
    var s = Vec3(S.r0[0], S.r1[1], S.r2[2])
    var l = Vec3(log(s[0]), log(s[1]), log(s[2]))
    var log_S_sum = (l[0] + l[1]) + l[2]
    var inv = Vec3(1.0 / s[0], 1.0 / s[1], 1.0 / s[2])
    var center = _diag(
        Vec3(
            2.0 * mu * l[0] * inv[0] + lam * log_S_sum * inv[0],
            2.0 * mu * l[1] * inv[1] + lam * log_S_sum * inv[1],
            2.0 * mu * l[2] * inv[2] + lam * log_S_sum * inv[2],
        )
    )
    var stress = U @ center @ V.transpose() @ F_new.transpose()
    return stress


# =====================================================================================
# ================== genesis/engine/materials/MPM/snow.py ==============================
# =====================================================================================


def _update_F_S_Jp_snow(
    yield_lower: Float64, yield_higher: Float64, J: Float64, F_tmp: Mat3, U: Mat3, S: Mat3, V: Mat3, Jp: Float64
) -> Tuple[Mat3, Mat3, Float64]:
    var s_new = _clamp_diag(S, 1.0 - yield_lower, 1.0 + yield_higher)
    var S_new = _diag(s_new)
    var Jp_new = Jp * (S.r0[0] / s_new[0]) * (S.r1[1] / s_new[1]) * (S.r2[2] / s_new[2])
    var F_new = U @ S_new @ V.transpose()
    return (F_new, S_new, Jp_new)


def _update_stress_snow(
    mu: Float64, lam: Float64, yield_lower: Float64, yield_higher: Float64,
    U: Mat3, S: Mat3, V: Mat3, F_tmp: Mat3, F_new: Mat3, J: Float64, Jp: Float64,
) -> Mat3:
    # Hardening coefficient: material harder when compressed
    var h = exp(10.0 * (1.0 - Jp))
    var mu_h = mu * h
    var lam_h = lam * h

    var r = U @ V.transpose()
    var stress = 2.0 * mu_h * (F_new - r) @ F_new.transpose() + Mat3.identity() * (lam_h * J * (J - 1.0))

    return stress


# =====================================================================================
# ================== genesis/engine/solvers/mpm_solver.py ==============================
# =====================================================================================


def stencil_range() -> Int:
    return 3


def compute_F_tmp(C: Mat3, F: Mat3, substep_dt: Float64) -> Mat3:
    """MPMSolver.compute_F_tmp for one particle."""
    return (Mat3.identity() + substep_dt * C) @ F


def _quadratic_weights(fx: Float64) -> Vec3:
    var w0 = 0.5 * (1.5 - fx) ** 2
    var w1 = 0.75 - (fx - 1.0) ** 2
    var w2 = 0.5 * (fx - 0.5) ** 2
    return Vec3(w0, w1, w2)


def _stencil_offset(k: Int) -> Vec3:
    # ndrange(3, 3, 3), flattened; upstream reads the offsets as integers and casts.
    return Vec3(Float64(k % 3), Float64((k // 3) % 3), Float64(k // 9))


def _stencil_weight(off: Vec3, w: Vec3, w1: Vec3, w2: Vec3) -> Float64:
    # Upstream's `for d in qd.static(range(3))`, written out: the product order is the same.
    return w[Int(off[0])] * w1[Int(off[1])] * w2[Int(off[2])]


def _cell_index(base: Vec3, offset: Vec3, grid_offset: Vec3, grid_res: Vec3) -> Int:
    var i = Int(base[0] + offset[0] - grid_offset[0])
    var j = Int(base[1] + offset[1] - grid_offset[1])
    var k = Int(base[2] + offset[2] - grid_offset[2])
    return (i * Int(grid_res[1]) + j) * Int(grid_res[2]) + k


def p2g(
    pos: Vec3,
    vel: Vec3,
    C: Mat3,
    F_tmp: Mat3,
    U: Mat3,
    S: Mat3,
    V: Mat3,
    Jp: Float64,
    F: Mat3,
    active: Bool,
    mass: Float64,
    free: Bool,
    mu: Float64,
    lam: Float64,
    yield_lower: Float64,
    yield_higher: Float64,
    use_von_mises: Bool,
    von_mises_yield_stress: Float64,
    friction_angle: Float64,
    model: Int32,
    needs_svd: Bool,
    substep_dt: Float64,
    dx: Float64,
    inv_dx: Float64,
    particle_volume: Float64,
    grid_offset: Vec3,
    grid_res: Vec3,
    grid_vel_in: Ptr,
    grid_mass: Ptr,
) -> Tuple[Mat3, Float64]:
    """MPMSolver.p2g for one particle: material update, stress, and the 27-cell scatter.

    The CPIC coupler and the sparse-reset dirty-cell bookkeeping of the upstream kernel are not
    ported; see the README. `model` selects the constitutive model the way upstream's
    `qd.static` dispatch does, and `needs_svd` is expressed by reading J from S rather than
    from F_tmp.
    """
    # A. update F (deformation gradient), S (Sigma from SVD(F), essentially represents volume) and Jp
    # (volume compression ratio) based on material type.
    var J: Float64
    if needs_svd:
        J = S.determinant()
    else:
        J = F_tmp.determinant()

    # Elastic / corotational and neohookean share the elastic F update.
    var r = _update_F_S_Jp_elastic(J, F_tmp, U, S, V, Jp)
    if model == 2:  # elasto_plastic
        r = _update_F_S_Jp_elasto_plastic(
            mu, use_von_mises, von_mises_yield_stress, yield_lower, yield_higher, J, F_tmp, U, S, V, Jp
        )
    elif model == 3:  # liquid
        r = _update_F_S_Jp_liquid(J, F_tmp, U, S, V, Jp)
    elif model == 4:  # sand
        r = _update_F_S_Jp_sand(lam, mu, sand_alpha(friction_angle), J, F_tmp, U, S, V, Jp)
    elif model == 5:  # snow
        r = _update_F_S_Jp_snow(yield_lower, yield_higher, J, F_tmp, U, S, V, Jp)
    var F_new = r[0]
    var S_new = r[1]
    var Jp_new = r[2]

    # B. compute stress
    # model 0 (elastic corotation) and model 2 (elasto-plastic) both take the
    # corotation stress, whose body is upstream's `_update_stress_default`.
    var stress = _update_stress_corotation(mu, lam, U, S_new, V, F_tmp, F_new, J)
    if model == 1:  # elastic, neohooken
        stress = _update_stress_neohooken(mu, lam, U, S_new, V, F_tmp, F_new, J)
    elif model == 2:  # elasto_plastic uses the default corotation stress
        stress = _update_stress_default(mu, lam, U, S_new, V, F_tmp, F_new, J)
    elif model == 3:  # liquid
        stress = _update_stress_liquid(mu, lam, U, S_new, V, F_tmp, F_new, J)
    elif model == 4:  # sand
        stress = _update_stress_sand(mu, lam, U, S, V, F_tmp, F_new, J)
    elif model == 5:  # snow
        stress = _update_stress_snow(mu, lam, yield_lower, yield_higher, U, S, V, F_tmp, F_new, J, Jp)
    stress = (-substep_dt * particle_volume * 4.0 * inv_dx * inv_dx) * stress
    var affine = stress + mass * C

    # C. project onto grid
    var base = Vec3(floor(pos[0] * inv_dx - 0.5), floor(pos[1] * inv_dx - 0.5), floor(pos[2] * inv_dx - 0.5))
    var fx = Vec3(pos[0] * inv_dx - base[0], pos[1] * inv_dx - base[1], pos[2] * inv_dx - base[2])
    var w = _quadratic_weights(fx[0])
    var w1 = _quadratic_weights(fx[1])
    var w2 = _quadratic_weights(fx[2])

    for offset_idx in range(27):
        var off = _stencil_offset(offset_idx)
        var dpos = (off - fx) * dx
        var weight = _stencil_weight(off, w, w1, w2)

        var cell = _cell_index(base, off, grid_offset, grid_res)
        if active:
            var contrib = weight * (mass * vel + affine @ dpos)
            var o = 3 * cell
            grid_vel_in.unsafe_offset(o)[] = grid_vel_in.unsafe_load(o) + contrib[0]
            grid_vel_in.unsafe_offset(o + 1)[] = grid_vel_in.unsafe_load(o + 1) + contrib[1]
            grid_vel_in.unsafe_offset(o + 2)[] = grid_vel_in.unsafe_load(o + 2) + contrib[2]
            var mass_contrib = weight * mass
            grid_mass.unsafe_offset(cell)[] = grid_mass.unsafe_load(cell) + mass_contrib
        if not free:  # non-free particles behave as boundary conditions
            var oc = 3 * _cell_index(base, off, grid_offset, grid_res)
            grid_vel_in.unsafe_offset(oc)[] = 0.0
            grid_vel_in.unsafe_offset(oc + 1)[] = 0.0
            grid_vel_in.unsafe_offset(oc + 2)[] = 0.0

    return (F_new, Jp_new)


def g2p(
    pos: Vec3,
    grid_vel_out: Ptr,
    lower: Vec3,
    upper: Vec3,
    substep_dt: Float64,
    inv_dx: Float64,
    grid_offset: Vec3,
    grid_res: Vec3,
) -> Tuple[Vec3, Vec3, Mat3]:
    """MPMSolver.g2p for one particle: the 27-cell gather, the C update and the boundary clamp.

    `dpos` here is `(offset - fx)`, not `(offset - fx) * dx` as in p2g; the `4 * inv_dx`
    factor compensates, and the two must not be "tidied up" to match.
    """
    var base = Vec3(floor(pos[0] * inv_dx - 0.5), floor(pos[1] * inv_dx - 0.5), floor(pos[2] * inv_dx - 0.5))
    var fx = Vec3(pos[0] * inv_dx - base[0], pos[1] * inv_dx - base[1], pos[2] * inv_dx - base[2])
    var w = _quadratic_weights(fx[0])
    var w1 = _quadratic_weights(fx[1])
    var w2 = _quadratic_weights(fx[2])
    var new_vel = Vec3()
    var new_C = Mat3()
    for offset_idx in range(27):
        var off = _stencil_offset(offset_idx)
        var dpos = off - fx
        var cell = _cell_index(base, off, grid_offset, grid_res)
        var o = 3 * cell
        var grid_vel = Vec3(grid_vel_out.unsafe_load(o), grid_vel_out.unsafe_load(o + 1), grid_vel_out.unsafe_load(o + 2))
        var weight = _stencil_weight(off, w, w1, w2)

        new_vel = new_vel + weight * grid_vel
        new_C = new_C + 4.0 * inv_dx * weight * grid_vel.outer_product(dpos)

    # compute actual new_pos with new_vel
    var new_pos = pos + substep_dt * new_vel

    # impose boundary for safety, in case simulation explodes and tries to access illegal cell address
    new_pos, new_vel = impose_pos_vel(new_pos, new_vel, lower, upper)

    return (new_pos, new_vel, new_C)


def impose_pos_vel(pos: Vec3, vel: Vec3, lower: Vec3, upper: Vec3) -> Tuple[Vec3, Vec3]:
    """CubeBoundary.impose_pos_vel with restitution 0.0, i.e. a component-wise clamp."""
    var p = pos
    var v = vel
    for d in range(3):
        if p[d] < lower[d]:
            p[d] = lower[d]
            v[d] = 0.0
        elif p[d] > upper[d]:
            p[d] = upper[d]
            v[d] = 0.0
    return (p, v)


def signmax(a: Float64, eps: Float64) -> Float64:
    if a >= 0.0:
        return max(a, eps)
    return -max(-a, eps)


def _fm_off(s: Vec3, i: Int, j: Int) -> Float64:
    if i == j:
        return 0.0
    return 1.0 / signmax(s[j] - s[i], 1e-6)


def backward_svd(grad_U: Mat3, grad_S: Mat3, grad_V: Mat3, U: Mat3, S: Mat3, V: Mat3) -> Mat3:
    """Backward of a 3x3 SVD, PyTorch Functions.cpp convention."""
    var s = Vec3(S.r0[0] ** 2, S.r1[1] ** 2, S.r2[2] ** 2)
    var Fm = Mat3(
        Vec3(_fm_off(s, 0, 0), _fm_off(s, 0, 1), _fm_off(s, 0, 2)),
        Vec3(_fm_off(s, 1, 0), _fm_off(s, 1, 1), _fm_off(s, 1, 2)),
        Vec3(_fm_off(s, 2, 0), _fm_off(s, 2, 1), _fm_off(s, 2, 2)),
    )

    var S_term = U @ grad_S @ V.transpose()
    var u_term = U @ ((Fm * (U.transpose() @ grad_U - grad_U.transpose() @ U)) @ S) @ V.transpose()
    var v_term = U @ (S @ ((Fm * (V.transpose() @ grad_V - grad_V.transpose() @ V)) @ V.transpose()))
    return u_term + v_term + S_term
