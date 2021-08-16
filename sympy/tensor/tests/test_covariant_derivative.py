#!/usr/bin/env python3

from sympy import Array, eye, tensorproduct, symbols, Function, sin, tan, cos
from sympy.tensor.tensor import (
    TensorCoordinateSystem,
    TensorIndexType,
    tensor_indices,
    TensDiff,
    tensor_heads,
)

def test_coordsys_default_metric():
    "Test that coordinate system gets a default metric of all zeros"
    x, y = symbols("x y")
    coord_sys = TensorCoordinateSystem("R", [x, y])
    assert coord_sys.metric == Array(eye(2))


def test_coordsys_default_christoffel():
    "Test that coordinate system gets a default christoffel of all zeros"
    x, y = symbols("x y")
    coord_sys = TensorCoordinateSystem("R", [x, y])
    expected = tensorproduct([0]*2, [0]*2, [0]*2)
    actual = coord_sys.christoffel_2
    assert expected == actual


def test_zeroth_order_cartesian():
    "Test differentiating a zeroth-order system"
    Euclid = TensorIndexType("Euclid", dummy_name="E")
    i = tensor_indices("i", Euclid)
    psi = tensor_heads("psi", [])
    x, y = symbols("x y")
    coord_sys = TensorCoordinateSystem("R", [x, y])
    f = symbols("f", cls=Function)
    replacement = {Euclid: coord_sys, psi(): f(x, y)}
    actual = psi().covar_diff(-i).replace_with_arrays(replacement)
    expected = Array([f(x, y).diff(x), f(x, y).diff(y)])
    # This fails because of issue #21870
    assert expected == actual

def test_first_order_cartesian():
    "Test differentiating a vector-valued function in cartesian coordinates"
    Euclid = TensorIndexType("Euclid", dummy_name="E")
    i, j = tensor_indices("i j", Euclid)
    u, v = tensor_heads("u v", [Euclid])
    x, y = symbols("x y")
    coord_sys = TensorCoordinateSystem("R", [x, y])
    f_x, f_y = symbols("f_x f_y", cls=Function)
    g_x, g_y = symbols("g_x g_y", cls=Function)
    replacement = {
        Euclid: coord_sys,
        v(i): Array([f_x(x, y), f_y(x, y)]),
        u(i): Array([g_x(x, y), g_y(x, y)]),
    }
    expected = Array([
        [f_x(x, y).diff(x), f_x(x, y).diff(y)],
        [f_y(x, y).diff(x), f_y(x, y).diff(y)],
    ])
    actual = v(i).covar_diff(j).replace_with_arrays(replacement)
    assert expected == actual

    expected = f_x(x, y).diff(x) + f_y(x, y).diff(y)
    actual = v(i).covar_diff(-i).replace_with_arrays(replacement)
    assert expected == actual

    expected = Array([
        [
            f_x(x, y).diff(x) + g_x(x, y).diff(x),
            f_x(x, y).diff(y) + g_x(x, y).diff(y)
        ],
        [
            f_y(x, y).diff(x) + g_y(x, y).diff(x),
            f_y(x, y).diff(y) + g_y(x, y).diff(y)
        ],
    ])
    actual = (u(i) + v(i)).covar_diff(j).replace_with_arrays(replacement)
    assert expected == actual

    expected = f_x(x, y).diff(x) + g_x(x, y).diff(x) + f_y(x, y).diff(y) + g_y(x, y).diff(y)
    actual = (u(i) + v(i)).covar_diff(-i).replace_with_arrays(replacement)
    assert expected == actual

def test_first_order_polar():
    Euclid = TensorIndexType("Euclid", dummy_name="E")
    i, j = tensor_indices("i j", Euclid)
    u, v = tensor_heads("u v", [Euclid])
    r, theta = symbols("r theta")
    coord_sys = TensorCoordinateSystem("P", [r, theta], Array([[1, 0], [0, r**2]]))
    f_r, f_theta = symbols("f_x f_y", cls=Function)
    g_r, g_theta = symbols("g_x g_y", cls=Function)
    replacement = {
        Euclid: coord_sys,
        v(i): Array([f_r(r, theta), f_theta(r, theta)]),
        u(i): Array([g_r(r, theta), g_theta(r, theta)]),
    }
    # Check that we get the right christoffel symbol
    expected = Array([
        [
            [0, 0],
            [0, -r],
        ],
        [
            [0, 1/r],
            [1/r, 0],
        ]
    ])
    assert coord_sys.christoffel_2 == expected

    # Check that the gradient of a vector field is correct
    expected = Array([
        [
            f_r(r, theta).diff(r),
            f_r(r, theta).diff(theta) - r * f_theta(r, theta),
        ],
        [
            f_theta(r, theta).diff(r) + f_theta(r, theta) / r,
            f_theta(r, theta).diff(theta) + f_r(r, theta) / r,
        ],
    ])
    actual = v(i).covar_diff(-j).replace_with_arrays(replacement)
    assert expected == actual

    # Check that the divergence of a vector field is correct
    expected = f_r(r, theta).diff(r) + f_theta(r, theta).diff(theta) + f_r(r, theta) / r
    actual = v(i).covar_diff(-i).replace_with_arrays(replacement).expand()
    assert expected == actual

    # Check the divergence again but with the indices swapped
    expected = f_r(r, theta).diff(r) + f_theta(r, theta).diff(theta) + f_r(r, theta) / r
    actual = v(-i).covar_diff(i).replace_with_arrays(replacement).expand()
    assert expected == actual

    expected = Array([
        [
            f_r(r, theta).diff(r),
            (f_r(r, theta).diff(theta) - r * f_theta(r, theta)) / r**2,
        ],
        [
            f_theta(r, theta).diff(r) + f_theta(r, theta) / r,
            (f_theta(r, theta).diff(theta) + f_r(r, theta) / r) / r**2,
        ],
    ])
    actual = v(i).covar_diff(j).replace_with_arrays(replacement)
    assert expected == actual

def test_cartesian_laplacian():
    "Test Laplace's equation in cartesian coordinates."
    Euclid = TensorIndexType("Euclid", dummy_name="E")
    psi = tensor_heads("psi", [])
    i, j = tensor_indices("i j", Euclid)
    x, y, z = symbols("x y z")
    psi_f = symbols("psi", cls=Function)(x, y, z)
    coord_sys = TensorCoordinateSystem("R", [x, y, z])
    replacement = {
        Euclid: coord_sys,
        psi(): psi_f
    }
    expected = psi_f.diff(x, x) + psi_f.diff(y, y) + psi_f.diff(z, z)
    actual = psi().covar_diff(i, -i).replace_with_arrays(replacement)
    assert expected == actual


def test_spherical_laplacian():
    "Test Laplace's equation in spherical coordinates."
    Euclid = TensorIndexType("Euclid", dummy_name="E")
    psi = tensor_heads("psi", [])
    i, j = tensor_indices("i j", Euclid)
    rho, theta, phi = symbols("rho theta phi")
    psi_f = symbols("psi", cls=Function)(rho, theta, phi)
    coord_sys = TensorCoordinateSystem(
        "S", [rho, theta, phi],
        Array([
            [1, 0, 0],
            [0, rho**2, 0],
            [0, 0, rho**2 * sin(theta)**2]
        ])
    )
    replacement = {
        Euclid: coord_sys,
        psi(): psi_f
    }
    expected = (
        psi_f.diff(rho, rho) + 2/rho * psi_f.diff(rho)
        + psi_f.diff(theta) * cos(theta) / (rho**2 * sin(theta))
        + psi_f.diff(theta, theta) / rho**2
        + psi_f.diff(phi, phi) / (rho**2 * sin(theta)**2)
    )
    actual = psi().covar_diff(i, -i).replace_with_arrays(replacement).expand()
    assert expected == actual

    actual = psi().covar_diff(-i, i).replace_with_arrays(replacement).expand()
    assert expected == actual
