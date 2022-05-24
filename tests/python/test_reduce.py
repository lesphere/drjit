import drjit as dr
import pytest
import importlib

@pytest.fixture(scope="module", params=['drjit.cuda', 'drjit.llvm'])
def m(request):
    if 'cuda' in request.param:
        if not dr.has_backend(dr.JitBackend.CUDA):
            pytest.skip('CUDA mode is unsupported')
    else:
        if not dr.has_backend(dr.JitBackend.LLVM):
            pytest.skip('LLVM mode is unsupported')
    yield importlib.import_module(request.param)


def test01_sum(m):
    assert dr.allclose(dr.sum(6.0), 6)

    a = dr.sum(m.Float([1, 2, 3]))
    assert dr.allclose(a, 6)
    assert type(a) == m.Float

    a = dr.sum([1.0, 2.0, 3.0])
    assert dr.allclose(a, 6)
    assert type(a) == float

    a = dr.sum(m.Array3f(1, 2, 3))
    assert dr.allclose(a, 6)
    assert type(a) == m.Float

    a = dr.sum(m.Array3i(1, 2, 3))
    assert dr.allclose(a, 6)
    assert type(a) == m.Int

    a = dr.sum(m.ArrayXf([1, 2, 3], [2, 3, 4], [3, 4, 5]))
    assert dr.allclose(a, [6, 9, 12])
    assert type(a) == m.Float


def test02_prod(m):
    assert dr.allclose(dr.prod(6.0), 6.0)

    a = dr.prod(m.Float([1, 2, 3]))
    assert dr.allclose(a, 6)
    assert type(a) == m.Float

    a = dr.prod([1.0, 2.0, 3.0])
    assert dr.allclose(a, 6)
    assert type(a) == float

    a = dr.prod(m.Array3f(1, 2, 3))
    assert dr.allclose(a, 6)
    assert type(a) == m.Float

    a = dr.prod(m.Array3i(1, 2, 3))
    assert dr.allclose(a, 6)
    assert type(a) == m.Int

    a = dr.prod(m.ArrayXf([1, 2, 3], [2, 3, 4], [3, 4, 5]))
    assert dr.allclose(a, [6, 24, 60])
    assert type(a) == m.Float


def test03_max(m):
    assert dr.allclose(dr.max(6.0), 6.0)

    a = dr.max(m.Float([1, 2, 3]))
    assert dr.allclose(a, 3)
    assert type(a) == m.Float

    a = dr.max([1.0, 2.0, 3.0])
    assert dr.allclose(a, 3)
    assert type(a) == float

    a = dr.max(m.Array3f(1, 2, 3))
    assert dr.allclose(a, 3)
    assert type(a) == m.Float

    a = dr.max(m.Array3i(1, 2, 3))
    assert dr.allclose(a, 3)
    assert type(a) == m.Int

    a = dr.max(m.ArrayXf([1, 2, 5], [2, 3, 4], [3, 4, 3]))
    assert dr.allclose(a, [3, 4, 5])
    assert type(a) == m.Float


def test03_min(m):
    assert dr.allclose(dr.min(6.0), 6.0)

    a = dr.min(m.Float([1, 2, 3]))
    assert dr.allclose(a, 1)
    assert type(a) == m.Float

    a = dr.min([1.0, 2.0, 3.0])
    assert dr.allclose(a, 1)
    assert type(a) == float

    a = dr.min(m.Array3f(1, 2, 3))
    assert dr.allclose(a, 1)
    assert type(a) == m.Float

    a = dr.min(m.Array3i(1, 2, 3))
    assert dr.allclose(a, 1)
    assert type(a) == m.Int

    a = dr.min(m.ArrayXf([1, 2, 5], [2, 3, 4], [3, 4, 3]))
    assert dr.allclose(a, [1, 2, 3])
    assert type(a) == m.Float


def test04_minimum(m):
    assert dr.allclose(dr.minimum(6.0, 4.0), 4.0)

    a = dr.minimum(m.Float([1, 2, 3]), m.Float(2))
    assert dr.allclose(a, [1, 2, 2])
    assert type(a) == m.Float

    a = dr.minimum([1, 2, 3], [2.0, 2.0, 2.0])
    assert dr.allclose(a, [1, 2, 2])
    assert type(a) == list

    a = dr.minimum(m.Float([1, 2, 3]), [2.0, 2.0, 2.0])
    assert dr.allclose(a, [1, 2, 2])
    assert type(a) == m.Float

    a = dr.minimum(m.Array3f(1, 2, 3), m.Float(2))
    assert dr.allclose(a, [1, 2, 2])
    assert type(a) == m.Array3f

    a = dr.minimum(m.Array3i(1, 2, 3), m.Float(2))
    assert dr.allclose(a, [1, 2, 2])
    assert type(a) == m.Array3f

    a = dr.minimum(m.ArrayXf([1, 2, 5], [2, 1, 4], [3, 4, 3]), m.Float(1, 2, 3))
    assert dr.allclose(a, [[1, 2, 3], [1, 1, 3], [1, 2, 3]])
    assert type(a) == m.ArrayXf

    a = dr.minimum(m.ArrayXf([1, 2, 5], [2, 1, 4], [3, 4, 3]), m.Array3f(1, 2, 3))
    assert dr.allclose(a, [[1, 1, 1], [2, 1, 2], [3, 3, 3]])
    assert type(a) == m.ArrayXf


def test05_maximum(m):
    assert dr.allclose(dr.maximum(6.0, 4.0), 6.0)

    a = dr.maximum(m.Float([1, 2, 3]), m.Float(2))
    assert dr.allclose(a, [2, 2, 3])
    assert type(a) == m.Float

    a = dr.maximum([1, 2, 3], [2.0, 2.0, 2.0])
    assert dr.allclose(a, [2, 2, 3])
    assert type(a) == list

    a = dr.maximum(m.Float([1, 2, 3]), [2.0, 2.0, 2.0])
    assert dr.allclose(a, [2, 2, 3])
    assert type(a) == m.Float

    a = dr.maximum(m.Array3f(1, 2, 3), m.Float(2))
    assert dr.allclose(a, [2, 2, 3])
    assert type(a) == m.Array3f

    a = dr.maximum(m.Array3i(1, 2, 3), m.Float(2))
    assert dr.allclose(a, [2, 2, 3])
    assert type(a) == m.Array3f

    a = dr.maximum(m.ArrayXf([1, 2, 5], [2, 1, 4], [3, 4, 3]), m.Float(1, 2, 3))
    assert dr.allclose(a, [[1, 2, 5], [2, 2, 4], [3, 4, 3]])
    assert type(a) == m.ArrayXf

    a = dr.maximum(m.ArrayXf([1, 2, 5], [2, 1, 4], [3, 4, 3]), m.Array3f(1, 2, 3))
    assert dr.allclose(a, [[1, 2, 5], [2, 2, 4], [3, 4, 3]])
    assert type(a) == m.ArrayXf