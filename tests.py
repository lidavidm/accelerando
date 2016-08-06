import unittest

import accelerando
import accelerando.coreast as ast


def infer(func):
    ut = ast.transform(func)
    inference = ast.InferenceVisitor()
    signature = inference.visit(ut)
    mgu = ast.solve(inference.constraints)
    return signature, mgu, ast.apply_solution(mgu, signature)


class TestTypeChecker(unittest.TestCase):
    def test_constant_int(self):
        def constant():
            return 42

        signature, mgu, inferred = infer(constant)
        self.assertEqual(inferred, ast.TFun([], ast.int_type))


class TestConstant(unittest.TestCase):
    pass


class TestIdentity(unittest.TestCase):
    def setUp(self):
        def identity(x):
            return x
        self.identity = accelerando.jitify(identity)

    def test_integers(self):
        for i in range(-100, 100):
            self.assertEqual(i, self.identity(i))


class TestAssign(unittest.TestCase):
    def setUp(self):
        def assign1(x, y):
            z = x + 2
            h = z + y
            return h
        self.assign1 = assign1
        self.assign1j = accelerando.jitify(assign1)

    def test_integers(self):
        for i in range(-128, 128):
            for j in range(-128, 128):
                self.assertEqual(self.assign1(i, j), self.assign1j(i, j))
