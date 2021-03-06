import ast
import collections
import inspect
import string
import textwrap
import types

from typing import List, Tuple


class Node:
    _fields = []  # type: List[str]

    def __init__(self, location: Tuple[int, int]=None) -> None:
        self.location = location

    def __repr__(self):
        fields = ", ".join("{}={}".format(name, getattr(self, name)) for name in self.__class__._fields)
        return "{}({})".format(self.__class__.__name__, fields)


class Noop(Node):
    pass


class Var(Node):
    _fields = ["name", "ty"]

    def __init__(self, name: str, ty=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.ty = ty


class Integer(Node):
    _fields = ["val"]

    def __init__(self, val: int, ty=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.val = val
        self.ty = ty


class Float(Node):
    _fields = ["val"]

    def __init__(self, val: int, ty=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.val = val
        self.ty = ty


class Bool(Node):
    _fields = ["val"]

    def __init__(self, val: int, ty=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.val = val
        self.ty = ty


class PrimOp(Node):
    _fields = ["op", "args"]

    def __init__(self, op: str, args: List[Node], ty=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.op = op
        self.args = args


class Function(Node):
    _fields = ["name", "args", "body"]

    def __init__(self, name: str, args: List[Var], body: List[Node], **kwargs) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.args = args
        self.body = body


class Return(Node):
    _fields = ["val"]

    def __init__(self, val: Node, **kwargs) -> None:
        super().__init__(**kwargs)
        self.val = val


class Assign(Node):
    _fields = ["var", "val", "ty"]

    def __init__(self, var: Var, val: Node, ty=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.var = var
        self.val = val
        self.ty = ty


class ForLoop(Node):
    _fields = ["var", "start", "end", "step", "body"]

    def __init__(self, var: Var, start: Node, end: Node, step: Node, body: List[Node], **kwargs) -> None:
        super().__init__(**kwargs)
        self.var = var
        self.start = start
        self.end = end
        self.step = step
        self.body = body


class Apply(Node):
    _fields = ["fn", "args"]

    def __init__(self, fn: Node, args: List[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self.fn = fn
        self.args = args


class PythonVisitor(ast.NodeVisitor):
    def transform(self, code):
        if isinstance(code, (types.FunctionType, types.ModuleType)):
            code = inspect.getsource(code)

        parsed = ast.parse(textwrap.dedent(code))
        return self.visit(parsed)

    def visit_Pass(self, node):
        return Noop()

    def visit_Module(self, node):
        return self.visit(node.body[0])

    def visit_Name(self, node):
        return Var(node.id)

    def visit_Num(self, node):
        if isinstance(node.n, int):
            return Integer(node.n)
        elif isinstance(node.n, float):
            return Float(node.n)
        raise NotImplementedError

    def visit_NameConstant(self, node):
        if node.value is True:
            return Bool(True)
        elif node.value is False:
            return Bool(False)
        else:
            raise NotImplementedError("NameConstant value: " + repr(node.value))

    def visit_FunctionDef(self, node):
        body = list(map(self.visit, node.body))
        args = [Var(arg.arg) for arg in node.args.args]
        return Function(node.name, args, body)

    def visit_Return(self, node):
        return Return(self.visit(node.value))

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Add):
            op = "+"
        else:
            raise ValueError("Unsupported op:", node.op)
        return PrimOp(op, [self.visit(node.left), self.visit(node.right)])

    def visit_Assign(self, node):
        if len(node.targets) != 1:
            raise NotImplementedError("Not implemented: multiple assignment")
        return Assign(self.visit(node.targets[0]), self.visit(node.value))

    def visit_For(self, node):
        if not isinstance(node.iter, ast.Call):
            raise ValueError("Cannot iterate over " + repr(node.iter))

        if node.iter.func.id != "range":
            raise ValueError("Must iterate over range, not " + node.iter.func.id)

        if len(node.iter.args) != 1:
            raise NotImplementedError("Not implemented: iteration not over range(n)")

        var = self.visit(node.target)
        start = Integer(0)
        end = self.visit(node.iter.args[0])
        step = Integer(1)
        return ForLoop(var, start, end, step, [self.visit(b) for b in node.body])


def transform(code):
    return PythonVisitor().transform(code)


def unique_names():
    n = 0
    while True:
        for char in string.ascii_lowercase:
            yield "'" + char + (str(n) if n > 0 else "")
        n += 1


class TVar:
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, TVar):
            return self.name == other.name
        return False

    def __repr__(self):
        return self.name


class TCon:
    def __init__(self, typename):
        self.typename = typename

    def __hash__(self):
        return hash(self.typename)

    def __eq__(self, other):
        if isinstance(other, TCon):
            return self.typename == other.typename
        return False

    def __repr__(self):
        return self.typename


class TFun:
    def __init__(self, arg_types, return_type):
        self.arg_types = arg_types
        self.return_type = return_type

    def __eq__(self, other):
        if isinstance(other, TFun):
            return self.arg_types == other.arg_types and \
                self.return_type == other.return_type
        return False

    def __repr__(self):
        return "({}) -> {}".format(", ".join(str(x) for x in self.arg_types),
                                   self.return_type)


def free_tvars(ty):
    if isinstance(ty, TVar):
        return {ty.name}
    elif isinstance(ty, TCon):
        return {}
    elif isinstance(ty, TFun):
        result = {}
        for t in ty.arg_types:
            result.update(free_tvars(t))
        return result.union(free_tvars(ty.return_type))
    else:
        raise ValueError("Not a type: " + repr(ty))


int_type = TCon("int")
float_type = TCon("float")
bool_type = TCon("bool")
void_type = TCon("void")


class InferenceVisitor:
    def __init__(self):
        self.constraints = []
        self.env = {}
        self.names = unique_names()

        # Yeah, this is ugly.
        self.return_type = None

    def new_var(self):
        return TVar("$" + next(self.names))

    def visit(self, node):
        return getattr(self, "visit_" + type(node).__name__)(node)

    def visit_Var(self, node):
        node.ty = self.env[node.name]
        return node.ty

    def visit_Integer(self, node):
        node.ty = int_type
        return int_type

    def visit_Float(self, node):
        node.ty = float_type
        return float_type

    def visit_Bool(self, node):
        node.ty = bool_type
        return bool_type

    def visit_Return(self, node):
        ty = self.visit(node.val)
        self.constraints.append((ty, self.return_type))

    def visit_Assign(self, node):
        ty = self.visit(node.val)
        if node.var.name in self.env:
            self.constraints.append((ty, self.env[node.var.name]))
        else:
            self.env[node.var.name] = self.new_var()
            self.constraints.append((ty, self.env[node.var.name]))
            node.ty = node.var.ty = self.env[node.var.name]

    def visit_ForLoop(self, node):
        self.env[node.var.name] = self.new_var()
        self.constraints.append((self.env[node.var.name], int_type))

        end_ty = self.visit(node.end)
        self.constraints.append((end_ty, int_type))

        for b in node.body:
            self.visit(b)

    def visit_Function(self, node):
        self.return_type = self.new_var()
        arg_types = []

        for arg in node.args:
            ty = self.new_var()
            arg.ty = ty
            self.env[arg.name] = ty
            arg_types.append(ty)

        for b in node.body:
            self.visit(b)

        return TFun(arg_types, self.return_type)

    def visit_PrimOp(self, node):
        if node.op == "+":
            left = self.visit(node.args[0])
            right = self.visit(node.args[1])
            self.constraints.append((left, right))
            return right
        else:
            raise ValueError("Unsupported primop: " + repr(node))


# Actual solver - Robinson's algorithm
def empty_solution():
    return {}


def apply_solution(solution, ty):
    if isinstance(ty, TCon):
        return ty
    elif isinstance(ty, TFun):
        arg_types = [apply_solution(solution, arg_type) for arg_type in ty.arg_types]
        return_type = apply_solution(solution, ty.return_type)

        return TFun(arg_types, return_type)
    elif isinstance(ty, TVar):
        return solution.get(ty.name, ty)


def apply_solution_equations(solution, equations):
    return [(apply_solution(solution, lhs), apply_solution(solution, rhs))
            for lhs, rhs in equations]


def unify_types(x, y):
    if isinstance(x, TCon) and isinstance(y, TCon) and x == y:
        return empty_solution()
    elif isinstance(x, TFun) and isinstance(y, TFun):
        if len(x.arg_types) != len(y.arg_types):
            raise Exception("Mismatch in number of arguments")
        sol1 = solve(zip(x.arg_types, y.arg_types))
        sol2 = unify_types(apply_solution(sol1, x.return_type), apply_solution(sol1, y.return_type))
        return compose_solutions(sol2, sol1)
    elif isinstance(x, TVar):
        return bind(x.name, y)
    elif isinstance(y, TVar):
        return bind(y.name, x)
    else:
        raise Exception("Type mismatch: {} vs {}".format(x, y))


def bind(name, ty):
    if hasattr(ty, "name") and ty.name == name:
        return empty_solution()
    elif name in free_tvars(ty):
        raise Exception("Infinite type")
    else:
        return { name: ty }


def solve(equations):
    mgu = empty_solution()
    equations = collections.deque(equations)

    while equations:
        (lhs, rhs) = equations.pop()
        sol = unify_types(lhs, rhs)
        mgu = compose_solutions(sol, mgu)
        equations = collections.deque(apply_solution_equations(sol, equations))

    return mgu


def union(sol1, sol2):
    nenv = sol1.copy()
    nenv.update(sol2)
    return nenv


def compose_solutions(sol1, sol2):
    sol3 = { t: apply_solution(sol1, u) for t, u in sol2.items() }
    return union(sol1, sol3)
