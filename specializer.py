import ctypes
import sys

import llvmlite.ir as ll
import llvmlite.binding as llvm

import coreast


llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()


int_type = ll.IntType(64)
float_type = ll.FloatType()


_llvm_type_map = {
    coreast.int_type: int_type,
    coreast.float_type: float_type,
    coreast.void_type: ll.VoidType,
}


_ctypes_type_map = {
    coreast.int_type: ctypes.c_int64,
    coreast.void_type: ctypes.c_void_p,
    coreast.float_type: ctypes.c_float,
}


def to_llvm_type(ty):
    return _llvm_type_map[ty]


def to_ctypes(ty):
    return _ctypes_type_map[ty]


def mangle_func_name(func_name, arg_types):
    return func_name + str(hash(tuple(repr(ty) for ty in arg_types)))


class LLVMSpecializer:
    def __init__(self, module, specialized_types, arg_types, return_type):
        self.function = None
        self.builder = None
        self.locals = {}
        self.specialized_types = specialized_types
        self.arg_types = arg_types
        self.return_type = return_type
        self.module = module

    def start_function(self, name, module, return_type, arg_types):
        func_type = ll.FunctionType(return_type, arg_types)
        func = ll.Function(module, func_type, name=name)
        entry_block = func.append_basic_block("entry")
        builder = ll.IRBuilder(entry_block)

        self.exit_block = func.append_basic_block("exit")
        self.function = func
        self.builder = builder

    def end_function(self):
        self.builder.position_at_end(self.exit_block)

        if "retval" in self.locals:
            retval = self.builder.load(self.locals["retval"])
            self.builder.ret(retval)
        else:
            self.builder.ret_void()

    def add_block(self, name):
        return self.function.append_basic_block(name)

    def set_block(self, block):
        self.block = block
        self.builder.position_at_end(self.exit_block)

    def specialize(self, node):
        if isinstance(node.ty, coreast.TVar):
            return to_llvm_type(self.specialized_types[node.ty.name])
        else:
            return to_llvm_type(node.ty)

    def visit_Integer(self, node):
        ty = self.specialize(node)
        return ll.Constant(ty, node.val)

    def visit_Float(self, node):
        ty = self.specialize(node)
        return ll.Constant(ty, node.val)

    def visit_Var(self, node):
        return self.builder.load(self.locals[node.name])

    def visit_Return(self, node):
        val = self.visit(node.val)
        self.builder.store(val, self.locals["retval"])

        self.builder.branch(self.exit_block)

    def visit_Function(self, node):
        return_type = to_llvm_type(self.return_type)
        arg_types = [to_llvm_type(ty) for ty in self.arg_types]
        func_name = mangle_func_name(node.name, arg_types)

        self.start_function(func_name, self.module, return_type, arg_types)

        for arg, llvmarg, arg_type in zip(node.args, self.function.args, self.arg_types):
            reference = self.builder.alloca(to_llvm_type(arg_type))
            self.locals[arg.name] = reference
            self.builder.store(llvmarg, reference)

        self.locals["retval"] = self.builder.alloca(return_type)

        for child in node.body:
            self.visit(child)

        self.end_function()

    def visit_PrimOp(self, node):
        left = self.visit(node.args[0])
        right = self.visit(node.args[1])
        if left.type == float_type:
            return self.builder.fadd(left, right)
        else:
            return self.builder.add(left, right)

    def visit(self, node):
        return getattr(self, "visit_" + type(node).__name__)(node)


def type_for_value(value):
    if isinstance(value, int):
        if value <= sys.maxsize and value >= -sys.maxsize:
            return coreast.int_type
        else:
            raise ValueError("Integer out of bounds")
    elif isinstance(value, float):
        return coreast.float_type

    raise ValueError("Unsupported type: " + repr(type(value)))


_target_machine = llvm.Target.from_default_triple().create_target_machine()
_engine = llvm.create_mcjit_compiler(llvm.parse_assembly(""), _target_machine)


def jitify(func):
    ast = coreast.transform(func)
    inference = coreast.InferenceVisitor()
    signature = inference.visit(ast)
    mgu = coreast.solve(inference.constraints)
    inferred_type = coreast.apply_solution(mgu, signature)

    print(func, inferred_type, file=sys.stderr)

    cache = {}

    def _wrapper(*args):
        spec_arg_types = [type_for_value(val) for val in args]
        key = mangle_func_name(ast.name, [to_llvm_type(ty) for ty in spec_arg_types])

        if key in cache:
            return cache[key](*args)

        spec_ty = coreast.TFun(arg_types=spec_arg_types, return_type=coreast.TVar("$retty"))
        unifier = coreast.unify_types(inferred_type, spec_ty)
        specializer = coreast.compose_solutions(unifier, mgu)

        spec_return_type = coreast.apply_solution(specializer, coreast.TVar("$retty"))
        spec_fun = coreast.TFun(arg_types=spec_arg_types, return_type=spec_return_type)

        module = ll.Module()
        generator = LLVMSpecializer(module, specializer, spec_arg_types, spec_return_type)
        generator.visit(ast)

        native_module = llvm.parse_assembly(repr(module))
        pmb = llvm.create_pass_manager_builder()
        pmb.opt_level = 2
        pm = llvm.create_module_pass_manager()
        pmb.populate(pm)
        pm.run(native_module)

        native_module.verify()

        print(key, spec_fun, "\n", native_module, file=sys.stderr)

        _engine.add_module(native_module)
        _engine.finalize_object()

        name = native_module.get_function(key).name
        fptr = _engine.get_function_address(name)
        ctypes_arg_types = [to_ctypes(ty) for ty in spec_arg_types]
        cfunc = ctypes.CFUNCTYPE(to_ctypes(spec_return_type), *ctypes_arg_types)(fptr)
        cfunc.__name__ = func.__name__
        cache[key] = cfunc
        return cfunc(*args)

    return _wrapper


@jitify
def identity(a):
    return a

@jitify
def constant():
    return 32

@jitify
def add_two(x):
    return x + 2

@jitify
def double(x):
    return x + x

if __name__ == "__main__":
    print(constant())
    print(constant())
    print(identity(42))
    print(identity(52))
    print(add_two(2))
    print(double(5.0))
    print(double(5))
