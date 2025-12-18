import logging
import logging.config
import threading
from concurrent import futures
from functools import lru_cache
from typing import Dict, Tuple, Callable, List

import grpc
import sympy as sp
import yaml
from sympy import sympify, latex
from sympy.core.sympify import SympifyError
from sympy.utilities.lambdify import lambdify

import calculator_pb2 as pb
import calculator_pb2_grpc as rpc


def setup_logging(config_path):
    with open(config_path, 'r') as f:
        log_config = yaml.safe_load(f.read())
        logging.config.dictConfig(log_config)


# In your application's entry point:
setup_logging('logging_config.yaml')  # Assuming ConfigMap is mounted at /etc/config
LOGGER = logging.getLogger('calculation')


def discount(rate, t):
    return (1 + rate) ** (-t)


def pv_level(rate, n):
    return sp.Piecewise((n, sp.Eq(rate, 0)), ((1 - (1 + rate) ** (-n)) / rate, True))


def fv_level(rate, n):
    return sp.Piecewise((n, sp.Eq(rate, 0)), (((1 + rate) ** n - 1) / rate, True))


SAFE_FUNCTIONS = {
    "Abs": sp.Abs, "Min": sp.Min, "Max": sp.Max, "Piecewise": sp.Piecewise,
    "sqrt": sp.sqrt, "log": sp.log, "ln": sp.log, "exp": sp.exp,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "floor": sp.floor, "ceiling": sp.ceiling, "E": sp.E, "pi": sp.pi,
    "discount": discount, "pv_level": pv_level, "fv_level": fv_level
}


def _compile_expression(expr_text: str) -> Tuple[Tuple[str, ...], Callable, str]:
    expr = sympify(expr_text, locals=SAFE_FUNCTIONS)
    arg_names = tuple(sorted(str(s) for s in expr.free_symbols))
    try:
        fn = lambdify(arg_names, expr, modules="numexpr")
    except Exception as e:
        LOGGER.error("_compile_expression numexpr", e)
        try:
            fn = lambdify(arg_names, expr, modules=["numpy", "math"])
        except Exception as ee:
            LOGGER.error("_compile_expression numpy_math", ee)
            raise ee
    ltx = latex(expr)
    return arg_names, fn, ltx


@lru_cache(maxsize=1024)
def _compile_cached(expr_text: str) -> Tuple[Tuple[str, ...], Callable, str]:
    return _compile_expression(expr_text)


class Registry:
    def __init__(self):
        self._lock = threading.RLock()
        self._store: Dict[str, Tuple[Tuple[str, ...], callable, str]] = {}

    def register(self, key: str, expr_text: str) -> Tuple[List[str], str]:
        arg_names, fn, ltx = _compile_expression(expr_text)
        with self._lock:
            self._store[key] = (arg_names, fn, ltx)
        return list(arg_names), ltx

    def get(self, key: str) -> Tuple[Tuple[str, ...], Callable, str]:
        with self._lock:
            if key not in self._store:
                raise KeyError(f"unknown key:{key}")
            return self._store[key]


REGISTRY = Registry()


def _args_from_vars(arg_names: Tuple[str, ...], vars_dict: Dict[str, float]) -> List[float]:
    missing = [n for n in arg_names if n not in vars_dict]
    if missing:
        raise ValueError(f"missing variables:{','.join(missing)}")
    return [float(vars_dict[n]) for n in arg_names]


class InsuranceCalculator(rpc.Calculator):
    def evaluate(self, request: pb.EvaluateRequest, context) -> pb.EvaluateResponse:
        try:
            vals = {v.name: v.value for v in request.vars}
            if request.key:
                arg_names, fn, _ = REGISTRY.get(request.key)
            elif request.expression:
                arg_names, fn, _ = _compile_cached(request.expression)
            else:
                context.set_details("key or expression not found")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return pb.EvaluateResponse()

            res = float(fn(*_args_from_vars(arg_names, vals)))
            return pb.EvaluateResponse(result=res)
        except (KeyError, ValueError) as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return pb.EvaluateResponse()
        except (SympifyError, Exception) as e:
            LOGGER.error("evaluate", e)
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb.EvaluateResponse()

    def toLatex(self, request: pb.ToLatexRequest, context) -> pb.ToLatexResponse:
        try:
            if request.key:
                _, _, ltx = REGISTRY.get(request.key)
            elif request.expression:
                _, _, ltx = _compile_cached(request.expression)
            else:
                context.set_details("key or expression not found")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return pb.ToLatexResponse()
            return pb.ToLatexResponse(result=ltx)
        except (KeyError) as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return pb.ToLatexResponse()
        except (SympifyError, Exception) as e:
            LOGGER.error("toLatex", e)
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb.ToLatexResponse()

    def register(self, request: pb.RegisterRequest, context) -> pb.RegisterResponse:
        try:
            if not request.key or not request.expression:
                context.set_details("key or expression not found")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return pb.RegisterResponse()
            arg_names, ltx = REGISTRY.register(request.key, request.expression)
            return pb.RegisterResponse(result=ltx, arg_names=arg_names)
        except (SympifyError, Exception) as e:
            LOGGER.error("register", e)
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb.RegisterResponse()


def serve(host="0.0.0.0", port=8888):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    rpc.add_CalculatorServicer_to_server(InsuranceCalculator(), server)
    server.add_insecure_port(f"{host}:{port}")
    LOGGER.info(f"awsurance-calculation {host}:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    REGISTRY.register("npv_premium", "premium * pv_level(rate,years) - Max(0,claim)")
    REGISTRY.register("death_benefit", "Max(sum_insured,account_value) - loan_balance")
    serve()
