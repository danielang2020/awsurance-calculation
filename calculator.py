import threading
from functools import lru_cache
from typing import Dict,Tuple,Callable,List

import grpc
from concurrent import futures
from sympy import sympify,latex
from sympy.core.sympify import SympifyError
from sympy.utilities.lambdify import lambdify

import calculator_pb2 as pb
import calculator_pb2_grpc as rpc
import sympy as sp

def discount(rate,t):
    return (1 + rate) ** (-t)

def pv_level(rate,n):
    return sp.Piecewise((n,sp.Eq(rate,0)),((1 - (1+rate)** (-n)) / rate ,True))

def fv_level(rate,n):
    return sp.Piecewise((n,sp.Eq(rate,0)),(((1+rate)**n-1)/rate,True))

SAFE_FUNCTIONS = {
    "Abs": sp.Abs,"Min":sp.Min,"Max":sp.Max,"Piecewise":sp.Piecewise,
    "sqrt":sp.sqrt,"log":sp.log,"ln":sp.log,"exp":sp.exp,
    "sin":sp.sin,"cos":sp.cos,"tan":sp.tan,
    "floor":sp.floor,"ceiling":sp.ceiling,"E":sp.E,"pi":sp.pi,
    "discount":discount,"pv_level":pv_level,"fv_level":fv_level
}

def _compile_expression(expr_text:str)->Tuple[Tuple[str,...],Callable,str]:
    expr = sympify(expr_text,locals=SAFE_FUNCTIONS)
    arg_names = tuple(sorted(str(s) for s in expr.free_symbols))
    try:
        fn = lambdify(arg_names,expr,modules="numexpr")
    except Exception:
        fn = lambdify(arg_names, expr, modules=["numpy", "math"])
    ltx = latex(expr)
    return arg_names,fn,ltx

@lru_cache(maxsize=1024)
def _compile_cached(expr_text:str)->Tuple[Tuple[str,...],Callable,str]:
    return _compile_expression(expr_text)

class Registry:
    def __init__(self):
        self._lock = threading.RLock()
        self._store: Dict[str,Tuple[Tuple[str,...],callable,str]] = {}

    def register(self,key:str,expr_text:str)-> Tuple[List[str],str]:
        arg_names,fn,ltx = _compile_expression(expr_text)
        with self._lock:
            self._store[key] = (arg_names,fn,ltx)
        return list(arg_names),ltx

    def get(self,key:str)->Tuple[Tuple[str,...],Callable,str]:
        with self._lock:
            if key not in self._store:
                raise KeyError(f"unknown key:{key}")
            return self._store[key]

REGISTRY = Registry()

def _args_from_vars(arg_names:Tuple[str,...],vars_dict:Dict[str,float])-> List[float]:
    missing = [n for n in arg_names if n not in vars_dict]
    if missing:
        raise ValueError(f"missing variables:{','.join(missing)}")
    return [float(vars_dict[n]) for n in arg_names]

class InsuranceCalculator(rpc.Calculator):
    def evaluate(self,request:pb.EvaluateRequest,context)->pb.EvaluateResponse:
        try:
            vals = {v.name:v.value for v in request.vars}
            if request.key:
                arg_names,fn,_ = REGISTRY.get(request.key)
            elif request.expression:
                arg_names,fn,_ = _compile_cached(request.expression)
            else:
                return pb.EvaluateResponse(code="000",message="error",result=0.0)

            res = float(fn(*_args_from_vars(arg_names,vals)))
            return pb.EvaluateResponse(code="000",result=res)
        except (KeyError,SympifyError,ValueError,Exception) as e:
            return pb.EvaluateResponse(code="000",message=str(e),result=0.0)

    def toLatex(self,request:pb.ToLatexRequest,context)->pb.ToLatexResponse:
        try:
            if request.key:
                _,_,ltx = REGISTRY.get(request.key)
            elif request.expression:
                _,_,ltx = _compile_cached(request.expression)
            else:
                return pb.ToLatexResponse(code="000",message="error",result="")
            return pb.ToLatexResponse(code="000",result=ltx)
        except (KeyError,SympifyError,Exception) as e:
            return pb.ToLatexResponse(code="000", message=str(e), result="")

    def register(self,request:pb.RegisterRequest,context)->pb.RegisterResponse:
        try:
            if not request.key or not request.expression:
                return pb.RegisterResponse(code="000",message="error",result="",arg_names=[])
            arg_names,ltx = REGISTRY.register(request.key,request.expression)
            return pb.RegisterResponse(code="000",result=ltx,arg_names=arg_names)
        except (SympifyError,Exception) as e:
            return pb.RegisterResponse(code="000", message="error", result="", arg_names=[])

def serve(host="0.0.0.0",port=8888):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    rpc.add_CalculatorServicer_to_server(InsuranceCalculator(),server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    REGISTRY.register("npv_premium","premium * pv_level(rate,years) - Max(0,claim)")
    REGISTRY.register("death_benefit","Max(sum_insured,account_value) - loan_balance")
    serve()
