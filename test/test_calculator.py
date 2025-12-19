import threading
import time
import unittest

import grpc

import calculator as svc
import calculator_pb2 as pb
import calculator_pb2_grpc as rpc


class CalculatorTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host, cls.port = "127.0.0.1", 8080

        def run_server():
            server = grpc.server(svc.futures.ThreadPoolExecutor(max_workers=10))
            rpc.add_CalculatorServicer_to_server(svc.InsuranceCalculator(), server)
            server.add_insecure_port(f"{cls.host}:{cls.port}")
            server.start()
            cls.server = server
            while getattr(cls, "_running", True):
                time.sleep(0.05)

        cls._running = True
        t = threading.Thread(target=run_server, daemon=True)
        t.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls._running = False
        if hasattr(cls, "server"):
            cls.server.stop(0)

    def test_evaluate_invalid_argument(self):
        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        stub = rpc.CalculatorStub(channel)
        try:
            stub.evaluate(pb.EvaluateRequest())
        except grpc.RpcError as e:
            self.assertEqual(e.code(), grpc.StatusCode.INVALID_ARGUMENT)

        channel.close()

    def test_evaluate_by_no_key(self):
        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        stub = rpc.CalculatorStub(channel)
        try:
            stub.evaluate(pb.EvaluateRequest(key="sum_xy1",
                                             vars=[pb.Variable(name="x", value=1.25),
                                                   pb.Variable(name="y", value=2.0)]))
        except grpc.RpcError as e:
            self.assertEqual(e.code(), grpc.StatusCode.INVALID_ARGUMENT)
        channel.close()

    def test_evaluate_by_key(self):
        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        stub = rpc.CalculatorStub(channel)
        svc.REGISTRY.register("sum_xy", "x+y")
        resp = stub.evaluate(pb.EvaluateRequest(key="sum_xy",
                                                vars=[pb.Variable(name="x", value=1.25),
                                                      pb.Variable(name="y", value=2.0)]))
        self.assertEqual(resp.result, 3.25)
        channel.close()

    def test_evaluate_by_no_expression(self):
        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        stub = rpc.CalculatorStub(channel)
        try:
            stub.evaluate(pb.EvaluateRequest(expression="sum_xy1",
                                             vars=[pb.Variable(name="x", value=1.25),
                                                   pb.Variable(name="y", value=2.0)]))
        except grpc.RpcError as e:
            self.assertEqual(e.code(), grpc.StatusCode.INVALID_ARGUMENT)
        channel.close()

    def test_evaluate_by_expression(self):
        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        stub = rpc.CalculatorStub(channel)

        resp = stub.evaluate(pb.EvaluateRequest(expression="x * y + z",
                                                vars=[pb.Variable(name="x", value=1.25),
                                                      pb.Variable(name="y", value=2),
                                                      pb.Variable(name="z", value=3)]))
        self.assertEqual(resp.result, 5.5)
        channel.close()

    def test_tolatex_invalid_argument(self):
        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        stub = rpc.CalculatorStub(channel)
        try:
            stub.toLatex(pb.ToLatexRequest())
        except grpc.RpcError as e:
            self.assertEqual(e.code(), grpc.StatusCode.INVALID_ARGUMENT)

        channel.close()

    def test_tolatex_by_no_key(self):
        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        stub = rpc.CalculatorStub(channel)
        try:
            stub.toLatex(pb.ToLatexRequest(key="sum_xy1"))
        except grpc.RpcError as e:
            self.assertEqual(e.code(), grpc.StatusCode.INVALID_ARGUMENT)
        channel.close()

    def test_tolatex_by_key(self):
        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        stub = rpc.CalculatorStub(channel)
        svc.REGISTRY.register("sum_xy", "x+y")
        resp = stub.toLatex(pb.ToLatexRequest(key="sum_xy"))
        self.assertEqual(resp.result, "x + y")
        channel.close()

    def test_tolatex_by_expression(self):
        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        stub = rpc.CalculatorStub(channel)
        resp = stub.toLatex(pb.ToLatexRequest(expression="x ** 2"))
        self.assertEqual(resp.result, "x^{2}")
        channel.close()

    def test_register_invalid_argument(self):
        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        stub = rpc.CalculatorStub(channel)
        try:
            stub.register(pb.RegisterRequest())
        except grpc.RpcError as e:
            self.assertEqual(e.code(), grpc.StatusCode.INVALID_ARGUMENT)

        channel.close()

    def test_register(self):
        channel = grpc.insecure_channel(f"{self.host}:{self.port}")
        stub = rpc.CalculatorStub(channel)
        resp = stub.register(pb.RegisterRequest(key="abc",expression="a + b -c"))
        self.assertEqual(resp.result, "a + b - c")
        self.assertEqual(resp.arg_names,["a", "b", "c"])
        channel.close()


if __name__ == '__main__':
    unittest.main()
