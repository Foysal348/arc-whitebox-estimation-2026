import flopscope.numpy as fnp
class Estimator:
    def setup(self, ctx): pass
    def predict(self, mlp, budget):
        return fnp.zeros((mlp.depth, mlp.width), dtype=fnp.float32)
