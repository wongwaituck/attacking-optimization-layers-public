#!/usr/bin/env python3
import onnxruntime
from maraboupy import Marabou
import numpy as np
import sys


options = Marabou.createOptions(verbosity = 0)

filename = sys.argv[1]
network = Marabou.read_onnx(filename)


inputVars = network.inputVars[0]
outputVars = network.outputVars

# should be from 0 to 1
for i in range(inputVars.shape[0]):
    network.setLowerBound(inputVars[i], 0.0)
    network.setUpperBound(inputVars[i], 1.0)

# all should equal to 0
for i, outputVar in enumerate(outputVars):
    if i == 0:
        network.addEquality([outputVar], [-1] , -0.1)
    elif i == 1:
        network.addEquality([outputVar], [-1] , -0.1)
    elif i == 2:
        network.addEquality([outputVar], [1] , -0.1)
    elif i == 3:
        network.addEquality([outputVar], [1] , -0.1)
    elif i == 4:
        network.addEquality([outputVar], [-1] , -0.1)
    elif i == 5:
        network.addEquality([outputVar], [1] , -0.2)
print("Check query with less restrictive output constraint (Should be SAT)")
vals, stats = network.solve(options = options)
assert len(vals) > 0

print(vals)
ans = []
for i in range(inputVars.shape[0]):
    ans.append(vals[i])

print(ans) 