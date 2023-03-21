import numpy as np
from correlated_pareto_nbd.data.generate_data import pareto_nbd_simulator

if __name__ == '__main__':
    model = pareto_nbd_simulator(N=100)
    model.simulate()
    print(model.B)