from correlated_pareto_nbd.data.generate_data import pareto_nbd_simulator
from correlated_pareto_nbd.models.pareto_nbd import pareto_nbd

if __name__ == '__main__':
    simulator = pareto_nbd_simulator(N=100)
    simulator.simulate() 

    model = pareto_nbd(
        simulator.T,
        simulator.t,
        simulator.x,
        simulator.features)
    model.fit(thinning = 10)
    print(simulator.B)
    print(model.B_samples.mean(axis=0))