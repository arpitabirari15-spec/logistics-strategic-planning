"""Capacitated Vehicle Routing Problem (CVRP) skeleton using Google OR-Tools.

Requires: pip install ortools
"""
from ortools.constraint_solver import routing_enums_pb2, pywrapcp


def solve_cvrp(distance_matrix, demands, vehicle_capacities, num_vehicles, depot_index=0):
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    def demand_callback(from_index):
        return demands[manager.IndexToNode(from_index)]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, vehicle_capacities, True, "Capacity"
    )

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    return routing.SolveWithParameters(search_params)


if __name__ == "__main__":
    # Placeholder toy example — replace with real hub/zone-centroid distances.
    distance_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0],
    ]
    demands = [0, 3, 4, 2]
    vehicle_capacities = [10, 10]
    solution = solve_cvrp(distance_matrix, demands, vehicle_capacities, num_vehicles=2)
    print("Solved:", solution is not None)
