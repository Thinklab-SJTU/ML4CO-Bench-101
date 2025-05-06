from ml4co_kit import *

###############################
#             TSP             #
###############################

tsp_solver = TSPSolver()
tsp_solver.from_txt("test_dataset/tsp/tsp50_concorde_5.688.txt", ref=True)
tsp_solver.from_txt("ml4co_result/tsp/tsp50_ml4co_5.688.txt", ref=False)
print("\"TSP-50\":", tsp_solver.evaluate(calculate_gap=True), flush=True)

tsp_solver = TSPSolver()
tsp_solver.from_txt("test_dataset/tsp/tsp100_concorde_7.756.txt", ref=True)
tsp_solver.from_txt("ml4co_result/tsp/tsp100_ml4co_7.756.txt", ref=False)
print("\"TSP-100\":", tsp_solver.evaluate(calculate_gap=True), flush=True)

tsp_solver = TSPSolver()
tsp_solver.from_txt("test_dataset/tsp/tsp500_concorde_16.546.txt", ref=True)
tsp_solver.from_txt("ml4co_result/tsp/tsp500_ml4co_16.588.txt", ref=False)
print("\"TSP-500\":", tsp_solver.evaluate(calculate_gap=True), flush=True)

tsp_solver = TSPSolver()
tsp_solver.from_txt("test_dataset/tsp/tsp1000_concorde_23.118.txt", ref=True)
tsp_solver.from_txt("ml4co_result/tsp/tsp1000_ml4co_23.271.txt", ref=False)
print("\"TSP-1K\":", tsp_solver.evaluate(calculate_gap=True), flush=True)

tsp_solver = TSPSolver()
tsp_solver.from_txt("test_dataset/tsp/tsp10000_lkh_500_71.755.txt", ref=True)
tsp_solver.from_txt("ml4co_result/tsp/tsp10000_ml4co_72.832.txt", ref=False)
print("\"TSP-10K\":", tsp_solver.evaluate(calculate_gap=True), flush=True)


###############################
#            ATSP             #
###############################

atsp_solver = ATSPSolver()
atsp_solver.from_txt("test_dataset/atsp/atsp50_uniform_lkh_1000_1.5545.txt", ref=True)
atsp_solver.from_txt("ml4co_result/atsp/atsp50_ml4co_1.557.txt", ref=False)
print("\"ATSP-50\":", atsp_solver.evaluate(calculate_gap=True), flush=True)

atsp_solver = ATSPSolver()
atsp_solver.from_txt("test_dataset/atsp/atsp100_uniform_lkh_1000_1.5660.txt", ref=True)
atsp_solver.from_txt("ml4co_result/atsp/atsp100_ml4co_1.581.txt", ref=False)
print("\"ATSP-100\":", atsp_solver.evaluate(calculate_gap=True), flush=True)

atsp_solver = ATSPSolver()
atsp_solver.from_txt("test_dataset/atsp/atsp200_uniform_lkh_1000_1.5647.txt", ref=True)
atsp_solver.from_txt("ml4co_result/atsp/atsp200_ml4co_1.588.txt", ref=False)
print("\"ATSP-200\":", atsp_solver.evaluate(calculate_gap=True), flush=True)

atsp_solver = ATSPSolver()
atsp_solver.from_txt("test_dataset/atsp/atsp500_uniform_lkh_1000_1.5734.txt", ref=True)
atsp_solver.from_txt("ml4co_result/atsp/atsp500_ml4co_1.598.txt", ref=False)
print("\"ATSP-500\":", atsp_solver.evaluate(calculate_gap=True), flush=True)


###############################
#            CVRP             #
###############################

cvrp_solver = CVRPSolver()
cvrp_solver.from_txt("test_dataset/cvrp/cvrp50_hgs-1s_10.366.txt", ref=True)
cvrp_solver.from_txt("ml4co_result/cvrp/cvrp50_ml4co_10.489.txt", ref=False)
print("\"CVRP-50\":", cvrp_solver.evaluate(calculate_gap=True), flush=True)

cvrp_solver = CVRPSolver()
cvrp_solver.from_txt("test_dataset/cvrp/cvrp100_hgs-20s_15.563.txt", ref=True)
cvrp_solver.from_txt("ml4co_result/cvrp/cvrp100_ml4co_15.822.txt", ref=False)
print("\"CVRP-100\":", cvrp_solver.evaluate(calculate_gap=True), flush=True)

cvrp_solver = CVRPSolver()
cvrp_solver.from_txt("test_dataset/cvrp/cvrp200_hgs-60s_19.630.txt", ref=True)
cvrp_solver.from_txt("ml4co_result/cvrp/cvrp200_ml4co_20.091.txt", ref=False)
print("\"CVRP-200\":", cvrp_solver.evaluate(calculate_gap=True), flush=True)

cvrp_solver = CVRPSolver()
cvrp_solver.from_txt("test_dataset/cvrp/cvrp500_hgs-300s_37.154.txt", ref=True)
cvrp_solver.from_txt("ml4co_result/cvrp/cvrp500_ml4co_37.901.txt", ref=False)
print("\"CVRP-500\":", cvrp_solver.evaluate(calculate_gap=True), flush=True)


###############################
#             MIS             #
###############################

mis_solver = MISSolver()
mis_solver.from_txt("test_dataset/mis/mis_rb-small_kamis-60s_20.090.txt", ref=True)
mis_solver.from_txt("ml4co_result/mis/mis_rb-small_ml4co_20.070.txt", ref=False, cover=False)
print("\"MIS-RB-SMALL\":", mis_solver.evaluate(calculate_gap=True), flush=True)

mis_solver = MISSolver()
mis_solver.from_txt("test_dataset/mis/mis_rb-large_kamis-60s_43.004.txt", ref=True)
mis_solver.from_txt("ml4co_result/mis/mis_rb-large_ml4co_42.400.txt", ref=False, cover=False)
print("\"MIS-RB-LARGE\":", mis_solver.evaluate(calculate_gap=True), flush=True)

mis_solver = MISSolver()
mis_solver.from_txt("test_dataset/mis/mis_er-700-800_kamis-60s_44.969.txt", ref=True)
mis_solver.from_txt("ml4co_result/mis/mis_er-700-800_ml4co_44.984.txt", ref=False, cover=False)
print("\"MIS-ER-700-800\":", mis_solver.evaluate(calculate_gap=True), flush=True)

mis_solver = MISSolver()
mis_solver.from_txt("test_dataset/mis/mis_satlib_kamis-60s_425.954.txt", ref=True)
mis_solver.from_txt("ml4co_result/mis/mis_satlib_ml4co_425.316.txt", ref=False, cover=False)
print("\"MIS-SATLIB\":", mis_solver.evaluate(calculate_gap=True), flush=True)

mis_solver = MISSolver()
mis_solver.from_txt("test_dataset/mis/mis_er-1400-1600_kamis-60s_50.938.txt", ref=True)
mis_solver.from_txt("ml4co_result/mis/mis_er-1400-1600_ml4co_50.719.txt", ref=False, cover=False)
print("\"MIS-ER-1400-1600\":", mis_solver.evaluate(calculate_gap=True), flush=True)

mis_solver = MISSolver()
mis_solver.from_txt("test_dataset/mis/mis_rb-giant_kamis-60s_49.260.txt", ref=True)
mis_solver.from_txt("ml4co_result/mis/mis_rb-giant_ml4co_47.880.txt", ref=False, cover=False)
print("\"MIS-RB-GIANT\":", mis_solver.evaluate(calculate_gap=True), flush=True)


###############################
#             MCl             #
###############################

mcl_solver = MClSolver()
mcl_solver.from_txt("test_dataset/mcl/mcl_rb-small_gurobi-60s_19.082.txt", ref=True)
mcl_solver.from_txt("ml4co_result/mcl/mcl_rb-small_ml4co_19.082.txt", ref=False, cover=False)
print("\"MCl-RB-SMALL\":", mcl_solver.evaluate(calculate_gap=True), flush=True)

mcl_solver = MClSolver()
mcl_solver.from_txt("test_dataset/mcl/mcl_rb-large_gurobi-300s_40.182.txt", ref=True)
mcl_solver.from_txt("ml4co_result/mcl/mcl_rb-large_ml4co_40.256.txt", ref=False, cover=False)
print("\"MCl-RB-LARGE\":", mcl_solver.evaluate(calculate_gap=True), flush=True)

mcl_solver = MClSolver()
mcl_solver.from_txt("test_dataset/mcl/mcl_twitter_gurobi-60s_14.210.txt", ref=True)
mcl_solver.from_txt("ml4co_result/mcl/mcl_twitter_ml4co_14.210.txt", ref=False, cover=False)
print("\"MCl-TWITTER\":", mcl_solver.evaluate(calculate_gap=True), flush=True)

mcl_solver = MClSolver()
mcl_solver.from_txt("test_dataset/mcl/mcl_collab_gurobi-60s_42.113.txt", ref=True)
mcl_solver.from_txt("ml4co_result/mcl/mcl_collab_ml4co_42.113.txt", ref=False, cover=False)
print("\"MCl-COLLAB\":", mcl_solver.evaluate(calculate_gap=True), flush=True)

mcl_solver = MClSolver()
mcl_solver.from_txt("test_dataset/mcl/mcl_rb-giant_gurobi-3600s_81.520.txt", ref=True)
mcl_solver.from_txt("ml4co_result/mcl/mcl_rb-giant_ml4co_85.380.txt", ref=False, cover=False)
print("\"MCl-RB-GIANT\":", mcl_solver.evaluate(calculate_gap=True), flush=True)


###############################
#             MVC             #
###############################

mvc_solver = MVCSolver()
mvc_solver.from_txt("test_dataset/mvc/mvc_rb-small_gurobi-60s_205.764.txt", ref=True)
mvc_solver.from_txt("ml4co_result/mvc/mvc_rb-small_ml4co_205.772.txt", ref=False, cover=False)
print("\"MVC-RB-SMALL\":", mvc_solver.evaluate(calculate_gap=True), flush=True)

mvc_solver = MVCSolver()
mvc_solver.from_txt("test_dataset/mvc/mvc_rb-large_gurobi-300s_968.228.txt", ref=True)
mvc_solver.from_txt("ml4co_result/mvc/mvc_rb-large_ml4co_968.398.txt", ref=False, cover=False)
print("\"MVC-RB-LARGE\":", mvc_solver.evaluate(calculate_gap=True), flush=True)

mvc_solver = MVCSolver()
mvc_solver.from_txt("test_dataset/mvc/mvc_twitter_gurobi-60s_85.251.txt", ref=True)
mvc_solver.from_txt("ml4co_result/mvc/mvc_twitter_ml4co_85.251.txt", ref=False, cover=False)
print("\"MVC-TWITTER\":", mvc_solver.evaluate(calculate_gap=True), flush=True)

mvc_solver = MVCSolver()
mvc_solver.from_txt("test_dataset/mvc/mvc_collab_gurobi-60s_65.086.txt", ref=True)
mvc_solver.from_txt("ml4co_result/mvc/mvc_collab_ml4co_65.086.txt", ref=False, cover=False)
print("\"MVC-COLLAB\":", mvc_solver.evaluate(calculate_gap=True), flush=True)

mvc_solver = MVCSolver()
mvc_solver.from_txt("test_dataset/mvc/mvc_rb-giant_gurobi-3600s_2396.780.txt", ref=True)
mvc_solver.from_txt("ml4co_result/mvc/mvc_rb-giant_ml4co_2397.360.txt", ref=False, cover=False)
print("\"MVC-RB-GIANT\":", mvc_solver.evaluate(calculate_gap=True), flush=True)


###############################
#            MCut             #
###############################

mcut_solver = MCutSolver()
mcut_solver.from_txt("test_dataset/mcut/mcut_ba-small_gurobi-60s_727.844.txt", ref=True)
mcut_solver.from_txt("ml4co_result/mcut/mcut_ba-small_ml4co_729.706.txt", ref=False, cover=False)
print("\"MCut-BA-SMALL\":", mcut_solver.evaluate(calculate_gap=True), flush=True)

mcut_solver = MCutSolver()
mcut_solver.from_txt("test_dataset/mcut/mcut_ba-large_gurobi-300s_2936.886.txt", ref=True)
mcut_solver.from_txt("ml4co_result/mcut/mcut_ba-large_ml4co_2994.118.txt", ref=False, cover=False)
print("\"MCut-BA-SMALL\":", mcut_solver.evaluate(calculate_gap=True), flush=True)

mcut_solver = MCutSolver()
mcut_solver.from_txt("test_dataset/mcut/mcut_ba-giant_gurobi-3600s_7217.900.txt", ref=True)
mcut_solver.from_txt("ml4co_result/mcut/mcut_ba-giant_ml4co_7389.300.txt", ref=False, cover=False)
print("\"MCut-BA-SMALL\":", mcut_solver.evaluate(calculate_gap=True), flush=True)