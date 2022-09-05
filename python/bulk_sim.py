import pandas as pd
import pandapower.networks as pn
import data_process
import importlib
import numpy as np

importlib.reload(data_process)
da = data_process.DataAction()
nc = data_process.net_calc()


def initiate():
    """general import stuff"""

    da.folder_path = r"G:\\My Drive\\docs\\Education\\University\\ALUF\\SSE\\6 sem\\Bachelor Thesis\\data\\"
    keep_cols = [
        "DE_KN_residential1_grid_import",
        "DE_KN_residential2_grid_import",
        "utc_timestamp",
    ]
    da.imp_procc("house_data.csv", keep_cols)

    # prepare network
    net = pn.ieee_european_lv_asymmetric("off_peak_1440")
    net.bus.vn_kv = net.bus.vn_kv / np.sqrt(3)

    # set asymmetric loads to zero
    net.asymmetric_load.p_a_mw = 0
    net.asymmetric_load.p_b_mw = 0
    net.asymmetric_load.p_c_mw = 0
    net.asymmetric_load.q_a_mvar = 0
    net.asymmetric_load.q_b_mvar = 0
    net.asymmetric_load.q_c_mvar = 0

    return net


def iterate(net, type):
    """common run"""

    sgen_val = da.sgen_val
    scaling_fac = 1
    nmbr_loads = net.asymmetric_load.shape[0]

    da.load_sgen_make(nmbr_loads)

    if type == "random":
        print("Going random!")
        da.sgen_rand(sgen_val)
    elif type == "community":
        print("Going community!")
        start_times = da.get_start_times(nmbr_loads)
        da.sgen_comm(start_times, sgen_val)
    elif type == "control":
        print("Going control!")
        pass
    else:
        print("I'm confused by your input.")

    nc.net_asym_prep(net, da.night_loads * scaling_fac, da.night_sgens)
    nc.output_writer("res_bus", "vm_pu")
    nc.ts_run()


def simulate(cycles, scenario):
    """main loop function"""

    min_vm_list = []
    min_time_list = []
    for i in range(cycles):
        while True:
            try:
                net = initiate()
                iterate(net, scenario)
                break
            except:
                print("whoops...")

        vm_pu = nc.read_output()
        min_stats, max_stats = nc.vm_stats(vm_pu)
        min_vm_list.append(pd.Series(min_stats[0], name="min load val"))
        min_time_list.append(pd.Series(max_stats[2], name="min load time"))

        print("List entry added.")

    res1 = pd.concat((min_vm_list))
    res2 = pd.concat((min_time_list))
    res = pd.concat((res1, res2), axis=1)
    res.index = range(len(res))

    print("Results created.")

    return res


# run simulation
print("Beginnig simualation...")
cycles = 100
path = "..\\results\\"

res1 = simulate(cycles, "random")
name = "random_scenario_sim.xlsx"
full = path + name
res1.to_excel(full)
print("All done with random stuff.")

res2 = simulate(cycles, "community")
name = "ener_comm_sim.xlsx"
full = path + name
res2.to_excel(full)
print("All done with community stuff.")

res3 = simulate(cycles, "control")
name = "control_scenario_sim.xlsx"
full = path + name
res3.to_excel(full)
print("All done with control stuff.")

print("Simulations complete!")
