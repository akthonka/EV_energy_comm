import pandas as pd
import pandapower.networks as pn
from pandapower.plotting.plotly import pf_res_plotly
import data_process
import numpy as np
import importlib

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

    # set asymmetric loads to zero
    for asy_load in net.asymmetric_load.name.tolist():
        net.asymmetric_load.p_a_mw = 0
        net.asymmetric_load.p_b_mw = 0
        net.asymmetric_load.p_c_mw = 0
        net.asymmetric_load.q_a_mvar = 0
        net.asymmetric_load.q_b_mvar = 0
        net.asymmetric_load.q_c_mvar = 0

    return net


def iterate(net):
    """common run"""

    sgen_val = 0.010
    scaling_fac = 0.7  # without it, the network underloads
    nmbr_loads = net.asymmetric_load.shape[0]

    da.load_sgen_make(nmbr_loads)
    start_times = da.get_start_times(nmbr_loads)
    da.sgen_comm(start_times, sgen_val)
    nc.net_asym_prep(net, da.night_loads * scaling_fac, da.night_sgens)
    nc.output_writer("res_bus", "vm_pu")
    nc.ts_run()


def simulate(cycles):
    """main loop function"""

    min_vm_list = []
    min_time_list = []
    for i in range(cycles):
        while True:
            try:
                net = initiate()
                iterate(net)
                break
            except Exception as e:
                print(e)  # sometimes the run goes whoops
                print("whoops...")

        vm_pu = nc.read_output()
        min_min_vm, min_min_ind, min_time = nc.vm_stats(vm_pu)
        min_vm_list.append(pd.Series(min_min_vm, name="min load val"))
        min_time_list.append(pd.Series(min_time, name="min load time"))

    res1 = pd.concat((min_vm_list))
    res2 = pd.concat((min_time_list))
    res = pd.concat((res1, res2), axis=1)
    res.index = range(len(res))

    return res


# run simulation
cycles = 10
res = simulate(cycles)

# export results to excel
path = "..\\results\\"
name = "ener_comm_sim.xlsx"
full = path + name
res.to_excel(full)
print("All done now, check Excel file.")
