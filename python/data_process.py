import os
import pandas as pd
import numpy as np
from datetime import timedelta
import pandapower as pp
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control import ConstControl
from pandapower.plotting.plotly import pf_res_plotly
from itertools import cycle


class DataAction:
    """
    This DataAction class contains the most common data processing functions specifically for
    working with the Open Power System Household Data sets. The main goal is to process
    raw data using pandas module in preparation for computation with pandapower module.

    The cricital data processing steps include:
    - file import
    - initial raw data filtering and formatting
    - splitting of imported data for piece-wise processing
    - datetime index parsing

    This class includes several helper functions for common actions, such as unique date
    identification for data spanning several days.

    Lastly, composite functions for heavy data manipulation are written here. These concern
    themselves with generation of test data sets for pandapower simulations - in particular,
    the timeseries iterations.

    Written by Daniil Akthonka for his Bachelor thesis:
    'Electric Vehicles in Energy Communities: Investigating the Distribution Grid Hosting Capacity'
    """

    def __init__(self):
        """ititialization of class variables"""

        self.folder_path = None
        self.imp = None
        self.df = None  # comment abbreviation from hereon: df = DataFrame
        self.chunk_size = 10000  # number of datapoints in each df
        self.dfList = []
        self.conv_fac = 1000 * 60  # convert from kW/min to W (1 min avg)
        self.night_evening_t = "18:00:00"
        self.night_morning_t = "06:00:00"
        self.night_max_t = "19:23:00"  # obtained from max_load_times.ipynb
        self.night_min_t = "02:26:00"  # obtained from max_load_times.ipynb
        self.wind_length = 60  # in minutes
        self.iter_time = None
        self.night_loads = None
        self.night_sgens = None
        self.sgen_val = 0.0077  # Level 2 7.7kW AC charger

    def data_imp(self, file_name):
        """basic data import from file string"""

        folder_path = self.folder_path
        file_path = folder_path + file_name
        self.imp = pd.read_csv(file_path, low_memory=False)

    def data_filter(self, df, keep_cols):
        """initial raw data filtering and formatting"""

        col_names = df.columns.tolist()
        for col in keep_cols:
            col_names.remove(col)
        df.drop(columns=col_names, inplace=True)
        df.dropna(inplace=True)
        df.rename(columns={"utc_timestamp": "date_time"}, inplace=True)
        self.df = df.set_index("date_time")

    def df_split(self, chunk_size):
        """fragmentation of import data into smaller dfs"""

        for i in range(0, self.df.shape[0], chunk_size):
            self.dfList.append(self.df[i : i + chunk_size])

    def imp_procc(self, file_name, keep_cols):
        """all-in-one import processing helper function"""
        self.dfList = []

        self.data_imp(file_name)
        self.data_filter(self.imp, keep_cols)
        self.df_split(self.chunk_size)
        print("dfList created successfully.")

    def parse_procc(self, df):
        """parsing of datetime for minute-wise energy differences"""

        df.index = pd.to_datetime(
            df.index,
            exact=True,
            cache=True,
            format="%Y-%m-%d %H:%M:%S",
            dayfirst=True,
            utc=True,
        )  # convert index to datetime
        df = df.tz_convert("Europe/Berlin")  # match region data
        ts = df.diff()  # obtain minute-wise energy changes
        ts.dropna(inplace=True)  # drops only the first row, from the diff()

        return ts

    def unique_date(self, df):  # helper function
        """find unique days in the time-series index df"""

        return df.index.map(lambda t: str(t.date())).unique().tolist()

    def get_night(self, ts, evening_date):
        """return overnight time-delimited df data slice for evening date"""

        # get evening and morning datetime limits
        start = evening_date + " " + self.night_evening_t
        foo = pd.to_datetime(evening_date)
        bar = foo + timedelta(days=1)
        end = bar.replace(
            hour=int(self.night_morning_t[0:2]),
            minute=int(self.night_morning_t[3:5]),
            second=int(self.night_morning_t[6:8]),
        ).strftime("%Y-%m-%d %H:%M:%S")

        # check for available date and return data
        dates = self.unique_date(ts)
        if evening_date in dates:
            return ts.loc[start:end]
        else:
            print("Error: Evening_date is not part of the selected dataset!")

    def time_wind(self, ts, wind_length):  # helper function for sgen_rand
        """select random night time window for a single party (household)"""

        length = len(ts.index) - wind_length + 1  # +1 due to zero based indexing
        time_0 = np.random.randint(0, length)
        foo = ts.index[time_0]
        start = foo.strftime("%Y-%m-%d %H:%M:%S")
        bar = foo + timedelta(minutes=wind_length)
        end = bar.strftime("%Y-%m-%d %H:%M:%S")

        return start, end

    def night_rand(self):
        """create a random night load profile"""

        # choose random df identifying list number from fragemented import set
        df_rand = np.random.choice(len(self.dfList[1:-1]))  # incomplete lists excluded

        # choose random load profile (between two) and parse selected data
        rand_col = np.random.randint(0, 2)
        ts1 = self.parse_procc(self.dfList[df_rand].iloc[:, rand_col])

        # limit df to a random night in data set
        date1 = np.random.choice(self.unique_date(ts1)[1:-2])
        night1 = self.get_night(ts1, date1).copy()

        return night1

    def load_sgen_make(self, load_number=55):
        """create pandapower time series simulation df's for any number of households"""

        # initialize targets
        night_loads = pd.DataFrame()  # output df
        list_loads = []  # helper list
        night_sgens = pd.DataFrame()  # output df
        list_sgens = []  # helper list

        # generate list of Series for concatenation
        for i in range(1, load_number + 1):
            night1 = self.night_rand()
            list_loads.append(pd.Series(night1.values, name="loadh_" + str(i)))
            list_sgens.append(pd.Series([0] * night1.shape[0], name="sgen_" + str(i)))

        # merge series into output df's
        night_loads = pd.concat(list_loads, axis=1) * self.conv_fac / 1000000  # to MW
        night_loads.index = night1.index
        night_sgens = pd.concat(list_sgens, axis=1)
        night_sgens.index = night1.index

        # write output to class variables for later use
        self.night_loads = night_loads
        self.night_sgens = night_sgens

    def sgen_write(self, ts, start, end, col_name, val):  # helper function
        """write sgen value to df on column across a given time window
        Note: can't overwrite filled time-slots! Empty those first."""

        ts.loc[start:end, col_name] = -val  # 'start' & 'end' variables are str

        return ts

    def sgen_rand(self, sgen_val):
        """fill self.night_sgens df at random times with select sgen value"""

        # reset existing values (else won't overwrite)
        self.night_sgens[:] = 0

        sgens = self.night_sgens.columns
        for name in sgens:
            # writes directly to night_mw
            start, end = self.time_wind(self.night_sgens, self.wind_length)
            self.sgen_write(self.night_sgens, start, end, name, sgen_val)

    def get_start_times(self, nmbr_sgens):  # helper function for sgen_comm()
        """make list of start times to cycle through based on number of sgens and time window"""

        time_pt = self.night_sgens.index[0]  # starting time point

        # get stop time with date
        foo = time_pt + timedelta(days=1)
        bar = foo.replace(
            hour=int(self.night_morning_t[0:2]),
            minute=int(self.night_morning_t[3:5]),
            second=int(self.night_morning_t[6:8]),
        )
        stop_time = bar + timedelta(minutes=self.wind_length)  # because of loop -1 min

        start_times = [time_pt.strftime("%Y-%m-%d %H:%M:%S")]
        for hour in range(nmbr_sgens):
            # perform window time step
            time_pt = pd.to_datetime(time_pt) + timedelta(minutes=self.wind_length)
            if time_pt < stop_time:
                start_times.append(time_pt.strftime("%Y-%m-%d %H:%M:%S"))
            else:
                break

        return start_times

    def sgen_comm(self, start_times, val):
        """fill self.night_sgens df consequently/cyclically based on start_times"""

        # reset existing values (else won't overwrite)
        self.night_sgens[:] = 0

        # initiate cycle and reset starting point
        time_cycle = cycle(start_times)
        self.iter_time = next(time_cycle)

        for col_name in self.night_sgens.columns.tolist():

            # starting time val
            foo = next(time_cycle)
            foo_dt = pd.to_datetime(foo)

            if foo_dt.strftime("%H:%M:%S") == self.night_evening_t:
                # special case: first interval
                start = foo_dt.strftime("%Y-%m-%d %H:%M:%S")

                # ending time val - 1 minute
                bar = (
                    foo_dt + timedelta(minutes=self.wind_length) - timedelta(minutes=1)
                ).strftime("%Y-%m-%d %H:%M:%S")

                # write gen value
                self.sgen_write(self.night_sgens, start, bar, col_name, val)
                foo = next(time_cycle)
            else:
                if foo_dt.strftime("%H:%M:%S") == self.night_morning_t:
                    # last minute is filled
                    bar = foo_dt.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    # ending time val - 1 minute
                    bar = (foo_dt - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
                # write gen value
                self.sgen_write(self.night_sgens, self.iter_time, bar, col_name, val)

            # update first time for next iter
            self.iter_time = foo


class net_calc:
    """
    This NetworkCalculation class is used in conjunction with the DataAction class. It contains
    functions for time series simulations using the pandapower module. The main goal is to run the
    time series iteration using preset controllers and settings.

    Further data analysis and results can be found in external jupyter notebook files.

    Written by Daniil Akthonka for his Bachelor thesis:
    'Electric Vehicles in Energy Communities: Investigating the Distribution Grid Hosting Capacity'
    """

    def __init__(self):
        """ititialization of class variables"""

        self.net = None
        self.night_mw = None
        self.n_timesteps = None
        self.time_steps = None
        self.iter_time = None  # start time cycle helper var

    def net_asym_prep(self, net, night_loads, night_sgens):
        """prepare an asymmetric load network for time series iteration"""

        # passed network will be stored as a class variable
        self.net = net

        # create load and sgen columns at asymmetric load bus locations
        for i in range(0, len(net.asymmetric_load)):
            bus_nmbr = net.asymmetric_load.bus.at[i]
            load_name = "loadh_" + str(i)
            sgen_name = "sgen_" + str(i)
            pp.create_load(net, bus_nmbr, 0, name=load_name)
            pp.create_sgen(net, bus_nmbr, 0, name=sgen_name)

        # create dataset copy w/ index for timeseries
        night_loads_ts = night_loads.copy()
        night_loads_ts.index = range(0, night_loads.shape[0])
        night_sgens_ts = night_sgens.copy()
        night_sgens_ts.index = range(0, night_sgens.shape[0])

        # create load controller
        ds_loads = DFData(night_loads_ts)
        ConstControl(
            net,
            element="load",
            variable="p_mw",
            element_index=net.load.index,
            profile_name=night_loads_ts.columns.tolist(),
            data_source=ds_loads,
        )

        # create sgen controller
        ds_sgens = DFData(night_sgens_ts)
        ConstControl(
            net,
            element="sgen",
            variable="p_mw",
            element_index=net.sgen.index,
            profile_name=night_sgens_ts.columns.tolist(),
            data_source=ds_sgens,
        )

        # note the time series iteration step variables
        self.n_timesteps = night_loads_ts.shape[0]
        self.time_steps = range(0, self.n_timesteps)

    def output_writer(self, var, index):
        """create output writer to store results"""

        path = "..\\results\\"  # one folder up the file tree
        ow = OutputWriter(
            self.net,
            time_steps=self.time_steps,
            output_path=path,
            output_file_type=".xlsx",
        )
        ow.log_variable(var, index)

    def ts_run(self):
        """run the time series iteration"""

        run_timeseries(self.net, time_steps=self.time_steps)

    def read_output(self):
        """load the saved output files as df for further processing"""

        # load excel file
        path = "..\\results\\"
        vm_file = os.path.join(path, "res_bus", "vm_pu.xlsx")
        vm_pu = pd.read_excel(vm_file, index_col=0)

        # create renaming dictionary
        line_dict = {}
        keys = vm_pu.columns.tolist()
        values = []
        for i in range(vm_pu.shape[1]):
            line_name = "bus_" + str(i)
            values.append(line_name)

        for i in range(vm_pu.shape[1]):
            line_dict[keys[i]] = values[i]

        # rename cols
        vm_pu.rename(columns=line_dict, inplace=True)

        return vm_pu

    def load_graph(self, time_step):  # helper function
        """update network with step value"""

        run_timeseries(self.net, time_steps=(0, time_step))

    def vm_stats(self, vm_pu):
        """return key timeseries result values for minima"""

        min_min_ind = (
            vm_pu.idxmin().unique()[1:].min()
        )  # first is always zero, from grid
        min_min_vm = round(vm_pu.min().min(), 5)  # min voltage across all busses
        min_time = (
            pd.to_datetime(DataAction().night_evening_t)
            + timedelta(minutes=int(min_min_ind))
        ).strftime(
            "%H:%M:%S"
        )  # time of min_min_ind

        max_max_ind = (
            vm_pu.idxmax().unique()[1:].max()
        )  # first is always zero, from grid
        max_max_vm = round(vm_pu.max().max(), 5)  # min voltage across all busses
        max_time = (
            pd.to_datetime(DataAction().night_evening_t)
            + timedelta(minutes=int(max_max_ind))
        ).strftime(
            "%H:%M:%S"
        )  # time of max_max_ind
        print("All-time min and max:", min_min_vm, ";", max_max_vm)
        # print("All-time min-load time across all busses:", min_min_ind, ",", min_time)

        return (min_min_vm, min_min_ind, min_time), (max_max_vm, max_max_ind, max_time)

    def plotly_res(self):  # helper function
        """return a network diagram of results"""

        x = pf_res_plotly(self.net, climits_volt=(0.95, 1.05))  # x is arbitrary

    def hosting_cap(self, sgen_val):
        """compute hosting capacity"""

        # reset network sgen values
        for sgen in self.net.sgen.name.tolist():
            self.net.sgen.p_mw = 0

        # create random sgen name list to populate loads
        rand_ind = np.arange(len(self.net.sgen.name.tolist()))
        np.random.shuffle(rand_ind)

        hosting_cap = 0
        for i in rand_ind:
            # compute power flow
            pp.runpp(self.net)
            min_vm_pm = self.net.res_bus.vm_pu.min()
            max_line_load = self.net.res_line.loading_percent.max()

            # conditional iteration step
            if min_vm_pm > 0.95 and max_line_load < 100:
                # set single sgen value and run power flow
                self.net.sgen.p_mw.at[i] = -sgen_val
                hosting_cap = hosting_cap + 1
            else:  # since last step exceeded limit, undo it
                hosting_cap = hosting_cap - 1
                self.net.sgen.p_mw.at[rand_ind[hosting_cap]] = 0
                print("Max hosting capacity:", hosting_cap, "out of", len(rand_ind))
                break

        host_pct = round((hosting_cap / len(rand_ind) * 100), 1)

        pp.runpp(self.net)
        self.plotly_res()

        return hosting_cap, host_pct
