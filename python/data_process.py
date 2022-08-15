import os
import random
import pandas as pd
import numpy as np
from datetime import timedelta
import pandapower as pp
import pandapower.networks as pn
import pandapower.plotting as plot
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control import ConstControl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

matplotlib.rcParams["timezone"] = "Europe/Berlin"


class DataAction:
    """
    This class contains the most common data processing functions specifically for
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
        self.conv_fac = 1000 * 60  # convert from MW/min to W (1 min avg)
        self.night_evening_t = "18:00:00"
        self.night_morning_t = "06:00:00"
        self.night_max_t = "19:19:00"  # obtained from max_load_times.ipynb
        self.wind_length = 60
        self.night_loads = None
        self.night_sgens = None

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
        print("Number of data frame segments = ", len(self.dfList))

    def imp_procc(self, file_name, keep_cols):
        """all-in-one import processing helper function"""

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
        end = bar.replace(  # replace time section of string
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

    def time_wind(self, ts, wind_length):
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
        df_rand = np.random.choice(len(self.dfList[:-1]))  # incomplete list excluded

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
        """write sgen value to df on column across a given time window"""

        ts.loc[start:end, col_name] = val  # 'start' & 'end' variables are str

        return ts

    def sgen_rand(self, sgen_val):
        """fill self.night_sgens df at random times with select sgen value"""

        sgens = self.night_sgens.columns
        for name in sgens:
            # writes directly to night_mw
            start, end = self.time_wind(self.night_sgens, self.wind_length)
            self.sgen_write(self.night_sgens, start, end, name, sgen_val)

    def sgen_max(self, sgen_val):
        """fill self.night_sgens df at single peak load time for all households"""

        # calculate end window time based on max load time
        foo = self.night_sgens.index[0]
        bar = foo.replace(  # replace time section of string
            hour=int(self.night_max_t[0:2]),
            minute=int(self.night_max_t[3:5]),
            second=int(self.night_max_t[6:8]),
        )
        start = bar.strftime("%Y-%m-%d %H:%M:%S")
        end = (bar + timedelta(minutes=self.wind_length)).strftime("%Y-%m-%d %H:%M:%S")

        # fill all columns with same value
        self.night_sgens.loc[start:end] = sgen_val


class net_calc:
    """
    This class is used in conjunction with the DataAction class. It contains functions for time series
    simulations using the pandapower module. The main goal is to run the time series iteration using
    preset controllers and settings.

    Further data analysis and results can be found in external python files.

    Written by Daniil Akthonka for his Bachelor thesis:
    'Electric Vehicles in Energy Communities: Investigating the Distribution Grid Hosting Capacity'
    """

    def __init__(self):
        """ititialization of class variables"""

        self.net = None
        self.night_mw = None
        self.n_timesteps = None
        self.time_steps = None
        self.ll = None  # line loading results df

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

    # def load_graph(self, net, time_step): # helper function
    #     # update network with step value
    #     run_timeseries(net, time_steps=(0, time_step))

    def end_vals_step(self, ll, end_vals):  # helper function
        """append max 'll' df values to 'end_val' df"""

        # get inputs from df
        max_val = ll.max()

        # append series as last line
        end_vals.loc[end_vals.shape[0]] = max_val

    def end_times_step(self, ll, end_times):  # helper function
        """append times of max values in 'll' df to 'end_times' df"""

        # get inputs
        max_ind = ll.idxmax()
        k = self.night_mw.index.values[max_ind.tolist()]
        max_time = pd.to_datetime(k).strftime("%H:%M:%S").tolist()

        # append to end_times
        end_times.loc[end_times.shape[0]] = max_time

