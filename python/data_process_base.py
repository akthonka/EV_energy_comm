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
matplotlib.rcParams['timezone'] = 'Europe/Berlin'



class DataAction:
  """
  The nuclear data-processing method.
  
  """

  def __init__(self):
    self.imp = None
    self.df = None
    self.chunk_size = 10000
    self.dfList = []
    self.conv_fac = 1000*60*(15) # convert to W (1 min avg) 
    self.night_evening_t = '20:00:00'
    self.night_morning_t = '06:00:00'
    self.night_mw = None


  def data_imp(self, file_name):
    # import from data location
    folder_path = r'G:\My Drive\docs\Education\University\ALUF\SSE\6 sem\Bachelor Thesis\\data\\'
    file_path = folder_path + file_name
    self.imp = pd.read_csv(file_path, low_memory = False)


  def data_filter(self, df, keep_cols):
    # Process df inplace
    col_names = df.columns.tolist()
    for col in keep_cols:
        col_names.remove(col)
    df.drop(columns=col_names, inplace = True)
    df.dropna(inplace=True)
    df.rename(columns = {'utc_timestamp':'date_time'}, inplace = True)
    self.df = df.set_index('date_time')


  def df_split(self, chunk_size):
    # splits the dataframe into smaller dataframes
    for i in range(0, self.df.shape[0], chunk_size):
        self.dfList.append(self.df[i:i+chunk_size])
    print('Number of data frame segments = ', len(self.dfList))


  def imp_procc(self, file_name, keep_cols):
    # easy import, data filter and split
    self.data_imp(file_name)
    self.data_filter(self.imp, keep_cols)
    self.df_split(self.chunk_size)
    print("dfList created successfully.")


  def parse_procc(self, df):
    # parse dates and convert index to time series
    # take the row difference and drop first NaN
    df.index = pd.to_datetime(df.index, exact=True, cache=True, format='%Y-%m-%d %H:%M:%S', dayfirst=True, utc=True)
    df = df.tz_convert('Europe/Berlin')
    ts = df.diff()
    ts.dropna(inplace=True) # drops only the first row, from the diff()
    return ts


  def unique_date(self, df): # helper function
    # find unique days in the time-series index df
    return df.index.map(lambda t: str(t.date())).unique().tolist()


  def get_night(self, ts, evening_date):
    # get the dates for loc slice
    start = evening_date + ' ' + self.night_evening_t
    foo = pd.to_datetime(evening_date)
    bar = foo + timedelta(days=1)
    end = bar.replace(hour=int(self.night_morning_t[0:2]), minute=int(self.night_morning_t[3:5]),
                      second=int(self.night_morning_t[6:8])).strftime('%Y-%m-%d %H:%M:%S')

    # check for available date
    dates = self.unique_date(ts)
    if evening_date in dates:
        return ts.loc[start : end]
    else:
        print("Error: Evening_date is not part of the selected dataset!")


  def time_wind(self, ts, wind_length, parties=1):
    # select random night time window of length wind_length
    length = len(ts.index) - wind_length*parties + 1 # due to zero based indexing
    time_0 = np.random.randint(0,length)
    foo = ts.index[time_0]

    start = foo.strftime('%Y-%m-%d %H:%M:%S')
    bar = foo + timedelta(minutes=wind_length)
    end = bar.strftime('%Y-%m-%d %H:%M:%S')

    return start, end


  def sgen_write(self, ts, start, end, col_name, val):
    # write sgen val to df on col in time window
    ts.loc[start : end, col_name] = val
    return ts


  def power_merge(self):
    """
    Randomly select two days and merge them into a single df

    """

    while True:
      try:
          # pick two random TimeSeries
          df_rand = np.random.choice(len(self.dfList[:-1]), 2, replace=False)
          ts1 = self.parse_procc(self.dfList[df_rand[0]])
          ts2 = self.parse_procc(self.dfList[df_rand[1]])

          # test that they are the same size then merge
          if ts1.size == ts2.size:            
              # limit TS to random night window (except for last one, for night over-roll)
              date1 = np.random.choice(self.unique_date(ts1)[2:-2])
              date2 = np.random.choice(self.unique_date(ts2)[2:-2])
              night1 = self.get_night(ts1, date1).copy()
              night2 = self.get_night(ts2, date2).copy()
              night2.index = night1.index # use the first index for both, since only the time matters

              # rename cols
              names1 = {'DE_KN_residential1_grid_import': 'load_1',
                      'DE_KN_residential2_grid_import': 'load_2'}
              night1.rename(columns = names1, inplace=True)
              names2 = {'DE_KN_residential1_grid_import': 'load_3',
                      'DE_KN_residential2_grid_import': 'load_4'}
              night2.rename(columns = names2, inplace=True)

              # convert units to W (avg value over a minute)
              night1 = night1*self.conv_fac
              night2 = night2*self.conv_fac
              night_merge = night1.join(night2)
              
              return night_merge

      except Exception as str_error: # deprecated error handling - should now be fixed via parameters
          print(str_error) 
          print("Huh, grabbing a different touple of dates...")


  def sgen_rand(self, ts, sgen_val):
    """
    Write sgen vals over random time window for cols

    """
    
    # create mw night dataset copy
    night_mw = ts.copy()/1000000   # convert to MW

    # create sgen columns
    night_mw.insert(1, 'sgen_1', 0)
    night_mw.insert(3, 'sgen_2', 0)
    night_mw.insert(5, 'sgen_3', 0)
    night_mw.insert(7, 'sgen_4', 0)

    sgens = ['sgen_1','sgen_2','sgen_3','sgen_4']
    val = sgen_val # 0.010 or 10kW is typical high-end

    for i in sgens:
        # writes directly to night_mw
        start, end = self.time_wind(night_mw, 60)
        self.sgen_write(night_mw, start, end, i, val)
    
    return night_mw


  def sgen_comm(self, ts, wind_length, sgen_val, parties):
    """
    Write sgen vals over random time window for cols

    """

    # create mw night dataset copy
    night_mw = ts.copy()/1000000   # convert to MW

    # create sgen columns
    night_mw.insert(1, 'sgen_1', 0)
    night_mw.insert(3, 'sgen_2', 0)
    night_mw.insert(5, 'sgen_3', 0)
    night_mw.insert(7, 'sgen_4', 0)
    all_sgens = ['sgen_1','sgen_2','sgen_3','sgen_4']
    
    # create list of active sgens
    sgens = random.sample(all_sgens, parties)

    # get random start time (w/ respect to nmbr of parties)
    start_og, _ = self.time_wind(night_mw, wind_length, parties)
    start = start_og
    
    np.random.shuffle(sgens)
    wind_length = wind_length-1 # due to zero based index  
    for i in sgens:
        # fill sgen columns with sgen_val
        foo = pd.to_datetime(start)
        bar = foo + timedelta(minutes=wind_length)
        end = bar.strftime('%Y-%m-%d %H:%M:%S')
        self.sgen_write(night_mw, start, end, i, sgen_val)

        # update new start value with old one + 1 min
        next = bar + timedelta(minutes=1)
        start = next.strftime('%Y-%m-%d %H:%M:%S')

    self.night_mw = night_mw
    
    return night_mw



class net_calc:
  """
  Automation of net-specific pandapower computations
  
  """

  def __init__(self):
    self.net = None
    self.night_mw = None
    self.n_timesteps = None
    self.time_steps = None
    self.ll = None # line loading results df
    

  def four_loads_branched_make(self, night_mw):
    # create net and assign load names
    net = pn.four_loads_with_branches_out()
    pp.create_sgen(net, 6, p_mw=0, name='sgen_1', q_mvar=0)
    pp.create_sgen(net, 7, p_mw=0, name='sgen_2', q_mvar=0)
    pp.create_sgen(net, 8, p_mw=0, name='sgen_3', q_mvar=0)
    pp.create_sgen(net, 9, p_mw=0, name='sgen_4', q_mvar=0)
    net.load.name.at[0] = "load_1"
    net.load.name.at[1] = "load_2"
    net.load.name.at[2] = "load_3"
    net.load.name.at[3] = "load_4"

    # create dataset copy w/ index for timeseries
    night_ts = night_mw.copy()
    night_ts.index = range(0, night_ts.shape[0])

    # create controllers
    ds = DFData(night_ts)
    ConstControl(net, element="sgen", variable="p_mw", element_index=net.sgen.index,
                profile_name=["sgen_1","sgen_2","sgen_3","sgen_4"], data_source=ds)
    ConstControl(net, element="load", variable="p_mw", element_index=net.load.index,
                profile_name=["load_1","load_2","load_3","load_4"], data_source=ds)

    # save network, ts (future ref) and timesteps
    self.net = net
    self.night_mw = night_mw
    self.n_timesteps = night_mw.shape[0]
    self.time_steps = range(0, self.n_timesteps)

    
  def four_loads_branched_out(self, var, index):
    # create output writer to store results
    path = '..\\results\\'
    ow = OutputWriter(self.net, time_steps=self.time_steps, output_path=path, output_file_type=".xlsx")
    ow.log_variable(var, index)


  def four_loads_branched_run(self):
    # run timeseries calculation
    run_timeseries(self.net, time_steps=self.time_steps)


  def four_loads_branched_read_loadpct(self):
    # read output data
    path = '..\\results\\'
    ll_file = os.path.join(path, "res_line", "loading_percent.xlsx")
    line_loading = pd.read_excel(ll_file, index_col=0)
    line_loading.columns = line_loading.columns.astype('str')
    names1 = {'0': 'line_1', '1': 'line_2', '2': 'line_3','3': 'line_4','4': 'line_5',
              '5': 'line_6', '6': 'line_7', '7': 'line_8'}
    line_loading.rename(columns = names1, inplace=True)
    self.ll = line_loading


  def four_loads_branched_plot_linepct(self):
    # plot timestep loaded in
    fig, ax = plt.subplots(figsize=(15,10))
    hours = mdates.HourLocator(interval = 1)
    h_fmt = mdates.DateFormatter('%H:%M')

    ax.plot(self.night_mw.index, self.ll.values)
    ax.xaxis.set_major_locator(hours)
    ax.xaxis.set_major_formatter(h_fmt)
    fig.autofmt_xdate()

    secax = ax.twiny()
    secax.plot(self.ll.index, self.ll.values)

    ax.set_ylabel("line loading [%]")
    ax.set_xlabel("time")
    secax.set_xlabel('time step')
    ax.legend(self.ll.columns)

    plt.show()


  def load_graph(self, net, time_step):
    # update network with step value
    run_timeseries(net, time_steps=(0,time_step))

    # plot line loading graph
    cmap_list=[(20, "green"), (50, "yellow"), (60, "red")]
    cmap, norm = plot.cmap_continuous(cmap_list)
    lc = plot.create_line_collection(net, net.line.index, zorder=1, cmap=cmap, 
                                     norm=norm, linewidths=2, use_bus_geodata=True)
    plot.draw_collections([lc], figsize=(8,6))


  def end_vals_step(self, ll, end_vals):
    """
    append vals to end_val df
    
    """
    # get inputs from df
    max_val = ll.max()
    
    # append series as last line
    end_vals.loc[end_vals.shape[0]] = max_val


  def end_times_step(self, ll, end_times):
    """
    append vals to end_times df
    
    """
    # get inputs
    max_ind = ll.idxmax()
    k = self.night_mw.index.values[max_ind.tolist()]
    max_time = pd.to_datetime(k).strftime('%H:%M:%S').tolist()

    # append to end_times
    end_times.loc[end_times.shape[0]] = max_time



  
  


  


    