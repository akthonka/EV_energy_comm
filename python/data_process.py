import pandas as pd
import numpy as np
from datetime import timedelta



class DataAction:
  """
  The nuclear data-processing method.
  
  """

  def __init__(self):
    self.imp = None
    self.df = None
    self.chunk_size = 10000
    self.dfList = []
    self.night_evening_t = '20:00:00'
    self.night_morning_t = '06:00:00'
    self.night_mw = None


  def data_imp(self, file_name):
    # import from data location
    folder_path = '..\\data\\'
    file_path = folder_path + file_name
    self.imp = pd.read_csv(file_path, low_memory = False)


  def data_filter(self, df, keep_cols):
    # Process df inplace
    col_names = df.columns.tolist()
    for col in keep_cols:
        col_names.remove(col)
    df.drop(columns=col_names, inplace = True)
    df.dropna(inplace=True)
    df.rename(columns = {'cet_cest_timestamp':'date_time'}, inplace = True)
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
    df.index = pd.to_datetime(df.index, exact=True, cache=True, format='%Y-%m-%d %H:%M:%S', dayfirst=True)
    ts = df.diff()
    ts.drop(str(ts.index[0]), inplace=True) # drop first NaN row
    return ts


  def unique_date(self, df): # helper function
    # find unique days in the time-series index df
    return df.index.map(lambda t: str(t.date())).unique().tolist()


  def get_night(self, ts, evening_date, start_time, end_time):
    # get the dates for loc slice
    start = evening_date + ' ' + start_time
    foo = pd.to_datetime(evening_date)
    bar = foo.replace(hour=int(end_time[0:2]), minute=int(end_time[3:5]),
                      second=int(end_time[6:8])).strftime('%H:%M:%S')
    end = str(foo + timedelta(days=1))[:11] + bar

    # check for available date
    dates = self.unique_date(ts)
    if evening_date in dates:
        # print("Date is accepted.")
        return ts.loc[start : end]
    else:
        print("Error: Evening_date is not part of the selected dataset!")


  def time_wind(self, ts, wind_length):
    # select random night time window of length wind_length
    length = len(ts.index) - wind_length
    time_0 = np.random.randint(0,length)
    foo = ts.index[time_0]

    start = foo.strftime('%Y-%m-%d %H:%M:%S')
    bar = foo + timedelta(minutes=wind_length)
    end = bar.strftime('%Y-%m-%d %H:%M:%S')
    # print("Start time: ", start)
    # print("End time: ", end)
    
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
          df_rand = np.random.choice(len(self.dfList), 2, replace=False)
          ts1 = self.parse_procc(self.dfList[df_rand[0]])
          ts2 = self.parse_procc(self.dfList[df_rand[1]])

          # test that they are the same size then merge
          if ts1.size == ts2.size:            
              # limit TS to night window
              night1 = self.get_night(ts1, self.unique_date(ts1)[0],
                                      self.night_evening_t, self.night_morning_t).copy()
              night2 = self.get_night(ts2, self.unique_date(ts2)[0],
                                      self.night_evening_t, self.night_morning_t).copy()
              night2.index = night1.index # use the first index for both, since only the time matters

              # rename cols
              names1 = {'DE_KN_residential1_grid_import': 'load_1',
                      'DE_KN_residential2_grid_import': 'load_2'}
              night1.rename(columns = names1, inplace=True)
              names2 = {'DE_KN_residential1_grid_import': 'load_3',
                      'DE_KN_residential2_grid_import': 'load_4'}
              night2.rename(columns = names2, inplace=True)

              # convert units to W (avg value over a minute)
              factor = 1000*60*(20)
              night1 = night1*factor
              night2 = night2*factor
              night_merge = night1.join(night2)
              
              return night_merge

      except:
          print("Whoops, grabbing a different touple of dates...")


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