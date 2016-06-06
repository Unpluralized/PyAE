# coding: utf-8
import pandas as pd
from numpy import gradient as np_gradient
import ConfigParser
from numba import jit
import time
import pickle

config = ConfigParser.ConfigParser()
config.read('./pridb_filter_config.ini')


def apply_filters(df, name, **kwargs):
    """
    :param df:          Pandas dataframe with AE hits & features
    :param name:        lookup name (test id) in pridb_filter_config.ini for filter parameters
    :param kwargs:      ignore_time_amount:         default False. If True, ignores TimeMaxAmount in filter file
                        load_pickle:                default False. If True, load previously filtered file from pickle.
                                                    If False, simply load virgin dataset and apply filters.
                        write_pickle:               Write filtered dataframe to pickle in ./pridb_filters_pickles;
                                                    filename = name.pickle
    :return:            Dataframe with all filters applied to it.
    """
    reframe_time = config_getter('General', 'ReframeTime')
    counts_threshold = config_getter('General', 'ReflectionsCountsThreshold')
    load_pickle = kwargs.get('load_pickle', False)
    write_pickle = kwargs.get('write_pickle', False)

    # disable annoying warnings about Pandas Chained Assignment. There is no danger here.
    pd.options.mode.chained_assignment = None
    # for name in names:
    if load_pickle:
        try:
            df = pickle.load(open('./pridb_filters_pickles/%s.pickle' % name, 'rb'))
        except IOError:
            raise IOError('pridb_filters: load_pickle = True but no pickled dataset found.')
        else:
            print('pridb_filters.apply_filters: read %s hits in %s minutes from pickled file.'
                  % (str(len(df[df['TRAI'] > 0].index)), str(round(df['Time'].max() - df['Time'].min(), 2))))
            return df
    else:
        t_lower = config_getter(name, 'TimeLowerBound')
        load_upper = config_getter(name, 'LoadUpperBound')
        loading_rate = config_getter(name, 'TestStartLoadingRateCriterion')
        load_smoothing = config_getter(name, 'LoadSmoothing')

        """
            Order of operations is important! Remove_reflections needs intact sequence of TRAI values
        """
        df = time_filter(df, from_config=name, ignore_amount=kwargs.get('ignore_time_amount', False))
        if kwargs.get('label_cycle', False):
            df = label_load(df, name)
        # df = remove_reflections(df, counts_threshold=counts_threshold)

        if 'PA0' in df.columns:
            # Convert the analog reading to kN
            load_conversion_factor = config_getter(name, 'AnalogReadInTokN', 0.01)
            df['PA0'] = df['PA0'].apply(lambda p: p*load_conversion_factor)
        if 'PA1' in df.columns:
            displ_conversion_factor = config_getter(name, 'AnalogReadInTomm', 0.01)
            df['PA1'] = df['PA1'].apply(lambda p: p*displ_conversion_factor)

            if name.startswith('S'):
                # Filter out everything past maximum load for static tests
                df = static_maxload_cutoff(df, load=load_upper)
                df = min_positive_loading_rate(df, t_lower=t_lower, loading_rate=loading_rate)
            if name.startswith('D'):
                df = split_min_max_load(df)

        if load_smoothing:
            df['SetID'] = df.index
            df.set_index('Time', inplace=True, drop=False)
            df['PA0'].interpolate(method='nearest', limit_direction='both', inplace=True)
            df['PA0'] = df['PA0'].rolling(window=int(load_smoothing), center=True, min_periods=10).mean()
            df.set_index('SetID', inplace=True)

        df = energy_filter(df, from_config=name)
        df = amplitude_filter(df, from_config=name)
        df = duration_filter(df, from_config=True)
        df = hit_num_filter(df, from_config=name)

        # Convert time column to minutes
        df['Time'] = df['Time'] / 60.

        pd.options.mode.chained_assignment = 'warn'

        print('pridb_filters.apply_filters: left with %s hits in %s minutes after filtering.'
              % (str(len(df[df['TRAI'] > 0].index)), str(round(df['Time'].max()-df['Time'].min(), 2))))
        # print df[df['TRAI'] > 0].head()
        if 'PA0' in df.columns or 'PA1' in df.columns:
            if write_pickle:
                with open('./pridb_filters_pickles/%s.pickle' % name, 'wb') as f:
                    pickle.dump(df.dropna(subset=['TRAI', 'PA0'], how='all'), f)
            return df.dropna(subset=['TRAI', 'PA0'], how='all')
        else:
            if write_pickle:
                with open('./pridb_filters_pickles/%s.pickle' % name, 'wb') as f:
                    pickle.dump(df[df['TRAI'] > 0], f)
            return df[df['TRAI'] > 0]


def config_getter(section, option, *default):
    try:
        value = config.get(section, option).strip()
        try:
            return eval(value)
        except NameError:
            return value
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        if default:
            return default[0]
        else:
            return None


def static_maxload_cutoff(df, **kwargs):
    """
    Function to automatically discard data after maximum load has been reached.
    (meant for static tests)
    :param df:          Pandas dataframe containing at least a 'Time' and 'PA0' (= Load) column
    :param kwargs:      'load': set maximum load cutoff point manually.
    :return:            Filtered dataframe containing only entries that
                        happen before 'PA0' reaches its maximum value
    """
    load_data = df[['Time', 'PA0']].dropna()

    max_load = kwargs.get('load')
    if max_load is None:
        max_load = load_data['PA0'].max()
    if len(load_data[load_data['PA0'] >= max_load].index) > 0:
        max_load_index = load_data[load_data['PA0'] >= max_load].index[-1]

        print('pridb_filters: filtered all data after max. load of %s kN is reached.' % str(max_load))
        df = df[df['Time'] < df.loc[max_load_index]['Time']].dropna(subset=['Time'])
    return df


def min_positive_loading_rate(df, **kwargs):
    """
    Function to automatically discard data that happens before a defined loading rate (dLoad).
    Meant for windowing of static tests.
    :param df:          Pandas dataframe containing at least a 'Time' and 'PA0' (= Load) column
    :param kwargs:      'loading_rate': set minimum (default 1 / second)
                        't_lowerbound': cut off data preemptively
    :return:            New dataframe containing only entries that happen
                        after the minimum loading rate has occurred.
    """
    # df = df[df['Time'] > kwargs.get('t_lower', 0)]
    load_data = df[['Time', 'PA0']].dropna()
    loading_rate = kwargs.get('loading_rate')
    if loading_rate is None:
        loading_rate = 0.012

    # Resample the load data
    load_data['Time'] = pd.to_timedelta(load_data['Time']*1E9)
    load_data = load_data.set_index('Time').resample('1S').mean()

    # Simple difference method to compute approximation of derivative
    load_data['dPA0'] = load_data['PA0'].diff()        # column now contains load change/second
    time_lower = load_data.loc[load_data['dPA0'] > loading_rate].index[0].total_seconds()
    load_lower = load_data.loc[load_data['dPA0'] > loading_rate]['PA0'][0]

    df = df[df['Time'] > time_lower].dropna(how='all', axis=1)
    print('pridb_filters: filtered all data before load reached %s kN (rate criterion: %s).'
          % (str(load_lower), str(loading_rate)))

    # import matplotlib.pyplot as plt
    # plt.plot(load_data['dPA0'])
    # plt.show()

    return time_filter(df, reframe_time=True)


def split_min_max_load(df, **kwargs):
    """
    For fatigue tests: split load data into rolling max & rolling min, so they
    can be legibly plotted alongside AE parameters as a function of time
    """
    # df['load_min'] = pd.rolling_min(df['PA0'], 1000, min_periods=200, center=True)
    # df['load_max'] = pd.rolling_max(df['PA0'], 1000, min_periods=200, center=True)
    # pd.rolling_min/_max is deprecated
    df['load_min'] = df['PA0'].rolling(min_periods=200, window=1000, center=True).min()
    df['load_max'] = df['PA0'].rolling(min_periods=200, window=1000, center=True).max()

    df['displ_min'] = df['PA1'].rolling(min_periods=200, window=1000, center=True).min().\
        rolling(min_periods=200, window=3000, center=True).median()
    df['displ_max'] = df['PA1'].rolling(min_periods=200, window=1000, center=True).max().\
        rolling(min_periods=200, window=3000, center=True).median()
    return df


def duration_filter(df, **kwargs):
    """
    :param df:      Pandas dataframe containing 'Dur' column
    :param kwargs:  int lowerbound and/or int upperbound
                    if test_id: lookup filter parameters in pridb_filter_config.ini
    :return:        New dataframe containing only entries with
                    duration between lowerbound & upperbound params
    """
    test_id = kwargs.get('from_config', None)
    d_lower = kwargs.get('lowerbound', None)
    d_upper = kwargs.get('upperbound', None)
    if test_id:
        d_lower = config_getter('General', 'DurationLowerBound')
        d_upper = config_getter('General', 'DurationUpperBound')
    oldlen = len(df[df['TRAI'] > 0].index)
    if d_upper:
        df.drop(df[df['Dur'] > d_upper].index, inplace=True)
    if d_lower:
        df.drop(df[df['Dur'] < d_lower].index, inplace=True)
    newlen = len(df[df['TRAI'] > 0].index)
    if oldlen != newlen:
        percent_deleted = round(100 * (float(oldlen - newlen)) / oldlen, 2)
        print('pridb_filters: filtered %s hits (%s %%) outside duration bounds <%s, %s>.'
              % (str(oldlen-newlen), str(percent_deleted), str(d_lower), str(d_upper)))
    return df


def hit_num_filter(df, **kwargs):
    """
    :param df:      Pandas dataframe containing AE hits
    Filter a defined amount of hits from a dataframe. Note this method does not use
    TRAI or SetID values, as this method is called after filtering.
    h_lower:        lowerbound of hit numbers to include
    h_upper:        upperbound of hit numbers (python bound, so exclusive)
    h_amount:       number of hits to keep.
                    If h_amount < 1: assume it represents a fraction of total hits

    if test_id: lookup filter parameters in pridb_filter_config.ini
    if h_amount: ignore h_upper
    if not h_lower: assume h_lower=0
    """
    test_id = kwargs.get('from_config', None)
    h_lower = kwargs.get('lowerbound', None)
    h_upper = kwargs.get('upperbound', None)
    h_amount = kwargs.get('amount', None)
    if test_id:
        h_lower = config_getter(test_id, 'HitCountLowerBound', 0)
        h_upper = config_getter(test_id, 'HitCountUpperBound')

    if not h_amount and not h_upper:    # Neither upper nor amount: not enough information
        return df

    # Separate data: we will only want to discard hit rows, not load rows
    load_data = df[df['TRAI'].isnull()].dropna()
    df = df[df['TRAI'] > 0].dropna(subset=['Time'])
    print df.head(10)
    if h_amount:
        if h_amount < 1:
            h_amount *= len(df.index)
        df = df.iloc[h_lower:(h_lower+h_amount)]
    elif h_upper:
        df = df.iloc[h_lower:h_upper]

    # Rejoin data
    df = df.append(load_data)
    newlen = len(df[df['TRAI'] > 0].index)
    print('pridb_filters: returned %s hits between hit number bounds <%s, %s>.'
          % (str(newlen), str(h_lower), str(h_upper)))
    return df


def time_filter(df, **kwargs):
    """
    :param df:      Pandas dataframe containing 'Time' column
    :param kwargs:  int lowerbound and/or int upperbound [seconds]
                    boolean reframe_time
                    if test_id: lookup filter parameters in pridb_filter_config.ini
    :return:        New dataframe containing only entries with
                    time between lowerbound & upperbound params.
                    If reframe_time set to True, also shift all
                    timecodes by -min(df['Time']).
    """
    test_id = kwargs.get('from_config', None)
    t_lower = kwargs.get('lowerbound', None)
    t_upper = kwargs.get('upperbound', None)
    t_amount = kwargs.get('max_amount', None)
    reframe_time = kwargs.get('reframe_time', False)
    if test_id:
        reframe_time = config_getter('General', 'ReframeTime')
        t_lower = config_getter(test_id, 'TimeLowerBound')
        t_upper = config_getter(test_id, 'TimeUpperBound')
        t_amount = config_getter(test_id, 'TimeMaxAmount')

    oldlen = len(df[df['TRAI'] > 0].index)
    if t_upper:
        df['Time'] = df[df['Time'] < t_upper]['Time']
        df.dropna(subset=['Time'], inplace=True)
    if t_lower:
        df['Time'] = df[df['Time'] > t_lower]['Time']
        df.dropna(subset=['Time'], inplace=True)
    newlen = len(df[df['TRAI'] > 0].index)
    if t_upper or t_lower:
        print('pridb_filters: filtered %s hits outside time bounds <%s, %s>.'
              % (str(oldlen-newlen), str(t_lower), str(t_upper)))

    if reframe_time:
        offset = df['Time'].min()
        df['Time'] = df['Time'].subtract(offset, axis=0)
        print('pridb_filters: offset time values by %s seconds.' % str(-offset))
    if t_amount and not kwargs.get('ignore_amount', False):
        t_amount_corr = t_amount - df['Time'].min()
        df['Time'] = df[df['Time'] < t_amount_corr]['Time']
        df.dropna(subset=['Time'], inplace=True)
        print('pridb_filters: filtered %s hits outside the first %s minutes' %
              (str(newlen - len(df[df['TRAI'] > 0].index)), str(t_amount/60)))
    return df


def energy_filter(df, **kwargs):
    """
    :param df:      Pandas dataframe containing 'Eny' column
    :param kwargs:  int lowerbound and/or int upperbound
                    if test_id: lookup filter parameters in pridb_filter_config.ini
    :return:        New dataframe containing only entries with
                    energy between lowerbound & upperbound params
    """
    test_id = kwargs.get('from_config', None)
    e_lower = kwargs.get('lowerbound', None)
    e_upper = kwargs.get('upperbound', None)
    if test_id:
        e_lower = config_getter(test_id, 'EnergyLowerBound')
        e_upper = config_getter(test_id, 'EnergyUpperBound')
    oldlen = len(df[df['TRAI'] > 0].index)
    if e_upper:
        df.drop(df[df['Eny'] > e_upper].index, inplace=True)
    if e_lower:
        df.drop(df[df['Eny'] < e_lower].index, inplace=True)
    newlen = len(df[df['TRAI'] > 0].index)
    if oldlen != newlen:
        print('pridb_filters: filtered %s hits outside energy bounds <%s, %s>.'
              % (str(oldlen-newlen), str(e_lower), str(e_upper)))
    return df


def amplitude_filter(df, **kwargs):
    """
    :param df:      Pandas dataframe containing 'Amp' column
    :param kwargs:  int lowerbound and/or int upperbound
                    if test_id: lookup filter parameters in pridb_filter_config.ini
    :return:        New dataframe containing only entries with
                    amplitude between lowerbound & upperbound params
    """
    test_id = kwargs.get('from_config', None)
    a_lower = kwargs.get('lowerbound', None)
    a_upper = kwargs.get('upperbound', None)
    if test_id:
        a_lower = config_getter(test_id, 'AmplitudeLowerBound')
        a_upper = config_getter(test_id, 'AmplitudeUpperBound')
    oldlen = len(df[df['TRAI'] > 0].index)
    if a_upper:
        df.drop(df[df['Amp'] > a_upper].index, inplace=True)
    if a_lower:
        df.drop(df[df['Amp'] < a_upper].index, inplace=True)
    newlen = len(df[df['TRAI'] > 0].index)
    if oldlen != newlen:
        print('pridb_filters: filtered %s hits outside amplitude bounds <%s, %s>.'
              % (str(oldlen-newlen), str(a_lower), str(a_upper)))
    return df


def remove_reflections(df, **kwargs):
    """
    Important note: In the current implementation, remove_reflections needs a sequentially INTACT TRAI index.
    It is therefore best to remove reflections before applying any other filters.

    :param df:      Pandas dataframe containing 'Counts' column
    :param kwargs:  int counts_threshold - number of counts above which to expect reflections.
    :return:        Dataframe that has all the 1-count rows remove that follow a row that has
                    >= counts_threshold counts.

    """
    counts_threshold = kwargs.get('counts_threshold', 3)
    oldlen = len(df[df['TRAI'] > 0].index)
    # df['DeltaTime'] = df['Time'].diff()
    # row i has Time(i) - Time(i-1). Therefore,
    # DeltaTime says for each hit how long it was since the hit before.

    # Find all entries above the counts threshold.
    # Keep (TRAI) indices of entries that follow aforementioned ones:
    df_delete_candidates = df[df['Counts'] >= counts_threshold]['TRAI'].apply(lambda x: x+1)

    # Check for index out of bounds danger
    if df['TRAI'].max() + 1 == max(df_delete_candidates):
        df_delete_candidates = \
            df_delete_candidates[df_delete_candidates != max(df_delete_candidates)]

    # Following statement successfully selects ONLY the to be discarded entries
    # WANT:  not (candidate and C=1). BUT: not(A and B) = not(A) or not(B)
    df = df[~df['TRAI'].isin(df_delete_candidates) | (df['Counts'] != 1)]
    newlen = len(df[df['TRAI'] > 0].index)
    percent_deleted = round(100*(float(oldlen-newlen))/oldlen, 2)
    print('pridb_filters: filtered %s hits (%s %%) as reflections.'
          % (str(int(oldlen-newlen)), str(percent_deleted)))
    return df


def label_load(df):
    """
    Label hits with the 'up'/'top'/'down'/'bottom' fatigue cycle phase labels
    :param df:      Pandas dataframe containing at least a 'PA0' column
    :return:        df with properly filled column 'cycle_label'
    """
    old_index_name = df.index.name
    df[old_index_name] = df.index
    df.set_index('Time', drop=False, inplace=True)
    t = time.time()

    df['PA0'].interpolate(method='index', limit_direction='both', inplace=True)
    df['dPA0'] = np_gradient(df['PA0'])
    if 'PA1' in df.columns:
        df['PA1'].interpolate(method='index', limit_direction='both', inplace=True)
        df['dPA1'] = np_gradient(df['PA1'])

    load_max = df['PA0'].rolling(window=100, center=True).max().median()
    load_min = df['PA0'].rolling(window=100, center=True).min().median()
    load_amplitude = (load_max-load_min)/2.
    load_middle = load_min + load_amplitude
    load_decide = load_amplitude*.5*2**.5       # 1/2 sqrt(2) of the maximum amplitude

    df['cycle_label'] = df.apply(_label_load_helper, args=(load_middle, load_decide), axis=1)
    df.set_index(old_index_name, inplace=True)
    print('pridb_filters: adding cycle labels took %s s' % str(time.time()-t))
    return df


@jit
def _label_load_helper(row, load_middle, load_decide):
    if row['PA0'] > load_middle + load_decide:
        return 'top'
    elif row['PA0'] < load_middle - load_decide:
        return 'bottom'
    elif row['dPA0'] > 0:
        return 'up'
    elif row['dPA0'] < 0:
        return 'down'


