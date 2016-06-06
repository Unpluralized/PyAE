# coding: utf-8
"""
Unify data import/read functions in one function library

read_pridb(str path_to_pridb_file):
-----------
Returns a pandas dataframe with the hits data
Threshold and Amplitude are converted to dB values

array columns in view_ae_data: (miV short for microvolts)
SetID     SetType   Time    Chan    Status  Thr     Amp     RiseT   Dur     Eny     RMS     Counts  TRAI    CHits
(DSET)    [#]       [s]     [#]     ?       [miV]   [miV]   [mi s]  [mi s]  [eu]    [miV]   [#]     [#]     [#]

It seems that the conversion from Amplitude and Threshold [miV] to dB (as exported by the Vallen software) is simply:
Amp [dB] = 20*log_10 (Amp [miV])
"""
import time
from numpy import fft
from progress.bar import Bar
import warnings
import sqlite3                      # reads the Vallen files as sqlite databases
import numpy as np                  # for maths, fast processing &more
import pandas as pd                 # for handling large dataframes
import scipy as sci


def read_pridb(path_to_pridb_file, **kwargs):
    # Import .pridb (sql) database of recorded hits. table view_ae_data contains the recorded AE hits

    # selective imports
    columns = kwargs.get('columns', None)

    # initialize sqlite connection
    conn = sqlite3.connect(path_to_pridb_file)
    c = conn.cursor()

    # Get number of hits from table ae_globalinfo
    try:
        c.execute("SELECT Value FROM ae_globalinfo where Key = 'TRAI'")
        number_of_hits = int(c.fetchall()[0][0])# + 1
    except Exception as e:
        warnings.warn('Error while trying to fetch number of hits from .pridb file.')
        raise e
    else:
        print('\nread_pridb: ' + str(number_of_hits) + ' hits found in ' +
              path_to_pridb_file.split('/')[-1] + '\n')

    # Load actual hits data from the file into a pandas dataframe
    if columns:
        columns = list(columns)
        if 'SetID' not in columns:
            columns.append('SetID')
        if 'TRAI' not in columns:
            columns.append('TRAI')
        columns_text = ', '.join(columns)
        ae_hits = pd.read_sql_query("SELECT " + columns_text + " FROM view_ae_data", conn)

    else:
        ae_hits = pd.read_sql_query(
            "SELECT TRAI, SetID, Time, Thr, Amp, RiseT, Dur, Eny, RMS, PA0, PA1, Counts "
            "FROM view_ae_data WHERE TRAI >= 0",
            conn)

    # Convert Amplitude and Threshold values from microvolts to decibels
    if 'Amp' in ae_hits.columns:
        # ae_hits['Amp'] = np.around(20*np.log10(ae_hits['Amp']), decimals=1)
        ae_hits['Amp'] = 20 * np.log10(ae_hits['Amp'])
        ae_hits['Amp'] = ae_hits['Amp'].round(decimals=1)
    if 'Thr' in ae_hits.columns:
        ae_hits['Thr'] = 20 * np.log10(ae_hits['Thr'])
        ae_hits['Thr'] = ae_hits['Thr'].round(decimals=1)
        # ae_hits['Thr'] = np.around(20*np.log10(ae_hits['Thr']), decimals=1)

    # Set the SetID columns (also called DSET the csv exports) as the index column.
    ae_hits = ae_hits.set_index('SetID')

    c.close()
    conn.close()

    return ae_hits


def read_tradb(path_to_tradb_file, **kwargs):
    """
    Read tradb file to memory
    :param path_to_tradb_file:          str path to .tradb file
    :return:                            numpy array with time intervals in row 0, trai i on row i
    """
    # initialize sqlite connection
    conn = sqlite3.connect(path_to_tradb_file)
    c = conn.cursor()

    # Read some data characteristics:
    c.execute("SELECT Value FROM tr_globalinfo where Key = 'TRAI'")
    number_of_hits = int(c.fetchall()[0][0])# + 1
    print('\nread_tradb: ' + str(number_of_hits) + ' hits found in ' +
              path_to_tradb_file.split('/')[-1])
    info = pd.read_sql_query("SELECT Pretrigger, Thr, SampleRate, Samples, TR_mV "
                             "FROM view_tr_data WHERE TRAI = 1", conn)
    # Harvest data characteristics to variables:
    threshold = np.around(20*np.log10(info['Thr'][0]), decimals=1)        # in decibels
    samples_per_waveform = info['Samples'][0]
    pretrigger_samples = info['Pretrigger'][0]
    sample_rate = info['SampleRate'][0]             # in Hz
    tr_mV = info['TR_mV'][0]    # Conversion factor for blob data to mV
    print('Waveforms info' + '\n--------------'
          '\nThreshold:\t\t\t' + str(threshold) + ' dB' +
          '\nSamples per waveform:\t\t' + str(samples_per_waveform) +
          '\nPretrigger samples:\t\t' + str(pretrigger_samples) +
          '\nSample rate:\t\t\t' + str(sample_rate/10.0**6) + ' MHz\n')

    trai = sorted(kwargs.get('trai', None))
    if trai:
        if not isinstance(trai, tuple) and not isinstance(trai, list):
            trai = [trai]
        if 0 in trai:
            trai = [i for i in trai if i is not 0]
    if not trai:
        trai = range(1, number_of_hits+1)

    print('read_tradb: loading ' + str(len(trai)) + ' waveform(s) to memory.')
    t_start = time.time()

    """ Retrieval method: single query to pandas, np.frombuffer conversion """      # Promising (1M waveforms in ~10s)
    ae_waveforms = pd.read_sql("SELECT Data FROM view_tr_data WHERE TRAI > 0 AND TRAI <=" + str(max(trai)),
                               conn, coerce_float=False)

    # print np.frombuffer(data['Data'][0], dtype=np.short)
    ae_waveforms = ae_waveforms['Data'].apply(lambda x: np.frombuffer(x, dtype=np.short))
    print('operation took ' + str(round(time.time()-t_start, 3)) + ' seconds\n')

    # Make room for time data in the first row (add one row of padding)
    ae_waveforms = sci.pad(ae_waveforms, (1, 0), mode='constant')
    # Make time array and save to first row
    t = sci.array(range(0, samples_per_waveform))/float(sample_rate)*10.0**6
    t = t - sci.array(info['Pretrigger'])/float(info['SampleRate'])*10.0**6
    ae_waveforms[0] = t

    # Convert all values to mV, except the time row:
    ae_waveforms[1:] = ae_waveforms[1:]*tr_mV

    c.close()
    conn.close()
    return ae_waveforms


def append_spectral_features(df, path_to_tradb_file, **kwargs):
    """
    :param df:                      AE hits&features dataframe, arbitrarily filtered.
    :param path_to_tradb_file:      str path to waveforms data (.tradb file)
    :param kwargs:                  None
    :return:                        AE hits dataframe with appended columns
                                    ['Ef_95_150', 'Ef_150_250', 'Ef_250_350', 'Ef_350_500', 'Ef_500_850']
    """
    df['TRAI'] = df['TRAI'].astype(np.object)
    trai = sorted(df.loc[df['TRAI'] > 0, 'TRAI'].dropna().astype(np.int).tolist())

    ae_waveforms = read_tradb(path_to_tradb_file, trai=trai)
    f_khz = np.linspace(0, 1000, 1025)
    locs = [np.argmax(f_khz >= 95), np.argmax(f_khz >= 150), np.argmax(f_khz >= 250),
            np.argmax(f_khz >= 350), np.argmax(f_khz >= 500), np.argmin(f_khz < 850)]

    progress_bar = Bar('Processing', max=len(trai))

    """
    4 threads:                          1000/455677 in 1:25 min
    3 threads:                          1000/455677 in 1:24 min
    2 threads:                          1000/455677 in 1:22 min
    linear process; one row a time:     1000/455677 in 1:14 min
    linear process; python lists only:  1000/455677 in 0:00.155 min
    2 threads; python lists only:       1000/455677 in 0:00.23 min
    """

    # Compute fft
    spectra_list = []
    for tr in trai:
        fft_result = fft.rfft(ae_waveforms[tr])
        fft_result = sci.apply_along_axis(lambda x: sci.absolute(x), 0, fft_result)
        spectra_list.append([sum(fft_result[locs[0]:locs[1] + 1]),
                            sum(fft_result[locs[1]:locs[2] + 1]),
                            sum(fft_result[locs[2]:locs[3] + 1]),
                            sum(fft_result[locs[3]:locs[4] + 1]),
                            sum(fft_result[locs[4]:locs[5] + 1])])
        progress_bar.next()
    progress_bar.finish()
    sf = pd.DataFrame(spectra_list,
                      index=trai, columns=['Ef_95_150', 'Ef_150_250', 'Ef_250_350', 'Ef_350_500', 'Ef_500_850'])

    # print sf.head(10)

    oldindex = df.index.name
    df[oldindex] = df.index
    df.set_index('TRAI', drop=False, inplace=True)
    # print len(df.index)
    df = pd.concat([df, sf], join_axes=[df.index], axis=1)
    # print len(df.index)
    df.set_index(oldindex, inplace=True)
    # print df.dropna(subset=['TRAI']).head(10)

    return df
