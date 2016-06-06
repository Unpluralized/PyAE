# PyAE
Library of functions to retrieve data exported by Vallen Acoustic Emission system software: .pridb (waveform features) and .tradb (full transients). Some processing- and filtering functions are also included. This function library is developed for myself and is shared as-is. Though all functions are in working order and are documented to a basic level, they are not necessarily elegant or robust.

### vallendata.py
Contains functions to retrieve waveform feature data from .pridb files and complete transient waveforms from .tradb files.

##### read_pridb(str path_to_pridb_file)
returns Pandas dataframe with AE hits and their waveform features. Amplitudes are converted to [dB], consistent with their representation in the VallenAE system software.

##### read_tradb(str path_to_tradb_file)
returns numpy array with a complete transient waveform on each row; row 0 is the time axis

##### append_spectral_features(DataFrame df, str path_to_tradb_file)
returns DataFrame df containing the AE hits and their waveform features, with appended spectral partial power feature columns ['Ef_95_150', 'Ef_150_250', 'Ef_250_350', 'Ef_350_500', 'Ef_500_850'], calculated from the full transient recordings.
An assumption of 2048 samples at 2.0 MHz is hard-coded in this function.

### pridb_filters.py
Set of filtering functions. pridb_filters.apply_filters(DataFrame df, str test_id) can be called to apply a predetermined set of filters, retrieving filter parameters from a config file: pridb_filter_config.ini, where each section must represent a test_id.

Individual filters can also be called, passing filter parameters as arguments.


### Dependencies
Numpy, scipy, pandas, sqlite3, progress, matplotlib, numba
