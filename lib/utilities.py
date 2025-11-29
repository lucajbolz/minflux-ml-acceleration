"""
Module with base-classes that provide some useful methods that are not associated with a specific module.
"""
from json import JSONEncoder
import json as json
import yaml as yaml
import numpy as np
import jax.numpy as jnp
import os
import re
from datetime import datetime
import git # package to obtain the current githash to reference it in artifacts.
import string
import random
import sqlite3
import subprocess
import inspect
import shutil
import importlib
import os
from multiprocessing.pool import ThreadPool

from lib.config import ROOT_DIR, OUTPUT_DIR

class Configuration:
    """
    Configuration class, used by many objects to initialize an internal (read only) configuration, which cannot
    be changed after initialization.

    :param default_params: Dictionary of default parameters.
    :type default_params: dict
    :param param_dict: Dictionary of parameters to be set.
    :type param_dict: dict

    :returns: None
    :rtype: None
    """

    def __init__(self, default_params, param_dict):
        """
        Initializes an instance of the Configuration class.

        :param default_params: Dictionary of default parameters.
        :type default_params: dict
        :param param_dict: Dictionary of parameters to be set.
        :type param_dict: dict

        :returns: None
        :rtype: None
        """
        self.set_params(default_params)
        self.set_params(param_dict)
        pass

    def set_params(self, param_dict):
        """
        Update an instance of Configuration with a dictionary,
        e.g. to modify values after a fit.
        Dependent quantities are recalculated.

        :param param_dict: Dictionary of parameters to be set.
        :type param_dict: dict

        :returns: None
        :rtype: None
        """
        for key in param_dict:
            setattr(self, key, param_dict[key])
        if 'mol_pos' in param_dict:
            setattr(self,'mol_pos',np.asarray(param_dict['mol_pos']))
        pass

    def do_sanity_check(self):
        """
        Perform a sanity check on the Configuration object.

        :returns: None
        :rtype: None
        """
        pass

class Status:
    """
    Class to document the status of a component. Sanity checks can change the
    status of an object and trigger an action, e.g. abortion of a measurement.

    :ivar ok: The status of the component. True if the sanity check was successful, False otherwise.
    :vartype ok: bool
    """
    def __init__(self):
        """
        Initializes an instance of the Status class.

        :returns: None
        :rtype: None
        """
        self.ok = False # default status is False, successful sanity check changes this to True


class FunctionLogger:
    """
    Class to log the activity of functions with an observer pattern.

    :ivar table_name: The name of the table in the database to store the log records.
    :vartype table_name: str
    :ivar data_base: The name of the SQLite database file.
    :vartype data_base: str
    :ivar comment: Additional comment for the log records.
    :vartype comment: str
    :ivar git_hash: The current Git hash.
    :vartype git_hash: str
    :ivar conn: The connection to the SQLite database.
    :vartype conn: sqlite3.Connection
    :ivar registered_functions: List of registered functions.
    :vartype registered_functions: List
    :ivar success: The success status of the logged function.
    :vartype success: bool
    """

    def __init__(self, table_name='test', data_base='log.sqlite', comment='test'):
        """
        Initializes an instance of the FunctionLogger class.

        :param table_name: The name of the table in the database to store the log records.
        :type table_name: str
        :param data_base: The name of the SQLite database file.
        :type data_base: str
        :param comment: Additional comment for the log records.
        :type comment: str

        :returns: None
        :rtype: None
        """
        self.git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        self.conn = sqlite3.connect(data_base)
        self.registered_functions = []
        self.comment = comment
        self.table_name = table_name
        self.data_base = data_base
        self.success = True

    def __call__(self, function):
        """
        Decorator to wrap a function and log its activity.

        :param function: The function to be wrapped and logged.
        :type function: function

        :returns: The wrapped function.
        :rtype: function
        """
        def wrapped_function(*args, **kwargs):
            self.register_function(function)
            result = function(*args, **kwargs)
            self.log_function_call(function.__name__, args, kwargs)
            return result
        return wrapped_function

    def register_function(self, function):
            """
            Registers a function to be logged.

            :param function: The function to be registered.
            :type function: function

            :returns: None
            :rtype: None
            """
            self.registered_functions.append(function)

    def log_function_call(self, function_name, args, kwargs):
        """
        Logs a function call.

        :param function_name: The name of the function.
        :type function_name: str
        :param args: The arguments passed to the function.
        :type args: tuple
        :param kwargs: The keyword arguments passed to the function.
        :type kwargs: dict

        :returns: None
        :rtype: None
        """
        now = datetime.now()
        args_str = str(args)
        kwargs_str = str(kwargs)
        self.conn.execute(f"CREATE TABLE IF NOT EXISTS {self.table_name} (date TEXT, function_name TEXT, success TEXT, comment TEXT, git_hash TEXT, args TEXT, kwargs TEXT)")
        self.conn.execute(f"INSERT INTO {self.table_name} (date, function_name, success, comment, git_hash, args, kwargs) VALUES (?, ?, ?, ?, ?, ?, ?)", (now, function_name, self.success, self.comment, self.git_hash, str(args), str(kwargs)))
        self.conn.commit()

    def rerun_f(self, date):
        """
        Reruns a function call based on the specified date.

        :param date: The date of the function call to rerun.
        :type date: str

        :returns: The result of the rerun function call.
        :rtype: Any

        :raises ValueError: If no function call is found with the specified date.
        """
        c = self.conn.cursor()
        c.execute(f"SELECT function_name, git_hash, args, kwargs FROM {self.table_name} WHERE date = ?", (date,))
        row = c.fetchone()
        self.conn.close()
        if row:
            f_name, git_hash, args, kwargs = row

            # Checkout the code from the specified Git hash
            subprocess.check_output(['git', 'checkout', git_hash])

            # Get the module containing the function
            module_name = inspect.getmodulename(inspect.getfile(inspect.currentframe().f_back))
            module = importlib.import_module(module_name)

            # Get the function from the module
            func = getattr(module, f_name)

            # Call the function with its arguments
            result = func(*eval(args), **eval(kwargs))

            # Restore the original state of the code by checking out the current Git hash
            subprocess.check_output(['git', 'checkout', 'HEAD'])

            return result
        else:
            raise ValueError(f"No function call found with id {date}")


    def terminate(self):
        """
        Terminates the connection to the SQLite database.

        :returns: None
        :rtype: None
        """
        self.conn.close()



class Log:
    def __init__(self, comment):
        """
        Initializes a Log object.

        :param comment: The comment to be logged.
        :type comment: str

        :returns: None
        :rtype: None
        """
        pass
    
    def log_to_database(self, database):
        """
        Logs a function call to the database.

        :param database: The name of the SQLite database file.
        :type database: str

        :returns: None
        :rtype: None
        """
        # Get the current Git hash
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        
        # Connect to the database or create it if it doesn't exist
        conn = sqlite3.connect(database)
        
        # Get the calling function's name and arguments
        frame = inspect.currentframe()
        function_name = inspect.getframeinfo(frame.f_back).function
        arg_values = inspect.getargvalues(frame.f_back).locals
        arg_dict = {key: arg_values[key] for key in arg_values.keys() if key in inspect.getargvalues(frame.f_back).args}
        
        # Get the current date and time
        now = datetime.datetime.now()
        
        conn.execute(f"CREATE TABLE IF NOT EXISTS {function_name} (date TEXT, success TEXT, comment TEXT, git_hash TEXT, args TEXT)")
        conn.execute(f"INSERT INTO {function_name} (date, flag, comment, git_hash, args) VALUES (?, ?, ?, ?, ?)", (now, self.success, self.comment, git_hash, str(arg_dict)))
        conn.commit()
        
        # Close the connection to the database
        conn.close()
        pass

class NumpyArrayEncoder(JSONEncoder):
    """
    Helper class for JSON export of NumPy arrays.

    This class is used as a custom encoder for JSON serialization of NumPy arrays. It handles the conversion of NumPy data types to their Python counterparts, as well as converting NumPy arrays to lists.

    Usage:
        json.dumps(data, cls=NumpyArrayEncoder)

    Note:
        To use this encoder, you need to import NumPy and JSONEncoder from the json module.

    Example:
        import json
        import numpy as np

        data = {'array': np.array([1, 2, 3])}
        json_str = json.dumps(data, cls=NumpyArrayEncoder)
        print(json_str)

    Output:
        {"array": [1, 2, 3]}
    """

    def default(self, obj):
        """
        Converts the specified object to a JSON serializable form.

        :param obj: The object to be converted.
        :type obj: Any

        :returns: The JSON serializable form of the object.
        :rtype: Any
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray) or isinstance(obj, jnp.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super(NumpyArrayEncoder, self).default(obj)
class Labeler:
    def __init__(self):
        """
        Initializes a Labeler object.

        :returns: None
        :rtype: None
        """
        self.HM_string = datetime.now().strftime('%H%M') # create a time stamp with hhmmss format for the current run
        pass

    def stamp(self, filename):
        """
        Creates a directory for the present date and a file string to save the artifacts with time and git-hash.

        :param filename: The name of the file to be saved.
        :type filename: str

        :returns: The file string and the directory path.
        :rtype: tuple(str, str)
        """
        year = datetime.now().strftime('%Y') # get the current date and time
        month = datetime.now().strftime('%m')
        day = datetime.now().strftime('%d')
        
        todays_dir = os.path.join(OUTPUT_DIR, f'{year}/{month}/{day}/')

        repo = git.Repo(search_parent_directories=True) # extract the git hash of the current build, to put it into the plots
        sha = repo.head.object.hexsha
        short_sha = repo.git.rev_parse(sha, short=8)
        file_str = self.id_generator() + '-' + filename + f'-{short_sha}'

        os.makedirs(todays_dir,exist_ok=True)
        return file_str, todays_dir

    def id_generator(self, size=6, chars=string.ascii_uppercase + string.digits):
        """
        Generates a random ID string.

        :param size: The length of the generated string.
        :type size: int

        :param chars: The characters to choose from.
        :type chars: str

        :returns: The generated ID string.
        :rtype: str
        """
        return ''.join(random.choice(chars) for _ in range(size))
class Converter:
    def __init__(self):
        """
        Initializes a Converter object.

        :returns: None
        :rtype: None
        """
        pass

    def to_yaml(self, obj):
        """
        Converts an object to YAML.

        :param obj: The object to be converted.
        :type obj: Any

        :returns: The serialized YAML output for the object, or None when saving to a file.
        :rtype: str or None
        """
        serialized = yaml.dump(obj, Dumper=yaml.CDumper)
        return serialized

class Importer:
    def __init__(self):
        """
        Initializes an Importer object.

        :returns: None
        :rtype: None
        """
        pass

    def load_yaml(self, obj):
        """
        Converts from YAML back to an object.

        :param obj: The path to the YAML file or the YAML object.
        :type obj: str or object

        :returns: The deserialized object.
        :rtype: object
        """
        if isinstance(obj, str) and obj.endswith(".yaml"):
            file = open(obj)
            deserialized = yaml.load(file, Loader=yaml.CLoader)
        else:
            try:
                deserialized = yaml.load(obj, Loader=yaml.CLoader)
            except:
                raise Exception('Could not import obj as YAML.')
        return deserialized

    def load_json(self, filename):
        """
        Loads a dictionary from a JSON file, such as an experimental configuration.

        :param filename: The path to the JSON file.
        :type filename: str

        :returns: The loaded dictionary.
        :rtype: dict
        """
        file = open(filename)
        dicts = json.load(file)
        return dicts
class Exporter:
    def __Init__(self):
        """
        Initializes an Exporter object.

        :returns: None
        :rtype: None
        """
        pass

    def write_yaml(self, obj, filename=None, out_dir=None):
        """
        Export obj to a YAML file.

        :param obj: The object to be exported.
        :type obj: Any

        :param filename: The name of the YAML file to be created. If None, a timestamp will be used.
        :type filename: str or None

        :param out_dir: The directory to save the YAML file. If None, the current working directory will be used.
        :type out_dir: str or None

        :returns: The full path to the created YAML file.
        :rtype: str
        """
        file_stamp, todays_dir = Labeler().stamp(filename)
        if out_dir is None:
            out_dir = todays_dir
        if filename is None:
            filename = file_stamp + '.yaml'
        os.makedirs(out_dir, exist_ok=True)
        with open(out_dir + filename, "w") as f:
            yaml.dump(obj, f, Dumper=yaml.CDumper)
        return out_dir + filename

    def write_json(self, new_data, filestr=None, filename=''):
        """
        Writes data to a JSON file.

        :param new_data: The data to be written to the JSON file.
        :type new_data: dict

        :param filestr: The path to the JSON file to be opened and data added. If None, a new file will be created.
        :type filestr: str or None

        :param filename: The filename of the JSON file to be created. If no path is provided, a timestamp will be used.
        :type filename: str

        :returns: The full path to the created JSON file.
        :rtype: str
        """
        if filestr is None:
            filestr, todays_dir = Labeler().stamp(filename)
            filestr += '.json'
            filestr = todays_dir + filestr
        try:
            with open(filestr, "r") as f:
                loaded = json.load(f)
        except IOError:
            loaded = {}
        full_data = loaded | new_data
        with open(filestr, "w") as f:
            json.dump(full_data, f, indent=4, cls=NumpyArrayEncoder)
        return filestr

class MultithreadedCopier:
    def __init__(self, max_threads):
        """
        Initializes a MultithreadedCopier object.

        :param max_threads: The maximum number of threads to use for copying.
        :type max_threads: int

        :returns: None
        :rtype: None
        """
        self.pool = ThreadPool(max_threads)

    def copy(self, source, dest):
        """
        Copy a file from source to destination using multiple threads.

        :param source: The path to the source file.
        :type source: str

        :param dest: The destination directory.
        :type dest: str

        :returns: None
        :rtype: None
        """
        self.pool.apply_async(shutil.copy2, args=(source, dest))

    def __enter__(self):
        """
        Enter the context manager.

        :returns: Self
        :rtype: MultithreadedCopier
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager.

        :param exc_type: The exception type, if an exception occurred.
        :type exc_type: type or None

        :param exc_val: The exception value, if an exception occurred.
        :type exc_val: Exception or None

        :param exc_tb: The exception traceback, if an exception occurred.
        :type exc_tb: traceback or None

        :returns: None
        :rtype: None
        """
        self.pool.close()
        self.pool.join()
class BaseFunc:
    def __init__(self):
        """
        Initialize the BaseFunc class.

        :return: None
        """
        pass

    def join_dict_values(self, dict1, dict2):
        """
        Join the values of dict1 with dict2.

        :param dict1: The first dictionary.
        :type dict1: dict

        :param dict2: The second dictionary.
        :type dict2: dict

        :returns: The joined dictionary.
        :rtype: dict
        """
        result = {}
        for key in dict2:
            value2 = dict2[key]
            if not key in dict1.keys():
                result[key] = value2
            else:
                value1 = dict1[key]
                # Combine values based on their types
                if isinstance(value2, (list, tuple, np.ndarray)):
                    #assert that value1 is the same type
                    result[key] = np.concatenate((value1, value2))
                elif isinstance(value2, dict):
                    #assert that value1 is the same type
                    result[key] = {**value1, **value2}
                else:
                    result[key] = (value1, value2)
        return result

    def remove_trailing_zeros(self, array):
        """
        Remove trailing zeros from an array.

        :param array: The input array.
        :type array: list or numpy.ndarray

        :returns: The array with trailing zeros removed.
        :rtype: list or numpy.ndarray
        """
        new_array = array
        for i in range(len(array) - 1, -1, -1):
            if array[i] != 0:
                break  # Stop iterating if a non-zero value is found
            else:
                new_array = new_array[:-1]  # Remove the trailing zero
        return new_array

    def calculate_norms(self, list_with_nans):
        """
        Calculate the L2 norm for each non-NaN element in a given list.

        :param list_with_nans: A list of floats or NaNs.
        :type list_with_nans: list

        :returns: A list containing the L2 norm for each non-NaN element.
        :rtype: list

        Example:

        >>> obj = BaseFunc()
        >>> obj.calculate_norms([np.array([1,2,3]), np.nan])
        [3.7416563645999, nan]
        """
        norms_list = [np.linalg.norm(elem) if not np.isnan(elem).any() else elem for elem in list_with_nans]
        return norms_list

    def flatten_list(self, lst):
        """
        Flatten a nested list.

        :param lst: The input list.
        :type lst: list

        :returns: The flattened list.
        :rtype: list

        Example:

        >>> obj = BaseFunc()
        >>> obj.flatten_list([1, [2, [3, 4]], 5])
        [1, 2, 3, 4, 5]
        """
        result = []
        for item in lst:
            if isinstance(item, list):
                result.extend(self.flatten_list(item))
            else:
                result.append(item)
        return result
    
    def nan_mask_list(self, list1, list2, list3=None):
        """
        Return a copy of list1 with values set to NaN where they differ from list2 or list3 if provided.

        :param list1: The list to be masked.
        :type list1: list
        :param list2: The subset of list1 that will be used to mask it.
        :type list2: list
        :param list3: The subset that will be used to mask list1 in place of list2 if not None and equal in length to list2.
        :type list3: list or None

        :return: A new list with values set to NaN where they differ from list2 or list3.
        :rtype: list

        :raises ValueError: If list2 is not a subset of list1 or list2 and list3 are not equal in length.

        :Example:

        >>> obj = BaseFunc()
        >>> list1 = [1, 2, 3, 4, 5, 6]
        >>> list2 = [3, 4, 5]
        >>> obj.nan_mask_list(list1, list2, list3=[0, 1, 2])
        [nan, nan, 0, 1, 2, nan]
        """
        if list3 is None:
            list3 = list2
        elif len(list2) != len(list3):
            raise ValueError("list2 and list3 must be equal in length.")

        arr1 = np.array(list1)
        arr2 = np.array(list2)
        arr3 = np.array(list3)
        
        mask = np.zeros_like(arr1, dtype=bool)
        for i in range(arr1.size - arr2.size + 1):
            if np.all(arr1[i:i+arr2.size] == arr2):
                mask[i:i+arr2.size] = True
            count = sum(mask==True)
            if count==arr2.size:
                break

        new_arr = np.empty(arr1.shape)
        new_arr[:] = np.nan
        new_arr[mask] = arr1[mask]
        
        new_arr[mask] = arr3

        return new_arr.tolist()

    def to_tuple(self, lst):
        """
        Convert a nested list to a tuple.

        :param lst: The input list.
        :type lst: list

        :returns: The converted tuple.
        :rtype: tuple

        :Example:

        >>> obj = BaseFunc()
        >>> obj.to_tuple([1, [2, [3, 4]], 5])
        (1, (2, (3, 4)), 5)
        """
        return tuple(self.to_tuple(i) if isinstance(i, list) else i for i in lst)

    
    def _moving_average(self, a, n=10):
        """
        Calculate the moving average of an array.

        :param a: The input array.
        :type a: list or numpy.ndarray
        :param n: The window size for the moving average. Default is 10.
        :type n: int

        :return: The moving average of the input array.
        :rtype: numpy.ndarray

        :Example:

        >>> obj = BaseFunc()
        >>> arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> obj._moving_average(arr, n=3)
        array([2., 3., 4., 5., 6., 7., 8.])
        """
        # Convert the input array to JAX array
        a = jnp.array(a)
        # Compute the cumulative sum using JAX
        ret = jnp.cumsum(a)
        # Compute the moving sum by subtracting shifted cumulative sums
        ret = ret.at[n:].set(ret[n:] - ret[:-n])
        # Compute the moving average and return the result
        return ret[n - 1:] / n
    
    def weighted_median(self, values, weights):
        """
        Calculate the weighted median of a list of values.

        :param values: The input values.
        :type values: list
        :param weights: The weights corresponding to the input values.
        :type weights: list

        :return: The weighted median.
        :rtype: float

        :Example:

        >>> obj = BaseFunc()
        >>> values = [1, 2, 3, 4, 5]
        >>> weights = [0.1, 0.2, 0.3, 0.2, 0.2]
        >>> obj.weighted_median(values, weights)
        3.0
        """
        # Sort the values and weights in ascending order of values
        sorted_values, sorted_weights = zip(*sorted(zip(values, weights), key=lambda x: x[0]))
        # Calculate the cumulative sum of weights
        cum_weights = [sum(sorted_weights[:i+1]) for i in range(len(sorted_weights))]
        # Find the index where the cumulative sum of weights is greater than or equal to 0.5
        index = next(i for i, cum_weight in enumerate(cum_weights) if cum_weight >= 0.5 * cum_weights[-1])
        # If the cumulative sum of weights at the found index is greater than 0.5, return the corresponding value
        if cum_weights[index] > 0.5 * cum_weights[-1]:
            weighted_median = sorted_values[index]
        else:
            # Otherwise, interpolate between the values at the found index and the next index
            weight1 = cum_weights[index] - 0.5 * cum_weights[-1]
            weight2 = 0.5 * cum_weights[-1] - cum_weights[index - 1]
            weighted_median = (sorted_values[index - 1] * weight1 + sorted_values[index] * weight2) / (weight1 + weight2)
        return weighted_median


    def match_pattern(self, string, pattern, match='partial'):
        """
        Check if a string matches a pattern.

        :param string: The input string.
        :type string: str
        :param pattern: The pattern to match against.
        :type pattern: str
        :param match: The type of match to perform. Default is 'partial'.
                    - 'full': Check for a full match.
                    - 'partial': Check for a partial match.
        :type match: str

        :return: True if the string matches the pattern, False otherwise.
        :rtype: bool

        :Example:

        >>> obj = BaseFunc()
        >>> obj.match_pattern('hello', 'hello', match='full')
        True
        >>> obj.match_pattern('hello world', 'hello', match='partial')
        True
        """
        if match == 'full':
            return bool(pattern == string)
        else:
            return bool(re.search(pattern, string))

    def find_files(self, in_folder, condition, max_files=100, max_depth=10):
        """
        Iterate through a nested directory and find files that meet a given condition.

        :param in_folder: The root directory to start the iteration from.
        :type in_folder: str
        :param condition: The condition that files must meet to be included.
        :type condition: callable
        :param max_files: The maximum number of files to be analyzed in one directory.
                        Defaults to 100.
        :type max_files: int
        :param max_depth: The maximum depth of subdirectories to be iterated through.
                        Defaults to 10.
        :type max_depth: int

        :return: A list of paths to the found files.
        :rtype: list

        :Example:

        >>> obj = BaseFunc()
        >>> def file_condition(filename):
        ...     return filename.endswith('.txt')
        >>> obj.find_files('/path/to/directory', file_condition, max_files=50, max_depth=5)
        ['/path/to/directory/file1.txt', '/path/to/directory/subdir/file2.txt', ...]
        """
        int_count = 0
        processable_files = []
        depth = 0
        for root, dirs, files in os.walk(in_folder, topdown=True):
            if max_depth is not None and root.count(os.sep) - in_folder.count(os.sep) > max_depth:
                break
                del dirs[:]  # Don't process subdirectories beyond max_depth
            if len(processable_files) < max_files and int_count < 10**6:
                for file in files:
                    path = os.path.join(root, file)
                    if condition(path) and len(processable_files) < max_files and int_count < 10**6:
                        processable_files.append(path)
                    int_count += 1
            else:
                break
        return processable_files
    
    def get_git_root(self, path):
        """
        Get the root directory of a Git repository.

        Given a file or directory path within a Git repository, this method returns the root directory of that repository.

        :param path: File or directory path within the Git repository.
        :type path: str
        :return: Root directory of the Git repository.
        :rtype: str
        """
        git_repo = git.Repo(path, search_parent_directories=True)
        git_root = git_repo.git.rev_parse("--show-toplevel")
        return git_root