import copy
import inspect
import os
import shlex
import sys

import numpy as np
import pandas

from ...datbase import DataListInterface, DataType
from ...discretization.structuredgrid import StructuredGrid
from ...discretization.unstructuredgrid import UnstructuredGrid
from ...discretization.vertexgrid import VertexGrid
from ...utils import datautil
from ..data import mfdata
from ..mfbase import ExtFileAction, FlopyException, MFDataException, \
    VerbosityLevel
from ...utils.datautil import clean_filename, DatumUtil
from .mfdatastorage import DataStorageType, DataStructureType
from .mfdatautil import convert_data
from .mffileaccess import MFFileAccessList
from .mfdatalist import MFList
from .mfstructure import DatumType, MFDataStructure


class ListHeader:
    def __init__(self, names=None, types=None, num_rows=None):
        self.names = names
        self.types = types
        self.num_rows = num_rows


class BasicListStorage:
    def __init__(self):
        self.internal_data = None
        self.fname = None
        self.iprn = None
        self.binary = False
        self.data_storage_type = None
        self.modified = False
        self.loaded = False

    def get_record(self):
        rec = {}
        if self.internal_data is not None:
            rec["data"] = copy.deepcopy(self.internal_data)
        if self.fname is not None:
            rec["filename"] = self.fname
        if self.iprn is not None:
            rec["iprn"] = self.iprn
        if self.binary:
            rec["binary"] = True
        return rec

    def set_record(self, rec):
        if "data" in rec:
            self.internal_data = rec["data"]
        if "filename" in rec:
            self.fname = rec["filename"]
        if "iprn" in rec:
            self.iprn = rec["iprn"]
        if "binary" in rec:
            self.binary = rec["binary"]

    def merge_data(self, internal_data):
        if internal_data is None:
            return
        elif self.internal_data is None:
            self.internal_data = internal_data
        elif isinstance(self.internal_data, dict) \
                and isinstance(internal_data, dict):
            for key, data in internal_data:
                if key in self.internal_data:
                    self.internal_data[key].append(data)
                else:
                    self.internal_data[key] = data
        else:
            raise Exception("Data types not supported.")

    def set_internal(self, internal_data, append=False):
        self.data_storage_type = DataStorageType.internal_array
        if append:
            self.merge_data(internal_data)
        else:
            self.internal_data = internal_data
        self.fname = None
        self.binary = False

    @property
    def internal_size(self):
        if not isinstance(self.internal_data, dict):
            return 0
        else:
            data_size = 0
            for data in self.internal_data.values():
                data_size += len(data)
            return data_size

    def set_external(self, fname):
        self.data_storage_type = DataStorageType.external_file
        self.internal_data = None
        self.fname = fname

    def has_data(self):
        if self.data_storage_type == DataStorageType.internal_array:
            return self.internal_data is not None
        else:
            return self.fname is not None


class MFBasicList(mfdata.MFMultiDimVar, DataListInterface):
    def __init__(
        self,
        sim_data,
        model_or_sim,
        structure,
        data=None,
        enable=None,
        path=None,
        dimensions=None,
        package=None,
        block=None,
    ):
        super().__init__(
            sim_data, model_or_sim, structure, enable, path, dimensions
        )
        self._data_storage = self._new_storage()
        self._package = package
        self._block = block
        self._last_line_info = []
        self._data_line = None
        self._temp_dict = {}
        self._crnt_line_num = 1
        self._data_header = None
        self._header_names = None
        self._mg = None
        self._current_key = 0
        if self._model_or_sim.type == "Model":
            self._mg = self._model_or_sim.modelgrid

        if data is not None:
            try:
                self.set_data(data, True)
            except Exception as ex:
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    structure.get_model(),
                    structure.get_package(),
                    path,
                    "setting data",
                    structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    None,
                    sim_data.debug,
                    ex,
                )

    @property
    def data_type(self):
        """Type of data (DataType) stored in the list"""
        return DataType.list

    @property
    def package(self):
        """Package object that this data belongs to."""
        return self._package

    @property
    def dtype(self):
        """Type of data (numpy.dtype) stored in the list"""
        return self.get_data().dtype

    def _append_type_list(self, data_name, data_type):
        self._data_header.append((data_name, data_type))
        self._header_names.append(data_name)

    def _get_default_mult(self):
        if self._data_type == DatumType.integer:
            return 1
        else:
            return 1.0

    def _process_internal_line(self, arr_line):
        multiplier = self._get_default_mult()
        print_format = None
        if isinstance(arr_line, list):
            index = 1
            while index < len(arr_line):
                if isinstance(arr_line[index], str):
                    word = arr_line[index].lower()
                    if word == "factor" and index + 1 < len(arr_line):
                        multiplier = convert_data(
                            arr_line[index + 1],
                            self._data_dimensions,
                            self._data_type,
                        )
                        index += 2
                    elif word == "iprn" and index + 1 < len(arr_line):
                        print_format = arr_line[index + 1]
                        index += 2
                    else:
                        break
                else:
                    break
        elif isinstance(arr_line, dict):
            for key, value in arr_line.items():
                if key.lower() == "iprn":
                    print_format = value
        return multiplier, print_format

    def _process_open_close_line(self, arr_line, store=True):
        # process open/close line
        index = 2
        if self._data_type == DatumType.integer:
            multiplier = 1
        else:
            multiplier = 1.0
        print_format = None
        binary = False
        data_file = None
        data = None

        data_dim = self._data_dimensions
        if isinstance(arr_line, list):
            if len(arr_line) < 2 and store:
                message = (
                    'Data array "{}" contains a OPEN/CLOSE '
                    "that is not followed by a file. {}".format(
                        data_dim.structure.name, data_dim.structure.path
                    )
                )
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self._data_dimensions.structure.get_model(),
                    self._data_dimensions.structure.get_package(),
                    self._data_dimensions.structure.path,
                    "processing open/close line",
                    data_dim.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    message,
                    self._simulation_data.debug,
                )
            while index < len(arr_line):
                if isinstance(arr_line[index], str):
                    word = arr_line[index].lower()
                    if word == "factor" and index + 1 < len(arr_line):
                        try:
                            multiplier = convert_data(
                                arr_line[index + 1],
                                self._data_dimensions,
                                self._data_type,
                            )
                        except Exception as ex:
                            message = (
                                "Data array {} contains an OPEN/CLOSE "
                                "with an invalid multiplier following "
                                'the "factor" keyword.'
                                ".".format(data_dim.structure.name)
                            )
                            type_, value_, traceback_ = sys.exc_info()
                            raise MFDataException(
                                self._data_dimensions.structure.get_model(),
                                self._data_dimensions.structure.get_package(),
                                self._data_dimensions.structure.path,
                                "processing open/close line",
                                data_dim.structure.name,
                                inspect.stack()[0][3],
                                type_,
                                value_,
                                traceback_,
                                message,
                                self._simulation_data.debug,
                                ex,
                            )
                        index += 2
                    elif word == "iprn" and index + 1 < len(arr_line):
                        print_format = arr_line[index + 1]
                        index += 2
                    elif word == "data" and index + 1 < len(arr_line):
                        data = arr_line[index + 1]
                        index += 2
                    elif word == "binary" or word == "(binary)":
                        binary = True
                        index += 1
                    else:
                        break
                else:
                    break
            if arr_line[0].lower() == "open/close":
                data_file = clean_filename(arr_line[1])
            else:
                data_file = clean_filename(arr_line[0])
        elif isinstance(arr_line, dict):
            for key, value in arr_line.items():
                if key.lower() == "factor":
                    try:
                        multiplier = convert_data(
                            value, self._data_dimensions, self._data_type
                        )
                    except Exception as ex:
                        message = (
                            "Data array {} contains an OPEN/CLOSE "
                            "with an invalid factor following the "
                            '"factor" keyword.'
                            ".".format(data_dim.structure.name)
                        )
                        type_, value_, traceback_ = sys.exc_info()
                        raise MFDataException(
                            self._data_dimensions.structure.get_model(),
                            self._data_dimensions.structure.get_package(),
                            self._data_dimensions.structure.path,
                            "processing open/close line",
                            data_dim.structure.name,
                            inspect.stack()[0][3],
                            type_,
                            value_,
                            traceback_,
                            message,
                            self._simulation_data.debug,
                            ex,
                        )
                if key.lower() == "iprn":
                    print_format = value
                if key.lower() == "binary":
                    binary = bool(value)
                if key.lower() == "data":
                    data = value
            if "filename" in arr_line:
                data_file = clean_filename(arr_line["filename"])

        #  add to active list of external files
        model_name = data_dim.package_dim.model_dim[0].model_name
        self._simulation_data.mfpath.add_ext_file(data_file, model_name)

        return multiplier, print_format, binary, data_file

    def _add_cellid_fields(self, data):
        for data_item in self.structure.data_item_structures:
            if data_item.type == DatumType.integer:
                if data_item.name.lower() == "cellid":
                    if isinstance(self._mg, StructuredGrid):
                        return data.assign(
                            cellid=lambda x: (x.layer, x.row, x.col))
                    elif isinstance(self._mg, VertexGrid):
                        return data.assign(
                            cellid=lambda x: (x.layer, x.cell))
                    elif isinstance(self._mg, UnstructuredGrid):
                        return data.assign(cellid=lambda x: (x.node,))
                    else:
                        raise FlopyException(
                            "ERROR: Unrecognized model grid "
                            "{str(self._mg)} not supported by MFBasicList")
        return data

    def _remove_cellid_fields(self, data):
        for data_item in self.structure.data_item_structures:
            if data_item.type == DatumType.integer:
                if data_item.name.lower() == "cellid":
                    # if there is a cellid field, remove it
                    if "cellid" in data.columns:
                        return data.drop("cellid", axis=1)
        return data

    def _get_cellid_size(self, data_item_name):
        model_num = DatumUtil.cellid_model_num(
            data_item_name,
            self._data_dimensions.structure.model_data,
            self._data_dimensions.package_dim.model_dim,
        )
        model_grid = self._data_dimensions.get_model_grid(model_num=model_num)
        return model_grid.get_num_spatial_coordinates()

    def _resolve_type(self, data_item, names, types, data_line, data_line_idx,
                      last_item, repeat_num=0):
        if repeat_num > 0 and len(data_line) <= data_line_idx:
            return data_line_idx
        current_name = data_item.name
        if repeat_num > 0:
            current_name = f"{current_name}_{repeat_num}"
        if data_item.name == "boundname" and not \
                self._data_dimensions.package_dim.boundnames():
            return data_line_idx
        elif data_item.name == "aux":
            aux_var_names = self._data_dimensions.package_dim.\
                get_aux_variables()
            if aux_var_names is not None and len(aux_var_names[0]) > 0:
                for name in aux_var_names[0]:
                    if name.lower() != "auxiliary":
                        names.append(name)
                        types.append(np.float64)
                        # types.append("f8")
                        data_line_idx += 1
                        if data_line_idx >= len(data_line):
                            return data_line_idx
            return data_line_idx
        elif data_item.type == DatumType.string:
            names.append(current_name)
            types.append(data_item.pandas_dtype)
            data_line_idx += 1
        elif data_item.type == DatumType.double_precision:
            names.append(current_name)
            types.append(data_item.pandas_dtype)
            data_line_idx += 1
        elif data_item.type == DatumType.keyword:
            names.append(current_name)
            types.append(data_item.pandas_dtype)
            data_line_idx += 1
        elif data_item.type == DatumType.keystring:
            # determine keystring type and switch to the keystring
            # definition data item list
            ks_key = data_line[data_line_idx].lower()
            if ks_key in data_item.keystring_dict:
                data_item_ks = data_item.keystring_dict[ks_key]
            else:
                ks_key = f"{ks_key}record"
                if ks_key in data_item.keystring_dict:
                    data_item_ks = data_item.keystring_dict[ks_key]
                else:
                    return data_line_idx
            if isinstance(data_item_ks, MFDataStructure):
                dis = data_item_ks.data_item_structures
                for data_item in dis:
                    data_line_idx = \
                        self._resolve_type(data_item, names, types,
                                           data_line, data_line_idx,
                                           last_item)
                    if data_line_idx >= len(data_line):
                        return data_line_idx
            else:
                names.append(f"{data_item_ks.name}_text")
                types.append(object)
                data_line_idx += 1
                data_line_idx = \
                    self._resolve_type(data_item_ks, names, types,
                                       data_line, data_line_idx,
                                       last_item)
        elif data_item.type == DatumType.integer:
            if data_item.name.lower() == "cellid":
                if isinstance(data_line[data_line_idx], tuple):
                    # clean up data line
                    raise Exception("CellID tuple conversion not "
                                    "supported yet")
                elif not isinstance(data_line[data_line_idx], int) and \
                        data_line[data_line_idx].upper() == "NONE":
                    names.append("cell")
                    types.append(object)
                    data_line_idx += 1
                else:
                    # determine cellid size
                    cellid_size = self._get_cellid_size(data_item.name)
                    if cellid_size == 1:
                        names.append("node")
                        types.append(np.int64)
                        data_line_idx += 1
                    elif cellid_size == 2:
                        names.append("layer")
                        types.append(np.int64)
                        names.append("cpl")
                        types.append(np.int64)
                        data_line_idx += 2
                    elif cellid_size == 3:
                        names.append("layer")
                        types.append(np.int64)
                        names.append("row")
                        types.append(np.int64)
                        names.append("column")
                        types.append(np.int64)
                        data_line_idx += 3
            else:
                names.append(current_name)
                types.append(np.int64)
                data_line_idx += 1
        if data_item.repeating and last_item:
            data_line_idx = \
                self._resolve_type(data_item, names, types, data_line,
                                   data_line_idx, last_item, repeat_num + 1)
        return data_line_idx

    def _resolve_column_name_type_list(self, data_line):
        names = []
        types = []
        data_line_idx = 0
        data_line_len = len(data_line)
        data_struct_len = len(self.structure.data_item_structures)
        for idx, data_item in enumerate(self.structure.data_item_structures):
            if data_line_idx >= data_line_len:
                return names, types
            last_item = idx + 1 == data_struct_len
            data_line_idx = self._resolve_type(data_item, names, types,
                                               data_line, data_line_idx,
                                               last_item)
        return names, types

    def _separate_data(self, data, data_types=None):
        # separate data into lists that share the same column headings
        # and types
        if data_types is None:
            data_types = {}
        for line in data:
            current_type = []
            for datum in line:
                current_type.append(type(datum))
            current_type = tuple(current_type)
            if current_type not in data_types:
                #names, types = self._resolve_column_name_type_list(line)
                #type_list = list(zip(names, types))
                data_types[current_type] = [line]
                #data_types[current_type] = pandas.DataFrame([line],
                #                                            columns=type_list)
                #                                            #dtype=type_list)
            else:
                data_types[current_type].append(line)
                #data_types[current_type].loc[len(data_types[current_type])] = \
                #        line
        panda_data_types = {}
        for dtype, data in data_types.items():
            names, types = self._resolve_column_name_type_list(data[0])
            # type_list = list(zip(names, types))
            panda_data_types[dtype] = pandas.DataFrame(data,
                                                       columns=names)

        return panda_data_types

    def _build_data_header(self):
        self._data_header = []
        self._header_names = []
        s_type = pandas.StringDtype
        #f_type = pandas.Float64Dtype
        #f_type = pandas.core.arrays.float.Float64Dtype
        f_type = np.float64
        #i_type = pandas.Int64Dtype
        #i_type = pandas.core.arrays.integer.Int64Dtype
        i_type = np.int64
        data_dim = self._data_dimensions
        for data_item, index in zip(
            self.structure.data_item_structures,
            range(0, len(self.structure.data_item_structures)),
        ):
            if data_item.name.lower() == "aux":
                aux_var_names = data_dim.package_dim.get_aux_variables()
                if aux_var_names is not None:
                    for aux_var_name in aux_var_names[0]:
                        if aux_var_name.lower() != "auxiliary":
                            self._append_type_list(aux_var_name, f_type)
            elif data_item.name.lower() == "boundname":
                if data_dim.package_dim.boundnames():
                    self._append_type_list("boundname", s_type)
            else:
                if data_item.type == DatumType.keyword:
                    self._append_type_list(data_item.name, s_type)
                elif data_item.type == DatumType.string:
                    self._append_type_list(data_item.name, s_type)
                elif data_item.type == DatumType.integer:
                    if data_item.name.lower() == "cellid":
                        if isinstance(self._mg, StructuredGrid):
                            self._append_type_list("layer", i_type)
                            self._append_type_list("row", i_type)
                            self._append_type_list("column", i_type)
                        elif isinstance(self._mg, VertexGrid):
                            self._append_type_list("layer", i_type)
                            self._append_type_list("cell", i_type)
                        elif isinstance(self._mg, UnstructuredGrid):
                            self._append_type_list("node", i_type)
                        else:
                            raise FlopyException(
                                "ERROR: Unrecognized model grid "
                                "{str(self._mg)} not supported by MFBasicList")
                    else:
                        self._append_type_list(data_item.name, i_type)
                elif data_item.type == DatumType.double_precision:
                    self._append_type_list(data_item.name, f_type)
                else:
                    self._data_header = None
                    self._header_names = None
                    #raise FlopyException(f"ERROR: Data type {data_item.type} "
                    #                     "not supported by MFBasicList",
                    #                     "get_data_header")

    def _set_data(self, data, check_data=True):
        # TODO: Convert any cellid tuple to individual integers
        # (re)build data header
        self._build_data_header()
        if isinstance(data, dict) and not self.has_data():
            self._set_record(data)
            return
        if isinstance(data, np.recarray) and self._header_names is not None:
            # verify data shape
            if len(data[0]) != len(self._header_names):
                raise FlopyException(f"ERROR: Data list {self._data_name} "
                                     " supplied the wrong number of columns "
                                     "of data, expected "
                                     f"{len(self._data_header)} got "
                                     f"{len(data[0])}.")
#            data = pandas.DataFrame(data).set_index('f0')
            data = self._separate_data(data)
            # data = [pandas.DataFrame(data)]
        elif isinstance(data, list) or isinstance(data, tuple):
            if not (isinstance(data[0], list) or isinstance(data[0], tuple)):
                data = [data]
            # verify data shape
            #if self._header_names is not None and \
            #        len(data[0]) != len(self._header_names):
            #    raise FlopyException(f"ERROR: Data list {self._data_name} "
            #                         " supplied the wrong number of columns "
            #                         "of data, expected "
            #                         f"{len(self._data_header)} got "
            #                         f"{len(data[0])}.")
            data = self._separate_data(data)
            # data = [pandas.DataFrame(data, columns=self._header_names)]
#            data = pandas.DataFrame(data, self._header_names).set_index('f0')

        data_storage = self._get_storage()
        """
        for df in data.values():
            if isinstance(df, pandas.DataFrame):
                data_types = df.dtypes
                # verify shape of data
                if len(data_types) != len(self._data_header):
                    raise FlopyException(f"ERROR: Data list {self._data_name} "
                                         " supplied the wrong number of columns "
                                         "of data, expected "
                                         f"{len(self._data_header)} got "
                                         f"{len(data_types)}.")
                col_rename = {}
                for header_column, data_column_type, data_column_name in \
                        zip(self._data_header, data_types.values, df.columns):
                    # verify column types
                    if header_column[1] != data_column_type:
                        raise FlopyException(f"ERROR: Data list {self._data_name} "
                                             " supplied the wrong type for column "
                                             f"{header_column[0]}.  Expected "
                                             f"{header_column[1]} got "
                                             f"{data_column_type}.")
                    # verify column header text
                    if header_column[0] != data_column_name:
                        # build rename dictionary
                        col_rename[data_column_name] = header_column[0]
                if len(col_rename) > 0:
                    # rename columns to the names given in the definition files
                    df.rename(columns=col_rename)
            else:
                # convert to pandas dataframe
                print("Not yet supported.")
        """
        if data_storage.data_storage_type == \
                DataStorageType.external_file:
            data_storage.merge_data(data)
        else:
            data_storage.set_internal(data, True)
            data_storage.modified = True

    def has_modified_ext_data(self):
        data_storage = self._get_storage()
        return data_storage.data_storage_type == \
            DataStorageType.external_file \
            and data_storage.internal_data is not None

    def binary_ext_data(self):
        data_storage = self._get_storage()
        return data_storage.binary

    def to_array(self, kper=0, mask=False):
        """Convert stress period boundary condition (MFDataList) data for a
        specified stress period to a 3-D numpy array.

        Parameters
        ----------
        kper : int
            MODFLOW zero-based stress period number to return. (default is
            zero)
        mask : bool
            return array with np.NaN instead of zero

        Returns
        ----------
        out : dict of numpy.ndarrays
            Dictionary of 3-D numpy arrays containing the stress period data
            for a selected stress period. The dictionary keys are the
            MFDataList dtype names for the stress period data."""
        i0 = 1
        sarr = self.get_data(key=kper)[0]
        if not isinstance(sarr, list):
            sarr = [sarr]
        if len(sarr) == 0 or sarr[0] is None:
            return None
        if "inode" in sarr[0].dtype.names:
            raise NotImplementedError()
        arrays = {}
        model_grid = self._data_dimensions.get_model_grid()

        if model_grid._grid_type.value == 1:
            shape = (
                model_grid.num_layers(),
                model_grid.num_rows(),
                model_grid.num_columns(),
            )
        elif model_grid._grid_type.value == 2:
            shape = (
                model_grid.num_layers(),
                model_grid.num_cells_per_layer(),
            )
        else:
            shape = (model_grid.num_cells_per_layer(),)

        for name in sarr[0].dtype.names[i0:]:
            if not sarr[0].dtype.fields[name][0] == object:
                arr = np.zeros(shape)
                arrays[name] = arr.copy()

        if np.isscalar(sarr[0]):
            # if there are no entries for this kper
            if sarr[0] == 0:
                if mask:
                    for name, arr in arrays.items():
                        arrays[name][:] = np.NaN
                return arrays
            else:
                raise Exception("MfList: something bad happened")

        for name, arr in arrays.items():
            cnt = np.zeros(shape, dtype=np.float64)
            for sp_rec in sarr:
                if sp_rec is not None:
                    for rec in sp_rec:
                        arr[rec["cellid"]] += rec[name]
                        cnt[rec["cellid"]] += 1.0
            # average keys that should not be added
            if name != "cond" and name != "flux":
                idx = cnt > 0.0
                arr[idx] /= cnt[idx]
            if mask:
                arr = np.ma.masked_where(cnt == 0.0, arr)
                arr[cnt == 0.0] = np.NaN

            arrays[name] = arr.copy()
        # elif mask:
        #     for name, arr in arrays.items():
        #         arrays[name][:] = np.NaN
        return arrays

    def set_data(self, data, autofill=False, check_data=True):
        """Sets the contents of the data to "data".  Data can have the
        following formats:
            1) recarray - recarray containing the datalist
            2) [(line_one), (line_two), ...] - list where each line of the
               datalist is a tuple within the list
        If the data is transient, a dictionary can be used to specify each
        stress period where the dictionary key is <stress period> - 1 and
        the dictionary value is the datalist data defined above:
        {0:ndarray, 1:[(line_one), (line_two), ...], 2:{'filename':filename})

        Parameters
        ----------
            data : ndarray/list/dict
                Data to set
            autofill : bool
                Automatically correct data
            check_data : bool
                Whether to verify the data

        """
        self._set_data(data, check_data=check_data)

    def set_record(self, data_record, autofill=False, check_data=True):
        """Sets the contents of the data and metadata to "data_record".
        Data_record is a dictionary with has the following format:
            {'filename':filename, 'binary':True/False, 'data'=data}
        To store to file include 'filename' in the dictionary.

        Parameters
        ----------
            data_record : ndarray/list/dict
                Data and metadata to set
            autofill : bool
                Automatically correct data
            check_data : bool
                Whether to verify the data

        """
        self._set_record(data_record, autofill, check_data)

    def _set_record(self, data_record, autofill=False, check_data=True):
        """Sets the contents of the data and metadata to "data_record".
        Data_record is a dictionary with has the following format:
            {'filename':filename, 'data'=data}
        To store to file include 'filename' in the dictionary.

        Parameters
        ----------
            data_record : ndarray/list/dict
                Data and metadata to set
            autofill : bool
                Automatically correct data
            check_data : bool
                Whether to verify the data

        """
        if isinstance(data_record, dict):
            data_storage = self._get_storage()
            if "filename" in data_record:
                data_storage.set_external(data_record["filename"])
                if "binary" in data_record:
                    if data_record["binary"] and \
                            self._data_dimensions.package_dim.boundnames():
                        message = (
                            "Unable to store list data ({}) to a binary "
                            "file when using boundnames"
                            ".".format(self._data_dimensions.structure.name)
                        )
                        type_, value_, traceback_ = sys.exc_info()
                        raise MFDataException(
                            self._data_dimensions.structure.get_model(),
                            self._data_dimensions.structure.get_package(),
                            self._data_dimensions.structure.path,
                            "writing list data to binary file",
                            self._data_dimensions.structure.name,
                            inspect.stack()[0][3],
                            type_,
                            value_,
                            traceback_,
                            message,
                            self._simulation_data.debug,
                        )
                    data_storage.binary = data_record["binary"]
                if "data" in data_record:
                    # data gets written out to file
                    self._set_data(data_record["data"])
            else:
                if "data" in data_record:
                    data_storage.modified = True
                    data_storage.set_internal(None)
                    self._set_data(data_record["data"])
            if "iprn" in data_record:
                data_storage.iprn = data_record["iprn"]

    def append_data(self, data):
        """Appends "data" to the end of this list.  Assumes data is in a format
        that can be appended directly to a numpy recarray.

        Parameters
        ----------
            data : list(tuple)
                Data to append.

        """
        try:
            self._resync()
            if self._get_storage() is None:
                self._data_storage = self._new_storage()
            data_storage = self._get_storage()
            if data_storage.data_storage_type == \
                    DataStorageType.internal_array:
                # update internal data
                if data_storage.internal_data is not None:
                    data_storage.internal_data = \
                        self._separate_data(data, data_storage.internal_data)
                else:
                    data_storage.internal_data = self._separate_data(data)
                # self._build_data_header()
                #data_storage.internal_data = data_storage.internal_data.append(
                #    pandas.DataFrame(data, columns=self._header_names),
                #    ignore_index=True)
            elif data_storage.data_storage_type == \
                    DataStorageType.external_file:
                # get external data from file
                external_data = self.get_data()
                # append data to external data
                external_data = self._separate_data(data, external_data)
                # external_data = external_data.append(
                #    pandas.DataFrame(data, columns=self._header_names),
                #    ignore_index=True)
                # get an update the external record
                ext_record = self.get_record()
                ext_record["data"] = external_data
                self.set_record(ext_record)
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.structure.get_model(),
                self.structure.get_package(),
                self._path,
                "appending data",
                self.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                None,
                self._simulation_data.debug,
                ex,
            )

    def append_list_as_record(self, record):
        """Appends the list `record` as a single record in this list's
        recarray.  Assumes "data" has the correct dimensions.

        Parameters
        ----------
            record : list
                List to be appended as a single record to the data's existing
                recarray.

        """
        self._resync()
        try:
            # convert to tuple
            tuple_record = ()
            for item in record:
                tuple_record += (item,)
            # store
            self.append_data([tuple_record])
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.structure.get_model(),
                self.structure.get_package(),
                self._path,
                "appending data",
                self.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                None,
                self._simulation_data.debug,
                ex,
            )

    def update_record(self, record, key_index):
        """Updates a record at index "key_index" with the contents of "record".
        If the index does not exist update_record appends the contents of
        "record" to this list's recarray.

        Parameters
        ----------
            record : list
                New record to update data with
            key_index : int
                Stress period key of record to update.  Only used in transient
                data types.
        """
        self.append_list_as_record(record)

    def store_internal(
        self,
        check_data=True,
    ):
        """Store all data internally.

        Parameters
        ----------
            check_data : bool
                Verify data prior to storing

        """
        storage = self._get_storage()
        # check if data is already stored external
        if (
            storage is None
            or storage.layer_storage.first_item().data_storage_type
            == DataStorageType.external_file
        ):
            data = self._get_data()
            # if not empty dataset
            if data is not None:
                if (
                    self._simulation_data.verbosity_level.value
                    >= VerbosityLevel.verbose.value
                ):
                    print(f"Storing {self.structure.name} internally...")
                internal_data = {
                    "data": data,
                }
                self._set_record(internal_data, check_data=check_data)

    def store_as_external_file(
        self,
        external_file_path,
        binary=False,
        replace_existing_external=True,
        check_data=True,
    ):
        """Store all data externally in file external_file_path. the binary
        allows storage in a binary file. If replace_existing_external is set
        to False, this method will not do anything if the data is already in
        an external file.

        Parameters
        ----------
            external_file_path : str
                Path to external file
            binary : bool
                Store data in a binary file
            replace_existing_external : bool
                Whether to replace an existing external file.
            check_data : bool
                Verify data prior to storing

        """
        # only store data externally (do not subpackage info)
        if self.structure.construct_package is None:
            storage = self._get_storage()
            # check if data is already stored external
            if (
                replace_existing_external
                or storage is None
                or storage.layer_storage.first_item().data_storage_type
                == DataStorageType.internal_array
                or storage.layer_storage.first_item().data_storage_type
                == DataStorageType.internal_constant
            ):
                data = self._get_data()
                # if not empty dataset
                if data is not None:
                    if (
                        self._simulation_data.verbosity_level.value
                        >= VerbosityLevel.verbose.value
                    ):
                        print(
                            "Storing {} to external file {}.."
                            ".".format(self.structure.name, external_file_path)
                        )
                    external_data = {
                        "filename": external_file_path,
                        "data": data,
                        "binary": binary,
                    }
                    self._set_record(external_data, check_data=check_data)

    def external_file_name(self):
        """Returns external file name, or None if this is not external data.

        Parameters
        ----------
            key : int
                Zero based stress period to return data from.
        """
        storage = self._get_storage()
        if storage is None:
            return None
        if storage.data_storage_type == DataStorageType.external_file and \
                storage.fname is not None and storage.fname != "":
            return storage.fname
        return None

    def _obj_to_pandas_dtype(self, obj):
        if isinstance(obj, int) or isinstance(obj, np.int64) or \
                isinstance(obj, np.int32):
            return np.int64
        elif isinstance(obj, float) or isinstance(obj, np.float64) or \
                isinstance(obj, np.float32):
            return np.float64
        else:
            return object

    def _str_dtype_to_pandas_dtype(self, str_dtype):
        if str_dtype == "int64":
            return np.int64
        elif str_dtype == "float64":
            return np.float64
        else:
            return object
            #return "S10"

    def _read_data_header(self, data_line):
        # verify that this is a flopy header
        data_line = data_line.strip()
        if len(data_line) < 5 or data_line[0] != "#" or \
                data_line[-3:] != "fph":
            # not header
            return None, None, None, None
        # process header
        data_line_lst = shlex.split(data_line)
        if len(data_line_lst) < 3 or not \
                datautil.DatumUtil.is_int(data_line_lst[-2]):
            # not header
            return None, None, None, None
        names = []
        dtypes = {}
        num_rows = int(data_line_lst[-3])
        header_continued = int(data_line_lst[-2])
        for entry in data_line_lst[1:-3]:
            name, dtype = entry.split(":")
            names.append(name)
            dtypes[name] = self._str_dtype_to_pandas_dtype(dtype)
            # TODO: verify correct names and types

        return names, dtypes, num_rows, header_continued

    def _read_text_data(self, fd_data_file, next_line, external_file=False):
        # read data header
        if next_line is None:
            next_line = fd_data_file.readline()
        continued = 1
        header_list = []
        while continued == 1:
            names, dtypes, num_rows, continued = \
                self._read_data_header(next_line)
            if names is not None:
                header_list.append(ListHeader(names, dtypes, num_rows))
            if continued == 1:
                next_line = fd_data_file.readline()
        #names.insert(0, "leading_space")

        if len(header_list) == 0:
            # read user formatted data using MFList class
            list_data = MFList(
                self._simulation_data,
                self._model_or_sim,
                self.structure,
                None,
                True,
                self.path,
                self._data_dimensions.package_dim,
                self._package,
                self._block
            )
            return_val = list_data.load(
                next_line,
                fd_data_file,
                self._block.block_headers[-1]
            )
            rec_array = list_data.get_data()
            if rec_array is None:
                return None, [False, None]
            data_frame = pandas.DataFrame(rec_array)
            data_frame_list = [data_frame]
            #data_frame = pandas.DataFrame(rec_array).set_index('f0')
        else:
            data_frame_list = []
            #names.insert(0, "leading_space")
            #dtypes["leading_space"] = "S1"
            for idx, header in enumerate(header_list):
                # read flopy formatted data
                data_frame = pandas.read_csv(fd_data_file, sep=" ",
                                             names=header.names,
                                             dtype=header.types,
                                             nrows=header.num_rows,
                                             index_col=False,
                                             skipinitialspace=True)
                if self.path in self._simulation_data.data_offset_location:
                    # fix file location to end of read
                    fd_data_file.seek(
                        self._simulation_data.data_offset_location[self.path]
                        [self._current_key][idx][1]
                    )
                #data_frame = data_frame.drop(columns="leading_space")
                self._decrement_id_fields(data_frame)
                data_frame_list.append(data_frame)
            return_val = [True, fd_data_file.readline()]
        return data_frame_list, return_val

    """
    def _get_external_path(self):
        data_storage = self._get_storage()
        model_name = self._data_dimensions.package_dim.model_dim[
            0].model_name
        return self._simulation_data.mfpath.resolve_path(
            data_storage.fname, model_name
        )
    """

    def _save_binary_data(self, fd_data_file, data):
        # write
        file_access = MFFileAccessList(
            self.structure,
            self._data_dimensions,
            self._simulation_data,
            self._path,
            self._current_key,
        )
        self._increment_id_fields(data)
        file_access.write_binary_file(
            data,
            fd_data_file,
            self._model_or_sim.modeldiscrit
        )
        self._decrement_id_fields(data)
        data_storage = self._get_storage()
        data_storage.internal_data = None

    def has_data(self, key=None):
        """Returns whether this MFList has any data associated with it."""
        try:
            if self._get_storage() is None:
                return False
            return self._get_storage().has_data()
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.structure.get_model(),
                self.structure.get_package(),
                self._path,
                "checking for data",
                self.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                None,
                self._simulation_data.debug,
                ex,
            )

    def _load_external_data(self, data_storage):
        #file_path = self._get_external_path()
        file_path = self._resolve_ext_file_path(data_storage)
        # parse next line in file as data header
        if data_storage.binary:
            file_access = MFFileAccessList(
                self.structure,
                self._data_dimensions,
                self._simulation_data,
                self._path,
                self._current_key,
            )
            np_data = file_access.read_binary_data_from_file(
                file_path,
                self._model_or_sim.modeldiscrit,
                inc_index=False,
            )
            pd_data = pandas.DataFrame(np_data)
            self._decrement_id_fields(pd_data)
        else:
            with open(file_path, "r") as fd_data_file:
                pd_data, return_val = self._read_text_data(fd_data_file, None,
                                                           True)
        return pd_data

    """
    def _load_from_external(self, data_storage):
        # load with pandas
        file_path = self._resolve_ext_file_path(data_storage)
        with open(file_path, "r") as fd_ext:
            pd_data, ret_val = self._read_text_data(
                fd_ext,
                None
            )
        return pd_data
        """

    def load(
        self,
        first_line,
        file_handle,
        block_header,
        pre_data_comments=None,
        external_file_info=None,
    ):
        """Loads data from first_line (the first line of data) and open file
        file_handle which is pointing to the second line of data.  Returns a
        tuple with the first item indicating whether all data was read
        and the second item being the last line of text read from the file.
        This method was only designed for internal FloPy use and is not
        recommended for end users.

        Parameters
        ----------
            first_line : str
                A string containing the first line of data in this list.
            file_handle : file descriptor
                A file handle for the data file which points to the second
                line of data for this list
            block_header : MFBlockHeader
                Block header object that contains block header information
                for the block containing this data
            pre_data_comments : MFComment
                Comments immediately prior to the data
            external_file_info : list
                Contains information about storing files externally
        Returns
        -------
            more data : bool,
            next data line : str

        """
        data_storage = self._get_storage()
        data_storage.modified = False
        # parse first line to determine if this is internal or external data
        datautil.PyListUtil.reset_delimiter_used()
        arr_line = datautil.PyListUtil.split_data_line(first_line)
        if arr_line and (
            len(arr_line[0]) >= 2 and arr_line[0][:3].upper() == "END"
        ):
            return [False, arr_line]
        if len(arr_line) >= 2 and arr_line[0].upper() == "OPEN/CLOSE":
            try:
                multiplier, iprn, binary, data_file = \
                    self._process_open_close_line(arr_line)
            except Exception as ex:
                message = (
                    "An error occurred while processing the following "
                    "open/close line: {}".format(arr_line)
                )
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self._path,
                    "processing open/close line",
                    self.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    message,
                    self._simulation_data.debug,
                    ex,
                )
            data_storage.set_external(data_file)
            data_storage.binary = binary
            data_storage.iprn = iprn
            return_val = [False, None]
        # else internal
        else:
            # parse first line as data header
            #multiplier, iprn = \
            #    self._process_internal_line(arr_line)
            # read data into pandas dataframe
            pd_data, return_val = self._read_text_data(file_handle, first_line,
                                                       True)
            # verify this is the end of the block?

            # store internal data
            data_storage.set_internal(pd_data)
        return return_val

    def _new_storage(self):
        return BasicListStorage()

    def _get_storage(self):
        return self._data_storage

    def _set_storage(self, data):
        self._data_storage.internal_data = data

    def _get_id_fields(self):
        id_fields = []
        for data_item_struct in self.structure.data_item_structures:
            if data_item_struct.numeric_index or data_item_struct.is_cellid:
                if data_item_struct.name.lower() == "cellid":
                    if isinstance(self._mg, StructuredGrid):
                        id_fields.append("layer")
                        id_fields.append("row")
                        id_fields.append("column")
                    elif isinstance(self._mg, VertexGrid):
                        id_fields.append("layer")
                        id_fields.append("cell")
                    elif isinstance(self._mg, UnstructuredGrid):
                        id_fields.append("node")
                    else:
                        raise FlopyException(
                            "ERROR: Unrecognized model grid "
                            "{str(self._mg)} not supported by MFBasicList")
                else:
                    id_fields.append(data_item_struct.name)
        return id_fields

    def _increment_id_fields(self, data_frame):
        for id_field in self._get_id_fields():
            if id_field in data_frame:
                data_frame[id_field] += 1

    def _decrement_id_fields(self, data_frame):
        for id_field in self._get_id_fields():
            if id_field in data_frame:
                data_frame[id_field] -= 1

    def _resolve_ext_file_path(self, data_storage):
        # pathing to external file
        data_dim = self._data_dimensions
        model_name = data_dim.package_dim.model_dim[0].model_name
        fp_relative = data_storage.fname
        if model_name is not None and fp_relative is not None:
            rel_path = self._simulation_data.mfpath.model_relative_path[
                model_name
            ]
            if rel_path is not None and len(rel_path) > 0 and rel_path != ".":
                # include model relative path in external file path
                # only if model relative path is not already in external
                #  file path i.e. when reading!
                fp_rp_l = fp_relative.split(os.path.sep)
                rp_l_r = rel_path.split(os.path.sep)[::-1]
                for i, rp in enumerate(rp_l_r):
                    if rp != fp_rp_l[len(rp_l_r) - i - 1]:
                        fp_relative = os.path.join(rp, fp_relative)
            fp = self._simulation_data.mfpath.resolve_path(
                fp_relative, model_name
            )
        else:
            fp = os.path.join(
                self._simulation_data.mfpath.get_sim_path(), fp_relative
            )
        return fp

    def _get_data(self, apply_mult=False, **kwargs):
        data_storage = self._get_storage()
        if data_storage is None or data_storage.data_storage_type is None:
            return None
        if data_storage.data_storage_type == \
                DataStorageType.internal_array:
            data = data_storage.internal_data
        else:
            # load data from file and return
            data = self._load_external_data(data_storage)
        return data
        #return self._add_cellid_fields(data)

    def get_data(self, apply_mult=False, **kwargs):
        """Returns the list's data.

        Parameters
        ----------
            apply_mult : bool
                Whether to apply a multiplier.

        Returns
        -------
            data : recarray

        """
        return self._get_data(apply_mult, **kwargs)

    def get_record(self):
        """Returns the list's data and metadata in a dictionary.  Data is in
        key "data" and metadata in keys "filename" and "binary".

        Returns
        -------
            data_record : dict

        """
        try:
            if self._get_storage() is None:
                return None
            record = self._get_storage().get_record()
        except Exception as ex:
            type_, value_, traceback_ = sys.exc_info()
            raise MFDataException(
                self.structure.get_model(),
                self.structure.get_package(),
                self._path,
                "getting record",
                self.structure.name,
                inspect.stack()[0][3],
                type_,
                value_,
                traceback_,
                None,
                self._simulation_data.debug,
                ex,
            )
        #if record["internal_data"] is not None:
        #    record["internal_data"] = \
        #        self._add_cellid_fields(record["internal_data"])

    def write_pre_entry(
        self,
        fd_data_file,
        ext_file_action=ExtFileAction.copy_relative_paths,
    ):
        data_storage = self._get_storage()
        if data_storage is None:
            return None
        if data_storage.data_storage_type == DataStorageType.internal_array:
            return DataStorageType.internal_array
        fd_data_file.write(f"open/close {data_storage.fname}\n")
        return data_storage.data_storage_type

    def write_file_entry(
        self,
        fd_data_file,
        ext_file_action=ExtFileAction.copy_relative_paths,
        fd_main=None,
    ):
        """Returns a string containing the data formatted for a MODFLOW 6
        file.

        Parameters
        ----------
            ext_file_action : ExtFileAction
                How to handle external paths.

        Returns
        -------
            file entry : str

        """
        return self._write_file_entry(fd_data_file, ext_file_action, fd_main)

    def get_file_entry(
        self,
        ext_file_action=ExtFileAction.copy_relative_paths,
    ):
        """Returns a string containing the data formatted for a MODFLOW 6
        file.

        Parameters
        ----------
            ext_file_action : ExtFileAction
                How to handle external paths.

        Returns
        -------
            file entry : str

        """
        # TODO: Fix this or remove
        return self._write_file_entry(None)

    def _write_file_entry(
        self,
        fd_data_file,
        ext_file_action=ExtFileAction.copy_relative_paths,
        fd_main=None,
    ):
        data_storage = self._get_storage()
        if data_storage is None or data_storage.internal_data is None:
            return ""
        if data_storage.data_storage_type == DataStorageType.external_file \
                and fd_main is not None:
            indent = self._simulation_data.indent_string
            ext_string, fname = self._get_external_formatting_str(
                data_storage.fname,
                None,
                data_storage.binary,
                data_storage.iprn,
                DataStructureType.recarray,
                ext_file_action
            )
            data_storage.fname = fname
            fd_main.write(f"{indent}{indent}{ext_string}")
        # Loop through data pieces
        data_list = []
        num_data_pieces = len(data_storage.internal_data)
        for idx, int_data in enumerate(data_storage.internal_data.values()):
            data = self._remove_cellid_fields(int_data)
            data_list.append(data)
            if data_storage.data_storage_type == \
                    DataStorageType.internal_array:
                # build file entry data header
                column_defs = []
                for column_heading, column_dtype in zip(data.columns,
                                                        data.dtypes):
                    column_defs.append(f"{column_heading}:"
                                       f"{str(column_dtype)}")
                column_def_text = '" "'.join(column_defs)
                num_rows = data.shape[0]
                if idx + 1 < num_data_pieces:
                    more_dat = 1
                else:
                    more_dat = 0
                header = f'  # "{column_def_text}" {num_rows} {more_dat} fph\n'
                # write header
                fd_data_file.write(header)

        result = ""
        for data in data_list:
            # if data is internal or has been modified
            if (data_storage.data_storage_type ==
                    DataStorageType.internal_array or data is not None):
                if data_storage.data_storage_type == \
                        DataStorageType.external_file and data_storage.binary:
                    # write old way using numpy
                    self._save_binary_data(fd_data_file, data)
                else:
                    # convert data to 1-based
                    self._increment_id_fields(data)
                    # add spacer column
                    data.insert(loc=0, column='leading_space', value="")
                    data.insert(loc=0, column='leading_space_2', value="")
                    # write converted data
                    data_location_start = fd_data_file.tell()
                    float_format = f"%{self._simulation_data.reg_format_str[2:-1]}"
                    result = data.to_csv(fd_data_file, sep=" ",
                                         header=False, index=False,
                                         float_format=float_format,
                                         lineterminator="\n")
                                         #lineterminator=f"{os.linesep}")
                    data_location_end = fd_data_file.tell()
                    offset_loc = self._simulation_data.data_offset_location
                    if self.path not in offset_loc:
                        offset_loc[self.path] = {}
                    if self._current_key not in offset_loc[self.path]:
                        offset_loc[self.path][self._current_key] = []
                    offset_loc[self.path][self._current_key].append(
                        [data_location_start, data_location_end])
                    # clean up
                    data = data.drop(columns="leading_space")
                    data = data.drop(columns="leading_space_2")
                    data_storage.modified = False
                    self._decrement_id_fields(data)
                    if data_storage.data_storage_type == \
                            DataStorageType.external_file:
                        data_storage.internal_data = None
        return result


class MFBasicTransientList(MFBasicList, mfdata.MFTransient, DataListInterface):
    """
    Provides an interface for the user to access and update MODFLOW transient
    list data.

    Parameters
    ----------
    sim_data : MFSimulationData
        data contained in the simulation
    structure : MFDataStructure
        describes the structure of the data
    enable : bool
        enable/disable the array
    path : tuple
        path in the data dictionary to this MFArray
    dimensions : MFDataDimensions
        dimension information related to the model, package, and array

    """

    def __init__(
        self,
        sim_data,
        model_or_sim,
        structure,
        enable=True,
        path=None,
        dimensions=None,
        package=None,
        block=None,
    ):
        super().__init__(
            sim_data=sim_data,
            model_or_sim=model_or_sim,
            structure=structure,
            data=None,
            enable=enable,
            path=path,
            dimensions=dimensions,
            package=package,
            block=block,
        )
        self._transient_setup(self._data_storage)
        self.repeating = True
        self.empty_keys = {}

    @property
    def data_type(self):
        return DataType.transientlist

    @property
    def dtype(self):
        data = self.get_data()
        if len(data) > 0:
            if 0 in data:
                return data[0].dtype
            else:
                return next(iter(data.values())).dtype
        else:
            return None

    @property
    def data(self):
        """Returns list data.  Calls get_data with default parameters."""
        return self.get_data()

    def to_array(self, kper=0, mask=False):
        """Returns list data as an array."""
        return super().to_array(kper, mask)

    def remove_transient_key(self, transient_key):
        """Remove transient stress period key.  Method is used
        internally by FloPy and is not intended to the end user.

        """
        if transient_key in self._data_storage:
            del self._data_storage[transient_key]

    def add_transient_key(self, transient_key):
        """Adds a new transient time allowing data for that time to be stored
        and retrieved using the key `transient_key`.  Method is used
        internally by FloPy and is not intended to the end user.

        Parameters
        ----------
            transient_key : int
                Zero-based stress period to add

        """
        super().add_transient_key(transient_key)
        if isinstance(transient_key, int):
            stress_period = transient_key
        else:
            stress_period = 1
        self._data_storage[transient_key] = super()._new_storage()

    def store_as_external_file(
        self,
        external_file_path,
        binary=False,
        replace_existing_external=True,
        check_data=True,
    ):
        """Store all data externally in file external_file_path. the binary
        allows storage in a binary file. If replace_existing_external is set
        to False, this method will not do anything if the data is already in
        an external file.

        Parameters
        ----------
            external_file_path : str
                Path to external file
            binary : bool
                Store data in a binary file
            replace_existing_external : bool
                Whether to replace an existing external file.
            check_data : bool
                Verify data prior to storing

        """
        self._cache_model_grid = True
        for sp in self._data_storage.keys():
            self._current_key = sp
            storage = self._get_storage()
            if storage.internal_size > 0 and (
                self._get_storage().data_storage_type
                != DataStorageType.external_file
                or replace_existing_external
            ):
                fname, ext = os.path.splitext(external_file_path)
                if datautil.DatumUtil.is_int(sp):
                    full_name = f"{fname}_{sp + 1}{ext}"
                else:
                    full_name = f"{fname}_{sp}{ext}"

                super().store_as_external_file(
                    full_name,
                    binary,
                    replace_existing_external,
                    check_data,
                )
        self._cache_model_grid = False

    def store_internal(
        self,
        check_data=True,
    ):
        """Store all data internally.

        Parameters
        ----------
            check_data : bool
                Verify data prior to storing

        """
        self._cache_model_grid = True
        for sp in self._data_storage.keys():
            self._current_key = sp
            if (
                self._get_storage().layer_storage[0].data_storage_type
                == DataStorageType.external_file
            ):
                super().store_internal(
                    check_data,
                )
        self._cache_model_grid = False

    def has_data(self, key=None):
        """Returns whether this MFList has any data associated with it in key
        "key"."""
        if key is None:
            for sto_key in self._data_storage.keys():
                self.get_data_prep(sto_key)
                if super().has_data():
                    return True
            return False
        else:
            self.get_data_prep(key)
            return super().has_data()

    def has_modified_ext_data(self, key=None):
        if key is None:
            for sto_key in self._data_storage.keys():
                self.get_data_prep(sto_key)
                if super().has_modified_ext_data():
                    return True
            return False
        else:
            self.get_data_prep(key)
            return super().has_modified_ext_data()

    def binary_ext_data(self, key=None):
        if key is None:
            for sto_key in self._data_storage.keys():
                self.get_data_prep(sto_key)
                if super().binary_ext_data():
                    return True
            return False
        else:
            self.get_data_prep(key)
            return super().binary_ext_data()

    def get_record(self, key=None):
        """Returns the data for stress period `key`.  If no key is specified
        returns all records in a dictionary with zero-based stress period
        numbers as keys.  See MFList's get_record documentation for more
        information on the format of each record returned.

        Parameters
        ----------
            key : int
                Zero-based stress period to return data from.

        Returns
        -------
            data_record : dict

        """
        if self._data_storage is not None and len(self._data_storage) > 0:
            if key is None:
                output = {}
                for key in self._data_storage.keys():
                    self.get_data_prep(key)
                    output[key] = super().get_record()
                return output
            self.get_data_prep(key)
            return super().get_record()
        else:
            return None

    def get_data(self, key=None, apply_mult=False, **kwargs):
        """Returns the data for stress period `key`.

        Parameters
        ----------
            key : int
                Zero-based stress period to return data from.
            apply_mult : bool
                Apply multiplier

        Returns
        -------
            data : recarray

        """
        if self._data_storage is not None and len(self._data_storage) > 0:
            if key is None:
                if "array" in kwargs:
                    output = []
                    sim_time = self._data_dimensions.package_dim.model_dim[
                        0
                    ].simulation_time
                    num_sp = sim_time.get_num_stress_periods()
                    data = None
                    for sp in range(0, num_sp):
                        if sp in self._data_storage:
                            self.get_data_prep(sp)
                            data = super().get_data(apply_mult=apply_mult)
                        elif self._block.header_exists(sp):
                            data = None
                        output.append(data)
                    return output
                else:
                    output = {}
                    for key in self._data_storage.keys():
                        self.get_data_prep(key)
                        output[key] = super().get_data(apply_mult=apply_mult)
                    return output
            self.get_data_prep(key)
            return super().get_data(apply_mult=apply_mult)
        else:
            return None

    def set_record(self, data_record, autofill=False, check_data=True):
        """Sets the contents of the data based on the contents of
        'data_record`.

        Parameters
        ----------
        data_record : dict
            Data_record being set.  Data_record must be a dictionary with
            keys as zero-based stress periods and values as dictionaries
            containing the data and metadata.  See MFList's set_record
            documentation for more information on the format of the values.
        autofill : bool
            Automatically correct data
        check_data : bool
            Whether to verify the data
        """
        self._set_data_record(
            data_record,
            autofill=autofill,
            check_data=check_data,
            is_record=True,
        )

    def set_data(self, data, key=None, autofill=False):
        """Sets the contents of the data at time `key` to `data`.

        Parameters
        ----------
        data : dict, recarray, list
            Data being set.  Data can be a dictionary with keys as
            zero-based stress periods and values as the data.  If data is
            a recarray or list of tuples, it will be assigned to the
            stress period specified in `key`.  If any is set to None, that
            stress period of data will be removed.
        key : int
            Zero based stress period to assign data too.  Does not apply
            if `data` is a dictionary.
        autofill : bool
            Automatically correct data.
        """
        self._set_data_record(data, key, autofill)

    def _set_data_record(
        self, data, key=None, autofill=False, check_data=False, is_record=False
    ):
        self._cache_model_grid = True
        if isinstance(data, dict):
            if "filename" not in data and "data" not in data:
                # each item in the dictionary is a list for one stress period
                # the dictionary key is the stress period the list is for
                del_keys = []
                for key, list_item in data.items():
                    if list_item is None:
                        self.remove_transient_key(key)
                        del_keys.append(key)
                        self.empty_keys[key] = False
                    elif isinstance(list_item, list) and len(list_item) == 0:
                        self.empty_keys[key] = True
                    else:
                        self.empty_keys[key] = False
                        if isinstance(list_item, dict):
                            is_record = True
                            if "check" in list_item:
                                check = list_item["check"]
                            else:
                                check = True
                        self._set_data_prep(list_item, key)
                        if is_record:
                            super().set_record(list_item, autofill, check_data)
                        else:
                            super().set_data(
                                list_item, autofill=autofill,
                                check_data=check_data
                            )
                for key in del_keys:
                    del data[key]
            else:
                self.empty_keys[key] = False
                self._set_data_prep(data["data"], key)
                super().set_data(data, autofill)
        else:
            if is_record:
                comment = (
                    "Set record method requires that data_record is a "
                    "dictionary."
                )
                type_, value_, traceback_ = sys.exc_info()
                raise MFDataException(
                    self.structure.get_model(),
                    self.structure.get_package(),
                    self._path,
                    "setting data record",
                    self.structure.name,
                    inspect.stack()[0][3],
                    type_,
                    value_,
                    traceback_,
                    comment,
                    self._simulation_data.debug,
                )
            if key is None:
                # search for a key
                new_key_index = self.structure.first_non_keyword_index()
                if new_key_index is not None and len(data) > new_key_index:
                    key = data[new_key_index]
                else:
                    key = 0
            if isinstance(data, list) and len(data) == 0:
                self.empty_keys[key] = True
            else:
                check = True
                if (
                    isinstance(data, list)
                    and len(data) > 0
                    and data[0] == "no_check"
                ):
                    # not checking data
                    check = False
                    data = data[1:]
                self.empty_keys[key] = False
                if data is None:
                    self.remove_transient_key(key)
                else:
                    self._set_data_prep(data, key)
                    super().set_data(data, autofill, check_data=check)
        self._cache_model_grid = False

    def external_file_name(self, key=0):
        """Returns external file name, or None if this is not external data.

        Parameters
        ----------
            key : int
                Zero based stress period to return data from.
        """
        if key in self.empty_keys and self.empty_keys[key]:
            return None
        else:
            self._get_file_entry_prep(key)
            return super().external_file_name()

    def write_file_entry(
        self,
        fd_data_file,
        key=0,
        ext_file_action=ExtFileAction.copy_relative_paths,
        fd_main=None,
    ):
        """Returns a string containing the data at time `key` formatted for a
        MODFLOW 6 file.

        Parameters
        ----------
            fd_data_file : file
                File to write to
            key : int
                Zero based stress period to return data from.
            ext_file_action : ExtFileAction
                How to handle external paths.

        Returns
        -------
            file entry : str

        """
        if key in self.empty_keys and self.empty_keys[key]:
            return ""
        else:
            self._get_file_entry_prep(key)
            return super().write_file_entry(
                fd_data_file,
                ext_file_action=ext_file_action,
                fd_main=fd_main,
            )

    def get_file_entry(
        self, key=0, ext_file_action=ExtFileAction.copy_relative_paths
    ):
        """Returns a string containing the data at time `key` formatted for a
        MODFLOW 6 file.

        Parameters
        ----------
            key : int
                Zero based stress period to return data from.
            ext_file_action : ExtFileAction
                How to handle external paths.

        Returns
        -------
            file entry : str

        """
        if key in self.empty_keys and self.empty_keys[key]:
            return ""
        else:
            self._get_file_entry_prep(key)
            return super()._write_file_entry(None,
                                             ext_file_action=ext_file_action)

    def load(
        self,
        first_line,
        file_handle,
        block_header,
        pre_data_comments=None,
        external_file_info=None,
    ):
        """Loads data from first_line (the first line of data) and open file
        file_handle which is pointing to the second line of data.  Returns a
        tuple with the first item indicating whether all data was read
        and the second item being the last line of text read from the file.

        Parameters
        ----------
            first_line : str
                A string containing the first line of data in this list.
            file_handle : file descriptor
                A file handle for the data file which points to the second
                line of data for this array
            block_header : MFBlockHeader
                Block header object that contains block header information
                for the block containing this data
            pre_data_comments : MFComment
                Comments immediately prior to the data
            external_file_info : list
                Contains information about storing files externally

        """
        self._load_prep(block_header)
        return super().load(
            first_line,
            file_handle,
            block_header,
            pre_data_comments,
            external_file_info,
        )

    def append_list_as_record(self, record, key=0):
        """Appends the list `data` as a single record in this list's recarray
        at time `key`.  Assumes `data` has the correct dimensions.

        Parameters
        ----------
            record : list
                Data to append
            key : int
                Zero based stress period to append data too.

        """
        self._append_list_as_record_prep(record, key)
        super().append_list_as_record(record)

    def update_record(self, record, key_index, key=0):
        """Updates a record at index `key_index` and time `key` with the
        contents of `record`.  If the index does not exist update_record
        appends the contents of `record` to this list's recarray.

        Parameters
        ----------
            record : list
                Record to append
            key_index : int
                Index to update
            key : int
                Zero based stress period to append data too

        """

        self._update_record_prep(key)
        super().update_record(record, key_index)

    def _new_storage(self):
        return {}

    def _get_storage(self):
        if (
            self._current_key is None
            or self._current_key not in self._data_storage
        ):
            return None
        return self._data_storage[self._current_key]
