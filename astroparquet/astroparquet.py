import os
import numpy as np
from astropy.table import Table, QTable, serialize, meta
from astropy.utils.data_info import serialize_context_as

import pyarrow as pa
from pyarrow import parquet, dataset


def write_astroparquet(filename, tbl, clobber=False):
    """
    Write an astropy table in parquet format.

    Parameters
    ----------
    filename : `str`
        Output filename.
    tbl : `astropy.Table`
        Table to output.
    """
    if os.path.exists(filename):
        raise NotImplementedError("No clobbering yet.")

    if not isinstance(tbl, (Table, QTable)):
        raise ValueError("Input tbl is not an astropy table.")

    with serialize_context_as('hdf5'):
        # This sets the way nulls are handled; fix later.
        encode_tbl = serialize.represent_mixins_as_columns(tbl)
    meta_yaml = meta.get_yaml_from_table(encode_tbl)
    meta_yaml_str = '\n'.join(meta_yaml)

    metadata = {}
    for col in encode_tbl.columns:
        # Special-case string types to record the length
        if encode_tbl[col].dtype.type is np.str_:
            metadata[f'table::strlen::{col}'] = str(tbl[col].dtype.itemsize//4)

    metadata['table_meta_yaml'] = meta_yaml_str

    type_list = [(name, pa.from_numpy_dtype(encode_tbl.dtype[name].type))
                 for name in encode_tbl.dtype.names]
    schema = pa.schema(type_list, metadata=metadata)

    with parquet.ParquetWriter(filename, schema) as writer:
        arrays = [pa.array(encode_tbl[col]) for col in encode_tbl.columns]
        pa_tbl = pa.Table.from_arrays(arrays, schema=schema)

        writer.write_table(pa_tbl)


def read_astroparquet(filename, columns=None, filter=None, schema_only=False):
    """
    Read an astropy table in parquet format.

    Parameters
    ----------
    filename : `str`
        Input filename
    columns : `list` [`str`], optional
        Name of astropy columns to read.
        This will automatically expand to all sub-columns if
        necessary.
    filter : `pyarrow.Expression`, optional
        Pyarrow filter expression to filter rows.
    schema_only : `bool`, optional
        Only read the schema/metadata with table information.

    Returns
    -------
    tbl : `astropy.Table`
        Table will have zero rows and only metadata information
        if schema_only is True.
    """
    ds = dataset.dataset(filename, format='parquet', partitioning='hive')

    schema = ds.schema

    # Convert from bytes to strings
    md = {key.decode(): schema.metadata[key].decode()
          for key in schema.metadata}

    meta_dict = {}
    if 'table_meta_yaml' in md:
        meta_yaml = md.pop('table_meta_yaml').split('\n')
        meta_hdr = meta.get_header_from_yaml(meta_yaml)
        if 'meta' in meta_hdr:
            meta_dict = meta_hdr['meta']
    else:
        meta_hdr = None

    full_table_columns = {name: name for name in schema.names}
    has_serialized_columns = False
    if '__serialized_columns__' in meta_dict:
        has_serialized_columns = True
        serialized_columns = meta_dict['__serialized_columns__']
        for scol in serialized_columns:
            for name in _get_names(serialized_columns[scol]):
                full_table_columns[name] = scol

    if columns is not None:
        columns_to_read = []
        for column in columns:
            cols = [full_table_column
                    for full_table_column in full_table_columns
                    if column == full_table_columns[full_table_column]]
            columns_to_read.extend(cols)

        if columns_to_read == []:
            # Should this raise instead?
            return Table()

        # We need to pop any unread serialized columns out of the meta_dict.
        if has_serialized_columns:
            for scol in list(meta_dict['__serialized_columns__'].keys()):
                if scol not in columns:
                    meta_dict['__serialized_columns__'].pop(scol)
    else:
        columns_to_read = schema.names

    if not schema_only:
        pa_tbl = ds.to_table(columns=columns_to_read, filter=filter)
        num_rows = pa_tbl.num_rows
    else:
        num_rows = 0

    dtype = []
    for name in columns_to_read:
        if schema.field(name).type == pa.string():
            md_name = f'table::strlen::{name}'
            if md_name in md:
                # String length from header.
                strlen = int(md[f'table::strlen::{name}'])
            else:
                # Find the maximum string length.
                if schema_only:
                    # Choose an arbitrary string length since we
                    # are not reading the table.
                    strlen = 10
                else:
                    strlen = max([len(row.as_py()) for row in pa_tbl['c']])
            dtype.append(f'U{strlen}')
        else:
            dtype.append(schema.field(name).type.to_pandas_dtype())

    data = np.zeros(num_rows, dtype=list(zip(columns_to_read, dtype)))

    if not schema_only:
        for name in columns_to_read:
            data[name][:] = pa_tbl[name].to_numpy()

    tbl = Table(data=data)
    if meta_dict is not None:
        tbl.meta = meta_dict

    if meta_hdr is not None:
        header_cols = dict((x['name'], x) for x in meta_hdr['datatype'])
        for col in tbl.columns.values():
            for attr in ('description', 'format', 'unit', 'meta'):
                if attr in header_cols[col.name]:
                    setattr(col, attr, header_cols[col.name][attr])

    tbl = serialize._construct_mixins_from_columns(tbl)

    return tbl


def _get_names(_dict):
    """Recursively find the names in a serialized column dictionary.

    Parameters
    ----------
    _dict : `dict`
        Dictionary from astropy __serialized_columns__

    Returns
    -------
    all_names : `list` [`str`]
        All the column names mentioned in _dict and sub-dicts.
    """
    all_names = []
    for key in _dict:
        if isinstance(_dict[key], dict):
            all_names.extend(_get_names(_dict[key]))
        else:
            if key == 'name':
                all_names.append(_dict['name'])
    return all_names
