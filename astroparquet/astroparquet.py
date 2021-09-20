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

    with serialize_context_as('fits'):
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
        arrays = [pa.array(encode_tbl[name].data)
                  for name in encode_tbl.dtype.names]
        pa_tbl = pa.Table.from_arrays(arrays, schema=schema)

        writer.write_table(pa_tbl)


def read_astroparquet(filename, columns=None, filter=None):
    """
    Read an astropy table in parquet format.

    Parameters
    ----------
    filename : `str`
        Input filename
    columns : `list` [`str`], optional
        Name of columns to read.
    filter : `expression thing`, option
        Pyarrow filter expression to filter rows.

    Returns
    -------
    tbl : `astropy.Table`
    """
    ds = dataset.dataset(filename, format='parquet', partitioning='hive')

    schema = ds.schema

    # Convert from bytes to strings
    md = {key.decode(): schema.metadata[key].decode()
          for key in schema.metadata}

    meta_dict = None
    if 'table_meta_yaml' in md:
        meta_yaml = md.pop('table_meta_yaml').split('\n')
        meta_hdr = meta.get_header_from_yaml(meta_yaml)
        if 'meta' in meta_hdr:
            meta_dict = meta_hdr['meta']
    else:
        meta_dict = None
        meta_hdr = None

    names = schema.names

    if columns is not None:
        names = [name for name in schema.names
                 if name in columns]

        if names == []:
            # Should this raise instead?
            return Table()
    else:
        names = schema.names

    dtype = []
    for name in names:
        if schema.field(name).type == pa.string():
            dtype.append('U%d' % (int(md[f'table::strlen::{name}'])))
        else:
            dtype.append(schema.field(name).type.to_pandas_dtype())

    pa_tbl = ds.to_table(columns=names, filter=None)
    data = np.zeros(pa_tbl.num_rows, dtype=list(zip(names, dtype)))

    for name in names:
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


def read_astroparquet_schema(filename):
    """
    Read an astroparquet schema (to get columns, units, etc)

    Parameters
    ----------
    filename : `str`
        Input filename

    Returns
    -------
    something
    """
    pass
