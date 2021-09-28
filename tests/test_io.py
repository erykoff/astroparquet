import unittest
import tempfile
import os
import shutil
import numpy.testing as testing
import numpy as np
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

import astroparquet


class IoTestCase(unittest.TestCase):
    def test_writeread_simpletable(self):
        """
        Test writing and reading of simple unadorned table.
        """
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestAstroParquet-')

        a = np.arange(10, dtype=np.int32)
        b = np.arange(10, dtype=np.int64)
        c = np.arange(10, dtype=np.float32)
        d = np.arange(10, dtype=np.float64)

        tbl = Table([a, b, c, d], names=('a', 'b', 'c', 'd'))

        fname = os.path.join(self.test_dir, 'test_simple.parquet')

        astroparquet.write_astroparquet(fname, tbl)

        tbl2 = astroparquet.read_astroparquet(fname)

        for name in tbl.columns:
            testing.assert_almost_equal(tbl2[name], tbl[name])
            self.assertEqual(tbl2[name].dtype, tbl[name].dtype)

    def test_writeread_unittable(self):
        """
        Test writing and reading of a table with units.
        """
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestAstroParquet-')

        a = np.arange(10, dtype=np.int32) * u.m
        b = np.arange(10, dtype=np.int64) * (u.m/u.s)
        c = np.arange(10, dtype=np.float32)
        d = np.arange(10, dtype=np.float64)

        tbl = Table([a, b, c, d], names=('a', 'b', 'c', 'd'))

        fname = os.path.join(self.test_dir, 'test_unit.parquet')

        astroparquet.write_astroparquet(fname, tbl)

        tbl2 = astroparquet.read_astroparquet(fname)

        for name in tbl.columns:
            testing.assert_almost_equal(tbl2[name], tbl[name])
            self.assertEqual(tbl2[name].dtype, tbl[name].dtype)
            self.assertEqual(tbl2[name].unit, tbl[name].unit)

    def test_writeread_descriptable(self):
        """
        Test writing and reading of a table with descriptions.
        """
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestAstroParquet-')

        a = np.arange(10, dtype=np.int32) * u.m
        b = np.arange(10, dtype=np.int64) * (u.m/u.s)
        c = np.arange(10, dtype=np.float32)
        d = np.arange(10, dtype=np.float64)

        tbl = Table([a, b, c, d], names=('a', 'b', 'c', 'd'))
        tbl['a'].description = 'column a is here'
        tbl['b'].description = 'this is column b'
        tbl['c'].description = 'and here is column c'

        fname = os.path.join(self.test_dir, 'test_descr.parquet')

        astroparquet.write_astroparquet(fname, tbl)

        tbl2 = astroparquet.read_astroparquet(fname)

        for name in tbl.columns:
            testing.assert_almost_equal(tbl2[name], tbl[name])
            self.assertEqual(tbl2[name].dtype, tbl[name].dtype)
            self.assertEqual(tbl2[name].description, tbl[name].description)

    def test_writeread_timetable(self):
        """
        Test writing and reading of a table with Time.
        """
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestAstroParquet-')

        a = np.arange(10, dtype=np.int32) * u.m
        b = np.arange(10, dtype=np.int64) * (u.m/u.s)
        c = np.arange(10, dtype=np.float32)
        d = Time(['2000:002']*10)

        tbl = Table([a, b, c, d], names=('a', 'b', 'c', 'd'))

        fname = os.path.join(self.test_dir, 'test_time.parquet')

        astroparquet.write_astroparquet(fname, tbl)

        tbl2 = astroparquet.read_astroparquet(fname)

        for name in tbl.columns:
            if name == 'd':
                self.assertTrue(isinstance(tbl2[name], Time))
                for i in range(len(tbl2[name])):
                    self.assertEqual(tbl2[name][i], tbl[name][i])
            else:
                testing.assert_almost_equal(tbl2[name], tbl[name])
                self.assertEqual(tbl2[name].dtype, tbl[name].dtype)

        # Read columns -- including Time
        tbl3 = astroparquet.read_astroparquet(fname, columns=['a', 'c', 'd'])
        self.assertEqual(tbl3.dtype.names, ('a', 'c', 'd'))
        for name in tbl3.columns:
            if name == 'd':
                self.assertTrue(isinstance(tbl3[name], Time))
                for i in range(len(tbl3[name])):
                    self.assertEqual(tbl3[name][i], tbl[name][i])
            else:
                testing.assert_almost_equal(tbl3[name], tbl[name])
                self.assertEqual(tbl3[name].dtype, tbl[name].dtype)

        # Read columns -- excluding Time
        tbl4 = astroparquet.read_astroparquet(fname, columns=['a', 'b', 'c'])
        self.assertEqual(tbl4.dtype.names, ('a', 'b', 'c'))
        for name in tbl4.columns:
            testing.assert_almost_equal(tbl4[name], tbl[name])
            self.assertEqual(tbl4[name].dtype, tbl[name].dtype)

    def test_writeread_coordtable(self):
        """
        Test writing and reading of a table with coordinates.
        """
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestAstroParquet-')

        a = np.arange(10, dtype=np.int32) * u.m
        b = np.arange(10, dtype=np.int64) * (u.m/u.s)
        c = np.arange(10, dtype=np.float32)
        d = SkyCoord(np.arange(10), np.arange(10), unit='deg')

        tbl = Table([a, b, c, d], names=('a', 'b', 'c', 'd'))

        fname = os.path.join(self.test_dir, 'test_skycoord.parquet')

        astroparquet.write_astroparquet(fname, tbl)

        tbl2 = astroparquet.read_astroparquet(fname)

        for name in tbl.columns:
            if name == 'd':
                self.assertTrue(isinstance(tbl2[name], SkyCoord))
                for i in range(len(tbl2[name])):
                    self.assertEqual(tbl2[name][i], tbl[name][i])
            else:
                testing.assert_almost_equal(tbl2[name], tbl[name])
                self.assertEqual(tbl2[name].dtype, tbl[name].dtype)

        # Read columns -- including SkyCoord
        tbl3 = astroparquet.read_astroparquet(fname, columns=['a', 'c', 'd'])
        self.assertEqual(tbl3.dtype.names, ('a', 'c', 'd'))
        for name in tbl3.columns:
            if name == 'd':
                self.assertTrue(isinstance(tbl3[name], SkyCoord))
                for i in range(len(tbl3[name])):
                    self.assertEqual(tbl3[name][i], tbl[name][i])
            else:
                testing.assert_almost_equal(tbl3[name], tbl[name])
                self.assertEqual(tbl3[name].dtype, tbl[name].dtype)

        # Read columns -- excluding Time
        tbl4 = astroparquet.read_astroparquet(fname, columns=['a', 'b', 'c'])
        self.assertEqual(tbl4.dtype.names, ('a', 'b', 'c'))
        for name in tbl4.columns:
            testing.assert_almost_equal(tbl4[name], tbl[name])
            self.assertEqual(tbl4[name].dtype, tbl[name].dtype)

    def test_writeread_masktable(self):
        """
        Test writing and reading of a table with a masked column.
        """
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestAstroParquet-')

        a = np.arange(10, dtype=np.int32) * u.m
        b = np.arange(10, dtype=np.int64) * (u.m/u.s)
        c = np.arange(10, dtype=np.float32)
        d = np.ma.array(np.arange(10, dtype=np.int32))
        msk = np.zeros(10, dtype=bool)
        msk[0] = True
        d.mask = msk

        tbl = Table([a, b, c, d], names=('a', 'b', 'c', 'd'))

        fname = os.path.join(self.test_dir, 'test_mask.parquet')
        astroparquet.write_astroparquet(fname, tbl)

        tbl2 = astroparquet.read_astroparquet(fname)

        for name in tbl.columns:
            testing.assert_almost_equal(tbl2[name], tbl[name])
            self.assertEqual(tbl2[name].dtype, tbl[name].dtype)

        # Read columns -- including masked array
        tbl3 = astroparquet.read_astroparquet(fname, columns=['a', 'c', 'd'])
        self.assertEqual(tbl3.dtype.names, ('a', 'c', 'd'))
        for name in tbl3.columns:
            testing.assert_almost_equal(tbl3[name], tbl[name])
            self.assertEqual(tbl3[name].dtype, tbl[name].dtype)

        # Read columns -- excluding masked array
        tbl4 = astroparquet.read_astroparquet(fname, columns=['a', 'b', 'c'])
        self.assertEqual(tbl4.dtype.names, ('a', 'b', 'c'))
        for name in tbl4.columns:
            testing.assert_almost_equal(tbl4[name], tbl[name])
            self.assertEqual(tbl4[name].dtype, tbl[name].dtype)

    def test_writeread_stringtable(self):
        """
        Test writing and reading of a table with string columns.
        """
        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestAstroParquet-')

        a = np.arange(10, dtype=np.int32)
        b = np.arange(10, dtype=np.int64)
        c = np.arange(10, dtype=np.int32).astype('U10')
        c[1] = '01'
        d = np.arange(10, dtype=np.float32).astype('U20')

        tbl = Table([a, b, c, d], names=('a', 'b', 'c', 'd'))

        fname = os.path.join(self.test_dir, 'test_string.parquet')

        astroparquet.write_astroparquet(fname, tbl)

        tbl2 = astroparquet.read_astroparquet(fname)

        for name in tbl.columns:
            if name == 'c' or name == 'd':
                testing.assert_array_equal(tbl2[name], tbl[name])
            else:
                testing.assert_almost_equal(tbl2[name], tbl[name])
            self.assertEqual(tbl2[name].dtype, tbl[name].dtype)

        # Read columns -- including string
        tbl3 = astroparquet.read_astroparquet(fname, columns=['a', 'c', 'd'])
        self.assertEqual(tbl3.dtype.names, ('a', 'c', 'd'))
        for name in tbl3.columns:
            if name == 'c' or name == 'd':
                testing.assert_array_equal(tbl2[name], tbl[name])
            else:
                testing.assert_almost_equal(tbl3[name], tbl[name])
            self.assertEqual(tbl3[name].dtype, tbl[name].dtype)

        # Read columns -- excluding string
        tbl4 = astroparquet.read_astroparquet(fname, columns=['a', 'b'])
        self.assertEqual(tbl4.dtype.names, ('a', 'b'))
        for name in tbl4.columns:
            testing.assert_almost_equal(tbl4[name], tbl[name])
            self.assertEqual(tbl4[name].dtype, tbl[name].dtype)

    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)


if __name__ == '__main__':
    unittest.main()
