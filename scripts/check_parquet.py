import sys
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq

for arg in sys.argv[1:]:
    path = Path(arg)
    try:
        if path.is_dir():
            table = ds.dataset(str(path), format="parquet").to_table()
            n_files = len(list(path.glob("*.parquet")))
            print(f"{path}: OK, {n_files} part-files, {table.num_rows} total rows, {table.num_columns} cols")
        else:
            table = pq.read_table(str(path))
            print(f"{path}: OK, {table.num_rows} rows, {table.num_columns} cols")
        print(table.slice(0, 3).to_pydict())
    except Exception as e:
        print(f"{path}: FAILED -- {e}")
