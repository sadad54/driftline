import sys

import pyarrow.parquet as pq

for path in sys.argv[1:]:
    try:
        table = pq.read_table(path)
        print(f"{path}: OK, {table.num_rows} rows, {table.num_columns} cols")
        print(table.slice(0, 3).to_pydict())
    except Exception as e:
        print(f"{path}: FAILED -- {e}")
