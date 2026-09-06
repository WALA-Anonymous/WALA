from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class LeRobotV3DataShard:
    path: Path
    row_count: int
    global_from_index: int
    global_to_index: int


class LeRobotV3DataShardIndex:
    """Locate LeRobot v3 parquet rows without trusting episode file indices."""

    def __init__(self, dataset_root: Path, expected_total_rows: int | None = None):
        self.dataset_root = Path(dataset_root)
        paths = tuple(sorted((self.dataset_root / "data").glob("chunk-*/*.parquet")))
        if not paths:
            raise ValueError(f"No LeRobot v3 parquet shards found under {self.dataset_root / 'data'}")

        shards = []
        cursor = 0
        for path in paths:
            row_count = int(pq.ParquetFile(path).metadata.num_rows)
            if row_count <= 0:
                raise ValueError(f"LeRobot v3 parquet shard is empty: {path}")
            shards.append(
                LeRobotV3DataShard(
                    path=path,
                    row_count=row_count,
                    global_from_index=cursor,
                    global_to_index=cursor + row_count,
                )
            )
            cursor += row_count

        if expected_total_rows is not None and cursor != int(expected_total_rows):
            raise ValueError(
                f"{self.dataset_root}: parquet rows {cursor} do not match "
                f"info.json total_frames {expected_total_rows}"
            )

        self.shards = tuple(shards)
        self.total_rows = cursor
        self._shard_ends = tuple(shard.global_to_index for shard in self.shards)

    def _locate_global_row(self, row_index: int) -> tuple[int, int]:
        if row_index < 0 or row_index >= self.total_rows:
            raise IndexError(f"Global row {row_index} is outside [0, {self.total_rows})")
        shard_index = bisect.bisect_right(self._shard_ends, row_index)
        shard = self.shards[shard_index]
        return shard_index, row_index - shard.global_from_index

    @staticmethod
    def _read_local_range(path: Path, start: int, stop: int) -> pa.Table:
        if start < 0 or stop <= start:
            raise ValueError(f"Invalid parquet row range [{start}, {stop}) for {path}")

        parquet = pq.ParquetFile(path)
        tables = []
        group_start = 0
        for group_index in range(parquet.num_row_groups):
            group_rows = int(parquet.metadata.row_group(group_index).num_rows)
            group_stop = group_start + group_rows
            overlap_start = max(start, group_start)
            overlap_stop = min(stop, group_stop)
            if overlap_start < overlap_stop:
                table = parquet.read_row_group(group_index)
                tables.append(
                    table.slice(overlap_start - group_start, overlap_stop - overlap_start)
                )
            if group_stop >= stop:
                break
            group_start = group_stop

        if not tables:
            raise RuntimeError(f"No rows read from {path} for local range [{start}, {stop})")
        return tables[0] if len(tables) == 1 else pa.concat_tables(tables)

    def read_episode(
        self,
        *,
        trajectory_id: int,
        dataset_from_index: int,
        length: int,
    ) -> pd.DataFrame:
        start = int(dataset_from_index)
        length = int(length)
        stop = start + length
        if length <= 0:
            raise ValueError(f"Episode {trajectory_id} has non-positive length {length}")
        if start < 0 or stop > self.total_rows:
            raise IndexError(
                f"Episode {trajectory_id} global range [{start}, {stop}) is outside "
                f"[0, {self.total_rows})"
            )

        tables = []
        current = start
        while current < stop:
            shard_index, local_start = self._locate_global_row(current)
            shard = self.shards[shard_index]
            rows_to_read = min(stop - current, shard.global_to_index - current)
            tables.append(
                self._read_local_range(
                    shard.path,
                    local_start,
                    local_start + rows_to_read,
                )
            )
            current += rows_to_read

        table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
        episode_data = table.to_pandas()
        if len(episode_data) != length:
            raise ValueError(
                f"Episode {trajectory_id} expected {length} rows but global index returned "
                f"{len(episode_data)}"
            )
        if "episode_index" not in episode_data.columns:
            raise ValueError("LeRobot v3 parquet data is missing the episode_index column")

        episode_ids = np.asarray(episode_data["episode_index"])
        if not np.all(episode_ids == int(trajectory_id)):
            observed = np.unique(episode_ids).tolist()
            raise ValueError(
                f"Episode {trajectory_id} global range contains episode IDs {observed}"
            )
        return episode_data
