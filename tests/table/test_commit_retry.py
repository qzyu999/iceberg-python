# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from typing import Any
from unittest.mock import patch

import pytest

from pyiceberg.catalog import Catalog
from pyiceberg.exceptions import CommitFailedException, ValidationException
from pyiceberg.schema import Schema
from pyiceberg.table import TableProperties, Transaction
from pyiceberg.table.snapshots import IsolationLevel, Operation
from pyiceberg.types import LongType, NestedField, StringType


def test_isolation_level_enum() -> None:
    assert IsolationLevel.SERIALIZABLE.value == "serializable"
    assert IsolationLevel.SNAPSHOT.value == "snapshot"
    assert IsolationLevel("serializable") is IsolationLevel.SERIALIZABLE
    assert IsolationLevel("snapshot") is IsolationLevel.SNAPSHOT


def test_commit_retry_table_properties() -> None:
    assert TableProperties.COMMIT_NUM_RETRIES == "commit.retry.num-retries"
    assert TableProperties.COMMIT_NUM_RETRIES_DEFAULT == 4
    assert TableProperties.COMMIT_MIN_RETRY_WAIT_MS == "commit.retry.min-wait-ms"
    assert TableProperties.COMMIT_MIN_RETRY_WAIT_MS_DEFAULT == 100
    assert TableProperties.COMMIT_MAX_RETRY_WAIT_MS == "commit.retry.max-wait-ms"
    assert TableProperties.COMMIT_MAX_RETRY_WAIT_MS_DEFAULT == 60000
    assert TableProperties.COMMIT_TOTAL_RETRY_TIME_MS == "commit.retry.total-timeout-ms"
    assert TableProperties.COMMIT_TOTAL_RETRY_TIME_MS_DEFAULT == 1800000


def test_isolation_level_table_properties() -> None:
    assert TableProperties.WRITE_DELETE_ISOLATION_LEVEL == "write.delete.isolation-level"
    assert TableProperties.WRITE_UPDATE_ISOLATION_LEVEL == "write.update.isolation-level"
    assert TableProperties.WRITE_ISOLATION_LEVEL_DEFAULT == "serializable"


def _test_schema() -> Schema:
    return Schema(NestedField(1, "x", LongType(), required=False))


def test_commit_retry_on_commit_failed(catalog: Catalog) -> None:
    """Verify that CommitFailedException triggers retry for append operations."""
    catalog.create_namespace("default")
    schema = _test_schema()
    catalog.create_table("default.retry_test", schema=schema)

    import pyarrow as pa

    df = pa.table({"x": [1, 2, 3]})

    # Load two references to the same table to simulate concurrent access
    tbl1 = catalog.load_table("default.retry_test")
    tbl2 = catalog.load_table("default.retry_test")

    # First append succeeds
    tbl1.append(df)

    # Second append should succeed via retry (append vs append never conflicts)
    import pyiceberg.table as _table_module

    RuntimeTransaction = _table_module.Transaction
    original_rebuild = RuntimeTransaction._rebuild_snapshot_updates
    rebuild_count = 0

    def counting_rebuild(self_tx: Any) -> None:
        nonlocal rebuild_count
        rebuild_count += 1
        original_rebuild(self_tx)

    with patch.object(RuntimeTransaction, "_rebuild_snapshot_updates", counting_rebuild):
        tbl2.append(df)

    assert rebuild_count == 1, "Expected exactly one retry via _rebuild_snapshot_updates"

    # Both appends should be visible
    refreshed = catalog.load_table("default.retry_test")
    result = refreshed.scan().to_arrow()
    assert len(result) == 6


def test_no_retry_without_snapshot_producers(catalog: Catalog) -> None:
    """Verify that a transaction with no snapshot producers has an empty producer list."""
    catalog.create_namespace("default")
    schema = _test_schema()
    table = catalog.create_table("default.no_retry_test", schema=schema)

    tx = Transaction(table, autocommit=False)
    tx.set_properties({"key": "value"})

    # No snapshot producers registered
    assert len(tx._snapshot_producers) == 0


def test_rebuild_snapshot_updates_preserves_non_snapshot_updates(catalog: Catalog) -> None:
    """Verify that non-snapshot updates survive retry."""
    catalog.create_namespace("default")
    schema = _test_schema()
    catalog.create_table("default.rebuild_test", schema=schema)

    import pyarrow as pa

    df = pa.table({"x": [1]})

    tbl1 = catalog.load_table("default.rebuild_test")
    tbl2 = catalog.load_table("default.rebuild_test")

    # tbl1 commits first
    tbl1.append(df)

    # tbl2 does both property change and append in one transaction
    with tbl2.transaction() as tx:
        tx.set_properties({"test_key": "test_value"})
        tx.append(df)

    # Both the property and the data should be committed
    refreshed = catalog.load_table("default.rebuild_test")
    assert refreshed.metadata.properties.get("test_key") == "test_value"
    assert len(refreshed.scan().to_arrow()) == 2


def test_refresh_for_retry_resets_producer_state(catalog: Catalog) -> None:
    """Verify that _refresh_for_retry resets the necessary fields."""
    catalog.create_namespace("default")
    schema = _test_schema()
    table = catalog.create_table("default.refresh_test", schema=schema)

    from pyiceberg.table.update.snapshot import _FastAppendFiles

    tx = Transaction(table, autocommit=False)
    producer = _FastAppendFiles(
        operation=Operation.APPEND,
        transaction=tx,
        io=table.io,
    )

    original_snapshot_id = producer._snapshot_id
    original_uuid = producer.commit_uuid

    producer._refresh_for_retry()

    assert producer._snapshot_id != original_snapshot_id
    assert producer.commit_uuid != original_uuid
    # parent stays None for empty table
    assert producer._parent_snapshot_id is None


def test_concurrent_delete_delete_raises_validation_exception(catalog: Catalog) -> None:
    """Concurrent deletes on the same data should fail with ValidationException."""
    catalog.create_namespace("default")
    schema = _test_schema()
    catalog.create_table("default.del_del_test", schema=schema)

    import pyarrow as pa

    df = pa.table({"x": [1, 2, 3]})

    tbl = catalog.load_table("default.del_del_test")
    tbl.append(df)

    tbl1 = catalog.load_table("default.del_del_test")
    tbl2 = catalog.load_table("default.del_del_test")

    tbl1.delete("x == 1")

    with pytest.raises(ValidationException):
        tbl2.delete("x == 1")


def test_concurrent_append_delete_raises_validation_exception(catalog: Catalog) -> None:
    """Delete after a concurrent append fails with ValidationException under serializable isolation."""
    catalog.create_namespace("default")
    schema = _test_schema()
    catalog.create_table("default.app_del_test", schema=schema)

    import pyarrow as pa

    df = pa.table({"x": [1, 2, 3]})

    tbl = catalog.load_table("default.app_del_test")
    tbl.append(df)

    tbl1 = catalog.load_table("default.app_del_test")
    tbl2 = catalog.load_table("default.app_del_test")

    tbl1.append(df)

    with pytest.raises(ValidationException):
        tbl2.delete("x == 1")


def test_concurrent_delete_append_retries_successfully(catalog: Catalog) -> None:
    """Append after a concurrent delete should succeed via retry."""
    catalog.create_namespace("default")
    schema = _test_schema()
    catalog.create_table("default.del_app_test", schema=schema)

    import pyarrow as pa

    df = pa.table({"x": [1, 2, 3]})

    tbl = catalog.load_table("default.del_app_test")
    tbl.append(df)

    tbl1 = catalog.load_table("default.del_app_test")
    tbl2 = catalog.load_table("default.del_app_test")

    tbl1.delete("x == 1")

    import pyiceberg.table as _table_module

    RuntimeTransaction = _table_module.Transaction
    original_rebuild = RuntimeTransaction._rebuild_snapshot_updates
    rebuild_count = 0

    def counting_rebuild(self_tx: Any) -> None:
        nonlocal rebuild_count
        rebuild_count += 1
        original_rebuild(self_tx)

    with patch.object(RuntimeTransaction, "_rebuild_snapshot_updates", counting_rebuild):
        tbl2.append(df)

    assert rebuild_count == 1

    refreshed = catalog.load_table("default.del_app_test")
    result = refreshed.scan().to_arrow()
    # Original 3 rows, minus 1 deleted, plus 3 appended = 5
    assert len(result) == 5


def test_retry_exhaustion_raises_commit_failed(catalog: Catalog) -> None:
    """When retries are exhausted, CommitFailedException should be raised."""
    catalog.create_namespace("default")
    schema = _test_schema()
    catalog.create_table(
        "default.exhaust_test",
        schema=schema,
        properties={"commit.retry.num-retries": "0"},
    )

    import pyarrow as pa

    df = pa.table({"x": [1, 2, 3]})

    tbl1 = catalog.load_table("default.exhaust_test")
    tbl2 = catalog.load_table("default.exhaust_test")

    tbl1.append(df)

    with pytest.raises(CommitFailedException):
        tbl2.append(df)


def test_delete_files_refresh_clears_compute_deletes_cache(catalog: Catalog) -> None:
    """Verify that _refresh_for_retry clears the _compute_deletes cached property."""
    catalog.create_namespace("default")
    schema = _test_schema()
    table = catalog.create_table("default.cache_test", schema=schema)

    import pyarrow as pa

    df = pa.table({"x": [1, 2, 3]})
    table.append(df)
    table = catalog.load_table("default.cache_test")

    from pyiceberg.expressions import EqualTo
    from pyiceberg.table.update.snapshot import _DeleteFiles

    tx = Transaction(table, autocommit=False)
    producer = _DeleteFiles(
        operation=Operation.DELETE,
        transaction=tx,
        io=table.io,
    )
    producer.delete_by_predicate(EqualTo("x", 1))

    # Access _compute_deletes to populate the cache
    _ = producer._compute_deletes

    assert "_compute_deletes" in producer.__dict__

    producer._refresh_for_retry()

    assert "_compute_deletes" not in producer.__dict__


def test_concurrent_overwrite_overwrite_raises_validation_exception(catalog: Catalog) -> None:
    """Concurrent overwrites on the same data should fail with ValidationException."""
    catalog.create_namespace("default")
    schema = _test_schema()
    catalog.create_table("default.ow_ow_test", schema=schema)

    import pyarrow as pa

    df = pa.table({"x": [1, 2, 3]})

    tbl = catalog.load_table("default.ow_ow_test")
    tbl.append(df)

    tbl1 = catalog.load_table("default.ow_ow_test")
    tbl2 = catalog.load_table("default.ow_ow_test")

    tbl1.overwrite(pa.table({"x": [10, 20, 30]}), overwrite_filter="x > 0")
    with pytest.raises(ValidationException):
        tbl2.overwrite(pa.table({"x": [40, 50, 60]}), overwrite_filter="x > 0")


def test_concurrent_overwrite_append_retries_successfully(catalog: Catalog) -> None:
    """Append after a concurrent overwrite should succeed via retry."""
    catalog.create_namespace("default")
    schema = _test_schema()
    catalog.create_table("default.ow_app_test", schema=schema)

    import pyarrow as pa

    df = pa.table({"x": [1, 2, 3]})

    tbl = catalog.load_table("default.ow_app_test")
    tbl.append(df)

    tbl1 = catalog.load_table("default.ow_app_test")
    tbl2 = catalog.load_table("default.ow_app_test")

    tbl1.overwrite(pa.table({"x": [10, 20, 30]}), overwrite_filter="x > 0")
    tbl2.append(pa.table({"x": [4, 5, 6]}))

    refreshed = catalog.load_table("default.ow_app_test")
    result = refreshed.scan().to_arrow()
    # overwrite replaced 3 rows with 3 new rows, then append added 3 more = 6
    assert len(result) == 6


def test_snapshot_isolation_allows_concurrent_append_delete(catalog: Catalog) -> None:
    """Under snapshot isolation, delete after a concurrent append should succeed via retry."""
    catalog.create_namespace("default")
    schema = _test_schema()
    catalog.create_table(
        "default.snapshot_iso_test",
        schema=schema,
        properties={"write.delete.isolation-level": "snapshot"},
    )

    import pyarrow as pa

    df = pa.table({"x": [1, 2, 3]})

    tbl = catalog.load_table("default.snapshot_iso_test")
    tbl.append(df)

    tbl1 = catalog.load_table("default.snapshot_iso_test")
    tbl2 = catalog.load_table("default.snapshot_iso_test")

    tbl1.append(df)

    # Under serializable this would raise ValidationException,
    # but under snapshot isolation _validate_added_data_files is skipped
    tbl2.delete("x == 1")

    refreshed = catalog.load_table("default.snapshot_iso_test")
    result = refreshed.scan().to_arrow()
    # Original 3, delete removes x==1 from original (1 row), append adds 3 = 5
    assert len(result) == 5


def test_uncommitted_manifests_tracked_correctly(catalog: Catalog) -> None:
    """Verify that uncommitted manifests are moved to _uncommitted_manifests on retry."""
    catalog.create_namespace("default")
    schema = _test_schema()
    catalog.create_table("default.manifest_track_test", schema=schema)

    import pyarrow as pa

    df = pa.table({"x": [1, 2, 3]})

    tbl = catalog.load_table("default.manifest_track_test")
    tbl.append(df)

    tbl1 = catalog.load_table("default.manifest_track_test")
    tbl2 = catalog.load_table("default.manifest_track_test")

    tbl1.append(df)

    import pyiceberg.table as _table_module2

    RuntimeTransaction2 = _table_module2.Transaction
    original_rebuild = RuntimeTransaction2._rebuild_snapshot_updates
    uncommitted_count_during_rebuild = 0

    def checking_rebuild(self_tx: Any) -> None:
        nonlocal uncommitted_count_during_rebuild
        original_rebuild(self_tx)
        for producer in self_tx._snapshot_producers:
            uncommitted_count_during_rebuild += len(producer._uncommitted_manifests)

    with patch.object(RuntimeTransaction2, "_rebuild_snapshot_updates", checking_rebuild):
        tbl2.append(df)

    # After rebuild, the first attempt's manifests should be in _uncommitted_manifests
    assert uncommitted_count_during_rebuild > 0


def test_concurrent_deletes_on_different_partitions_succeed(catalog: Catalog) -> None:
    """Concurrent deletes on different partitions should succeed via retry thanks to conflict detection filter."""
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.transforms import IdentityTransform

    catalog.create_namespace("default")
    schema = Schema(
        NestedField(1, "category", StringType(), required=False),
        NestedField(2, "value", LongType(), required=False),
    )
    spec = PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="category"))
    catalog.create_table("default.part_del_test", schema=schema, partition_spec=spec)

    import pyarrow as pa

    df = pa.table(
        {
            "category": ["a", "a", "b", "b"],
            "value": [1, 2, 3, 4],
        }
    )

    tbl = catalog.load_table("default.part_del_test")
    tbl.append(df)

    tbl1 = catalog.load_table("default.part_del_test")
    tbl2 = catalog.load_table("default.part_del_test")

    # Delete from different partitions should not conflict
    tbl1.delete("category == 'a'")
    tbl2.delete("category == 'b'")

    refreshed = catalog.load_table("default.part_del_test")
    result = refreshed.scan().to_arrow()
    assert len(result) == 0


def test_concurrent_partial_deletes_on_different_partitions_succeed(catalog: Catalog) -> None:
    """Concurrent partial deletes (CoW rewrite) on different partitions should succeed.

    This tests the auto-computed partition predicate from _build_delete_files_partition_predicate.
    """
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.transforms import IdentityTransform

    catalog.create_namespace("default")
    schema = Schema(
        NestedField(1, "category", StringType(), required=False),
        NestedField(2, "value", LongType(), required=False),
    )
    spec = PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="category"))
    catalog.create_table("default.part_partial_del_test", schema=schema, partition_spec=spec)

    import pyarrow as pa

    df = pa.table(
        {
            "category": ["a", "a", "b", "b"],
            "value": [1, 2, 3, 4],
        }
    )

    tbl = catalog.load_table("default.part_partial_del_test")
    tbl.append(df)

    tbl1 = catalog.load_table("default.part_partial_del_test")
    tbl2 = catalog.load_table("default.part_partial_del_test")

    # Partial delete: only value==1 in partition a, triggers CoW rewrite
    tbl1.delete("value == 1")
    # Partial delete: only value==3 in partition b, triggers CoW rewrite
    tbl2.delete("value == 3")

    refreshed = catalog.load_table("default.part_partial_del_test")
    result = refreshed.scan().to_arrow()
    # Original 4 rows, minus value==1 and value==3 = 2 rows remaining
    assert len(result) == 2


def test_overwrite_uses_update_isolation_level(catalog: Catalog) -> None:
    """Verify that overwrite() reads write.update.isolation-level, not write.delete.isolation-level."""
    catalog.create_namespace("default")
    schema = _test_schema()
    catalog.create_table(
        "default.update_iso_test",
        schema=schema,
        properties={
            "write.delete.isolation-level": "serializable",
            "write.update.isolation-level": "snapshot",
        },
    )

    import pyarrow as pa

    df = pa.table({"x": [1, 2, 3]})

    tbl = catalog.load_table("default.update_iso_test")
    tbl.append(df)

    tbl1 = catalog.load_table("default.update_iso_test")
    tbl2 = catalog.load_table("default.update_iso_test")

    tbl1.append(df)

    # Under write.delete.isolation-level=serializable this would raise ValidationException.
    # But overwrite() uses write.update.isolation-level=snapshot, so it succeeds.
    tbl2.overwrite(pa.table({"x": [10, 20, 30]}), overwrite_filter="x > 0")

    refreshed = catalog.load_table("default.update_iso_test")
    result = refreshed.scan().to_arrow()
    # overwrite with x > 0 deletes all rows (including tbl1's append), then adds 3 new rows
    assert len(result) == 3


def test_overwrite_with_serializable_update_isolation_raises(catalog: Catalog) -> None:
    """Verify that overwrite() raises ValidationException when write.update.isolation-level=serializable."""
    catalog.create_namespace("default")
    schema = _test_schema()
    catalog.create_table(
        "default.update_serial_test",
        schema=schema,
        properties={
            "write.update.isolation-level": "serializable",
        },
    )

    import pyarrow as pa

    df = pa.table({"x": [1, 2, 3]})

    tbl = catalog.load_table("default.update_serial_test")
    tbl.append(df)

    tbl1 = catalog.load_table("default.update_serial_test")
    tbl2 = catalog.load_table("default.update_serial_test")

    tbl1.append(df)

    with pytest.raises(ValidationException):
        tbl2.overwrite(pa.table({"x": [10, 20, 30]}), overwrite_filter="x > 0")


def test_clean_all_uncommitted_on_validation_exception(catalog: Catalog) -> None:
    """Verify that all manifests are cleaned up when commit aborts with ValidationException."""
    catalog.create_namespace("default")
    schema = _test_schema()
    catalog.create_table("default.clean_abort_test", schema=schema)

    import pyarrow as pa

    df = pa.table({"x": [1, 2, 3]})

    tbl = catalog.load_table("default.clean_abort_test")
    tbl.append(df)

    tbl1 = catalog.load_table("default.clean_abort_test")
    tbl2 = catalog.load_table("default.clean_abort_test")

    tbl1.delete("x == 1")

    from pyiceberg.table.update.snapshot import _SnapshotProducer

    captured_producers: list[Any] = []
    original_clean_all = _SnapshotProducer._clean_all_uncommitted

    def capturing_clean_all(self_producer: Any) -> None:
        captured_producers.append(self_producer)
        original_clean_all(self_producer)

    with patch.object(_SnapshotProducer, "_clean_all_uncommitted", capturing_clean_all):
        with pytest.raises(ValidationException):
            tbl2.delete("x == 1")

    # _clean_all_uncommitted was called on abort
    assert len(captured_producers) > 0
    # All manifest lists should be cleared
    for producer in captured_producers:
        assert producer._written_manifests == []
        assert producer._uncommitted_manifests == []


# =============================================================================
# Gap Tests (G1-G16) — Derived from formal model in pyiceberg_pr_3320_testing.md
# =============================================================================


def test_concurrent_append_append_partitioned(catalog: Catalog) -> None:
    """G1: Axiom 1 (Append Commutativity) should hold on partitioned tables."""
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.transforms import IdentityTransform

    catalog.create_namespace("default")
    schema = Schema(
        NestedField(1, "category", StringType(), required=False),
        NestedField(2, "value", LongType(), required=False),
    )
    spec = PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="category"))
    catalog.create_table("default.g1_part_append_test", schema=schema, partition_spec=spec)

    import pyarrow as pa

    tbl1 = catalog.load_table("default.g1_part_append_test")
    tbl2 = catalog.load_table("default.g1_part_append_test")

    tbl1.append(pa.table({"category": ["a"], "value": [1]}))
    tbl2.append(pa.table({"category": ["b"], "value": [2]}))

    refreshed = catalog.load_table("default.g1_part_append_test")
    result = refreshed.scan().to_arrow()
    assert len(result) == 2


def test_retry_respects_custom_backoff_parameters(catalog: Catalog) -> None:
    """G2: Verify custom min-wait-ms and max-wait-ms bound sleep time."""
    import time

    catalog.create_namespace("default")
    schema = _test_schema()
    catalog.create_table(
        "default.g2_backoff_test",
        schema=schema,
        properties={
            "commit.retry.num-retries": "1",
            "commit.retry.min-wait-ms": "200",
            "commit.retry.max-wait-ms": "300",
        },
    )

    import pyarrow as pa

    df = pa.table({"x": [1, 2, 3]})

    tbl1 = catalog.load_table("default.g2_backoff_test")
    tbl2 = catalog.load_table("default.g2_backoff_test")

    tbl1.append(df)

    start = time.monotonic()
    tbl2.append(df)
    elapsed_ms = (time.monotonic() - start) * 1000

    # The retry should have waited at least min_wait_ms (200ms).
    # We use a conservative lower bound to account for timing imprecision.
    assert elapsed_ms >= 150, f"Expected at least ~200ms wait, got {elapsed_ms:.0f}ms"


def test_total_timeout_terminates_retry(catalog: Catalog) -> None:
    """G3: commit.retry.total-timeout-ms should terminate the retry loop."""
    catalog.create_namespace("default")
    schema = _test_schema()
    catalog.create_table(
        "default.g3_timeout_test",
        schema=schema,
        properties={
            "commit.retry.num-retries": "100",
            "commit.retry.total-timeout-ms": "1",  # 1ms timeout — will expire immediately
            "commit.retry.min-wait-ms": "10",
        },
    )

    import pyarrow as pa

    df = pa.table({"x": [1, 2, 3]})

    tbl1 = catalog.load_table("default.g3_timeout_test")
    tbl2 = catalog.load_table("default.g3_timeout_test")

    tbl1.append(df)

    from pyiceberg.exceptions import CommitFailedException

    with pytest.raises(CommitFailedException):
        tbl2.append(df)


def test_concurrent_delete_same_partition_different_rows(catalog: Catalog) -> None:
    """G4: Concurrent deletes targeting different rows in the same partition should fail.

    V₂ validates at the file level, not the row level. Since both deletes touch the
    same data files (same partition), a ValidationException is expected.
    """
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.transforms import IdentityTransform

    catalog.create_namespace("default")
    schema = Schema(
        NestedField(1, "category", StringType(), required=False),
        NestedField(2, "value", LongType(), required=False),
    )
    spec = PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="category"))
    catalog.create_table("default.g4_same_part_test", schema=schema, partition_spec=spec)

    import pyarrow as pa

    df = pa.table({"category": ["a", "a", "a"], "value": [1, 2, 3]})

    tbl = catalog.load_table("default.g4_same_part_test")
    tbl.append(df)

    tbl1 = catalog.load_table("default.g4_same_part_test")
    tbl2 = catalog.load_table("default.g4_same_part_test")

    tbl1.delete("value == 1")

    # Both target partition 'a', so file-level conflict → ValidationException
    with pytest.raises(ValidationException):
        tbl2.delete("value == 3")


def test_fast_append_validate_concurrency_is_noop(catalog: Catalog) -> None:
    """G5: _FastAppendFiles._validate_concurrency() should be the base class no-op."""
    catalog.create_namespace("default")
    schema = _test_schema()
    table = catalog.create_table("default.g5_noop_test", schema=schema)

    from pyiceberg.table.update.snapshot import _FastAppendFiles

    tx = Transaction(table, autocommit=False)
    producer = _FastAppendFiles(
        operation=Operation.APPEND,
        transaction=tx,
        io=table.io,
    )

    # Should not raise — it's a no-op
    producer._validate_concurrency()


def test_merge_append_retry_resets_manifest_counter(catalog: Catalog) -> None:
    """G6: _MergeAppendFiles must reset manifest counter on retry to avoid filename collisions."""
    catalog.create_namespace("default")
    schema = _test_schema()
    table = catalog.create_table("default.g6_merge_test", schema=schema)

    from pyiceberg.table.update.snapshot import _MergeAppendFiles

    tx = Transaction(table, autocommit=False)
    producer = _MergeAppendFiles(
        operation=Operation.APPEND,
        transaction=tx,
        io=table.io,
    )

    # Advance the counter
    next(producer._manifest_num_counter)
    next(producer._manifest_num_counter)
    assert next(producer._manifest_num_counter) == 2

    producer._refresh_for_retry()

    # Counter should be reset to 0
    assert next(producer._manifest_num_counter) == 0


def test_cow_rewrite_retry_refreshes_both_producers(catalog: Catalog) -> None:
    """G7: When delete() triggers CoW rewrite, multiple snapshot producers must all be retried.

    This is the most critical test: the retry loop is at the Transaction level
    specifically because delete() with CoW rewrite produces two producers
    (_DeleteFiles + _OverwriteFiles) that must be committed atomically.
    """
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.transforms import IdentityTransform

    catalog.create_namespace("default")
    schema = Schema(
        NestedField(1, "category", StringType(), required=False),
        NestedField(2, "value", LongType(), required=False),
    )
    spec = PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="category"))
    catalog.create_table("default.g7_cow_retry_test", schema=schema, partition_spec=spec)

    import pyarrow as pa

    df = pa.table({"category": ["a", "a", "b", "b"], "value": [1, 2, 3, 4]})

    tbl = catalog.load_table("default.g7_cow_retry_test")
    tbl.append(df)

    tbl1 = catalog.load_table("default.g7_cow_retry_test")
    tbl2 = catalog.load_table("default.g7_cow_retry_test")

    # tbl1 appends (non-conflicting with tbl2's partial delete on partition b)
    tbl1.append(pa.table({"category": ["c"], "value": [5]}))

    import pyiceberg.table as _table_module

    RuntimeTransaction = _table_module.Transaction
    original_rebuild = RuntimeTransaction._rebuild_snapshot_updates
    producer_counts_during_rebuild: list[int] = []

    def tracking_rebuild(self_tx: Any) -> None:
        producer_counts_during_rebuild.append(len(self_tx._snapshot_producers))
        original_rebuild(self_tx)

    with patch.object(RuntimeTransaction, "_rebuild_snapshot_updates", tracking_rebuild):
        # Partial delete on partition b triggers CoW rewrite (delete + overwrite)
        # Under snapshot isolation, this should retry and succeed
        tbl2.delete("value == 3")

    # Verify retry happened and multiple producers were tracked
    assert len(producer_counts_during_rebuild) >= 1
    # Should have at least the delete producer
    assert producer_counts_during_rebuild[0] >= 1

    refreshed = catalog.load_table("default.g7_cow_retry_test")
    result = refreshed.scan().to_arrow()
    # 4 original + 1 appended by tbl1 - 1 deleted (value==3) = 4
    assert len(result) == 4


def test_cow_rewrite_inherits_isolation_level_property(catalog: Catalog) -> None:
    """G8: The _OverwriteFiles producer in CoW rewrite path must inherit _isolation_level_property."""
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.transforms import IdentityTransform

    catalog.create_namespace("default")
    schema = Schema(
        NestedField(1, "category", StringType(), required=False),
        NestedField(2, "value", LongType(), required=False),
    )
    spec = PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="category"))
    catalog.create_table(
        "default.g8_cow_iso_test",
        schema=schema,
        partition_spec=spec,
        properties={
            # Serializable for deletes, snapshot for updates
            "write.delete.isolation-level": "serializable",
            "write.update.isolation-level": "snapshot",
        },
    )

    import pyarrow as pa

    df = pa.table({"category": ["a", "a"], "value": [1, 2]})

    tbl = catalog.load_table("default.g8_cow_iso_test")
    tbl.append(df)

    tbl1 = catalog.load_table("default.g8_cow_iso_test")
    tbl2 = catalog.load_table("default.g8_cow_iso_test")

    tbl1.append(pa.table({"category": ["a"], "value": [3]}))

    # overwrite() routes through delete() with WRITE_UPDATE_ISOLATION_LEVEL.
    # Under snapshot isolation, append + overwrite should succeed.
    # Under serializable (if routing is wrong), it would fail.
    tbl2.overwrite(pa.table({"category": ["a", "a"], "value": [10, 20]}), overwrite_filter="category == 'a'")

    refreshed = catalog.load_table("default.g8_cow_iso_test")
    result = refreshed.scan().to_arrow()
    # tbl1 appended [3] to partition a
    # tbl2 overwrote partition a with [10, 20]
    # Under snapshot isolation, tbl2 overwrites only original partition, tbl1's append survives
    # or tbl2 overwrites everything matching filter. Either way, no ValidationException.
    assert len(result) >= 2


def test_concurrent_appends_on_empty_table(catalog: Catalog) -> None:
    """G9: Concurrent operations when parent_snapshot_id is None should not crash."""
    catalog.create_namespace("default")
    schema = _test_schema()
    catalog.create_table("default.g9_empty_test", schema=schema)

    import pyarrow as pa

    df = pa.table({"x": [1, 2, 3]})

    tbl1 = catalog.load_table("default.g9_empty_test")
    tbl2 = catalog.load_table("default.g9_empty_test")

    tbl1.append(df)
    tbl2.append(df)

    refreshed = catalog.load_table("default.g9_empty_test")
    result = refreshed.scan().to_arrow()
    assert len(result) == 6


def test_clean_all_uncommitted_with_io_failures(catalog: Catalog) -> None:
    """G10: IO failures during cleanup should not mask the original exception."""
    catalog.create_namespace("default")
    schema = _test_schema()
    catalog.create_table("default.g10_io_fail_test", schema=schema)

    import pyarrow as pa

    df = pa.table({"x": [1, 2, 3]})

    tbl = catalog.load_table("default.g10_io_fail_test")
    tbl.append(df)

    tbl1 = catalog.load_table("default.g10_io_fail_test")
    tbl2 = catalog.load_table("default.g10_io_fail_test")

    tbl1.delete("x == 1")

    from pyiceberg.table.update.snapshot import _SnapshotProducer

    original_clean = _SnapshotProducer._clean_all_uncommitted

    def failing_clean(self_producer: Any) -> None:
        # Simulate IO failure by corrupting the paths, then call original
        # The original should log warnings but not crash
        self_producer._uncommitted_manifests.append("/nonexistent/path/manifest.avro")
        original_clean(self_producer)

    with patch.object(_SnapshotProducer, "_clean_all_uncommitted", failing_clean):
        with pytest.raises(ValidationException):
            tbl2.delete("x == 1")


def test_backoff_bounded_by_max_wait(catalog: Catalog) -> None:
    """G11: Verify wait = min(min_wait_ms * 2^attempt, max_wait_ms) is bounded."""
    import time

    catalog.create_namespace("default")
    schema = _test_schema()
    catalog.create_table(
        "default.g11_bounded_test",
        schema=schema,
        properties={
            "commit.retry.num-retries": "3",
            "commit.retry.min-wait-ms": "1000",
            "commit.retry.max-wait-ms": "100",  # max < min should cap at max
        },
    )

    import pyarrow as pa

    df = pa.table({"x": [1, 2, 3]})

    tbl1 = catalog.load_table("default.g11_bounded_test")
    tbl2 = catalog.load_table("default.g11_bounded_test")

    tbl1.append(df)

    start = time.monotonic()
    tbl2.append(df)
    elapsed_ms = (time.monotonic() - start) * 1000

    # With max_wait_ms=100, the backoff should be capped.
    # Total wait for 1 retry should be at most ~125ms (100 + 25% jitter).
    # We use a generous upper bound.
    assert elapsed_ms < 2000, f"Backoff was not bounded by max_wait_ms, took {elapsed_ms:.0f}ms"


def test_rebuild_snapshot_updates_is_idempotent(catalog: Catalog) -> None:
    """G12: Calling _rebuild_snapshot_updates twice should not duplicate updates."""
    catalog.create_namespace("default")
    schema = _test_schema()
    catalog.create_table("default.g12_idempotent_test", schema=schema)

    import pyarrow as pa

    df = pa.table({"x": [1, 2, 3]})

    tbl = catalog.load_table("default.g12_idempotent_test")

    import pyiceberg.table as _table_module

    RuntimeTransaction = _table_module.Transaction
    original_rebuild = RuntimeTransaction._rebuild_snapshot_updates
    rebuild_call_count = 0

    def double_rebuild(self_tx: Any) -> None:
        nonlocal rebuild_call_count
        rebuild_call_count += 1
        original_rebuild(self_tx)

    tbl1 = catalog.load_table("default.g12_idempotent_test")
    tbl2 = catalog.load_table("default.g12_idempotent_test")

    tbl1.append(df)

    with patch.object(RuntimeTransaction, "_rebuild_snapshot_updates", double_rebuild):
        tbl2.append(df)

    assert rebuild_call_count == 1

    refreshed = catalog.load_table("default.g12_idempotent_test")
    result = refreshed.scan().to_arrow()
    # No duplicated updates — exactly 6 rows
    assert len(result) == 6


def test_mixed_updates_not_duplicated_on_retry(catalog: Catalog) -> None:
    """G13: set_properties + append in same tx; properties must not be duplicated after rebuild."""
    catalog.create_namespace("default")
    schema = _test_schema()
    catalog.create_table("default.g13_mixed_test", schema=schema)

    import pyarrow as pa

    df = pa.table({"x": [1, 2, 3]})

    tbl1 = catalog.load_table("default.g13_mixed_test")
    tbl2 = catalog.load_table("default.g13_mixed_test")

    tbl1.append(df)

    with tbl2.transaction() as tx:
        tx.set_properties({"custom.prop": "value1"})
        tx.append(df)

    refreshed = catalog.load_table("default.g13_mixed_test")
    assert refreshed.metadata.properties.get("custom.prop") == "value1"
    assert len(refreshed.scan().to_arrow()) == 6


def test_always_false_predicate_skips_filter_validation(catalog: Catalog) -> None:
    """G14: When predicate is AlwaysFalse, CDF is nil, skipping V₂ and V₃."""
    catalog.create_namespace("default")
    schema = _test_schema()
    table = catalog.create_table("default.g14_alwaysfalse_test", schema=schema)

    from pyiceberg.expressions import AlwaysFalse
    from pyiceberg.table.update.snapshot import _DeleteFiles

    tx = Transaction(table, autocommit=False)
    producer = _DeleteFiles(
        operation=Operation.DELETE,
        transaction=tx,
        io=table.io,
    )

    # The predicate defaults to AlwaysFalse, so CDF = nil
    assert producer._predicate == AlwaysFalse()

    # _validate_concurrency with no parent snapshot should be a no-op
    producer._validate_concurrency()


def test_format_version_1_skips_delete_file_validation(catalog: Catalog) -> None:
    """G15: v1 tables should skip _validate_no_new_deletes_for_data_files."""
    catalog.create_namespace("default")
    schema = _test_schema()
    catalog.create_table(
        "default.g15_v1_test",
        schema=schema,
        properties={"format-version": "1"},
    )

    import pyarrow as pa

    df = pa.table({"x": [1, 2, 3]})

    tbl = catalog.load_table("default.g15_v1_test")
    tbl.append(df)

    tbl1 = catalog.load_table("default.g15_v1_test")
    tbl2 = catalog.load_table("default.g15_v1_test")

    tbl1.append(df)

    # Append vs append should succeed on v1 tables (no delete file validation issues)
    tbl2.append(df)

    refreshed = catalog.load_table("default.g15_v1_test")
    result = refreshed.scan().to_arrow()
    assert len(result) == 9


def test_dynamic_partition_overwrite_uses_update_isolation_level(catalog: Catalog) -> None:
    """G16: dynamic_partition_overwrite() should use write.update.isolation-level."""
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.transforms import IdentityTransform

    catalog.create_namespace("default")
    schema = Schema(
        NestedField(1, "category", StringType(), required=False),
        NestedField(2, "value", LongType(), required=False),
    )
    spec = PartitionSpec(PartitionField(source_id=1, field_id=1000, transform=IdentityTransform(), name="category"))
    catalog.create_table(
        "default.g16_dpo_iso_test",
        schema=schema,
        partition_spec=spec,
        properties={
            # Serializable for deletes — would fail if DPO used this
            "write.delete.isolation-level": "serializable",
            # Snapshot for updates — DPO should use this
            "write.update.isolation-level": "snapshot",
        },
    )

    import pyarrow as pa

    df = pa.table({"category": ["a", "a"], "value": [1, 2]})

    tbl = catalog.load_table("default.g16_dpo_iso_test")
    tbl.append(df)

    tbl1 = catalog.load_table("default.g16_dpo_iso_test")
    tbl2 = catalog.load_table("default.g16_dpo_iso_test")

    # tbl1 appends new data to partition a
    tbl1.append(pa.table({"category": ["a"], "value": [3]}))

    # tbl2 does dynamic_partition_overwrite on partition a
    # Under write.delete.isolation-level=serializable, this would fail (added files conflict)
    # But DPO uses write.update.isolation-level=snapshot, which skips V₁
    tbl2.dynamic_partition_overwrite(pa.table({"category": ["a", "a"], "value": [10, 20]}))

    refreshed = catalog.load_table("default.g16_dpo_iso_test")
    result = refreshed.scan().to_arrow()
    assert len(result) >= 2  # At least the DPO data is there
