"""Smoke test: can PyFlink's Table API actually read from our Redpanda topic and print rows?
Proves the Kafka connector jar + JSON format + watermark declaration all work before any
windowing logic is added on top.
"""
from pathlib import Path

from pyflink.table import EnvironmentSettings, TableEnvironment

JAR_PATH = Path(__file__).resolve().parent / "jars" / "flink-sql-connector-kafka-5.0.0-2.2.jar"


def main():
    env_settings = EnvironmentSettings.in_streaming_mode()
    table_env = TableEnvironment.create(env_settings)
    table_env.get_config().set("pipeline.jars", f"file://{JAR_PATH}")

    table_env.execute_sql("""
        CREATE TABLE transactions_raw (
            TransactionID BIGINT,
            TransactionDT BIGINT,
            TransactionAmt DOUBLE,
            card1 BIGINT,
            event_time AS TO_TIMESTAMP(FROM_UNIXTIME(TransactionDT)),
            WATERMARK FOR event_time AS event_time - INTERVAL '5' MINUTE
        ) WITH (
            'connector' = 'kafka',
            'topic' = 'transactions.raw',
            'properties.bootstrap.servers' = 'localhost:19092',
            'properties.group.id' = 'flink-smoke-test',
            'scan.startup.mode' = 'earliest-offset',
            'format' = 'json',
            'json.ignore-parse-errors' = 'true'
        )
    """)

    result = table_env.execute_sql(
        "SELECT TransactionID, TransactionDT, TransactionAmt, card1 FROM transactions_raw LIMIT 10"
    )
    result.print()


if __name__ == "__main__":
    main()
