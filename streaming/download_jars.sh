#!/usr/bin/env bash
# Downloads the Flink Kafka connector jar matching our installed PyFlink 2.3.0.
# Connector versioning is independent of Flink core versioning (e.g. "5.0.0-2.2" = connector
# 5.0.0 built for Flink 2.2) -- no exact 2.3 build exists yet, 2.2 is binary-compatible since
# it's a patch-level difference within the Flink 2.x line.
set -euxo pipefail
mkdir -p "$(dirname "$0")/jars"
curl -sL -o "$(dirname "$0")/jars/flink-sql-connector-kafka-5.0.0-2.2.jar" \
    https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-kafka/5.0.0-2.2/flink-sql-connector-kafka-5.0.0-2.2.jar
ls -la "$(dirname "$0")/jars/"
