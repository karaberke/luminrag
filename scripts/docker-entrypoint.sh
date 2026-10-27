#!/bin/sh
set -eu

# Named volumes from older self-hosted releases were written by root.
# Repair only writable data/cache mounts, then drop every bootstrap privilege.
# Code, configuration, home and verified model weights remain root-owned.
chown -R lumin:lumin /app/backend/data/processed /app/backend/data/raw /var/cache/luminrag
exec gosu lumin "$@"
