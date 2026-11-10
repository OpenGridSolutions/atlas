#!/bin/sh
# Hand reclaimable page cache back to the GB10 unified-memory pool.
# NVIDIA's documented stopgap for DGX Spark; must run as root ON THE HOST.
# Installed root-owned to /usr/local/sbin/atlas-drop-caches by install.sh and
# called from the quadlet's ExecStopPost via a NOPASSWD sudoers rule.
set -eu
sync
echo 3 > /proc/sys/vm/drop_caches
