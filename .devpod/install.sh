#!/usr/bin/env bash
# Host setup for the Atlas quadlet. Run ON THE HOST as the user that owns the
# rootless Podman store (NOT as root, NOT inside the container):
#
#   ./.devpod/install.sh
#
# No flags: every setting comes from .devpod/atlas-gb10.env (the same file
# the container reads), so the checkout is the single source of truth.
#
# Idempotent: every step writes only what differs, so re-running it is safe.
# It never stops or starts the Atlas unit itself. Settings that only apply
# after a host reboot (memlock limit, huge-page pool, NVIDIA driver memory
# pools) are detected at the end, and the reboot is the LAST thing the script does, after a prompt.
set -euo pipefail

if [[ $# -gt 0 ]]; then
    echo "install.sh takes no arguments - settings live in .devpod/atlas-gb10.env" >&2
    exit 2
fi

if [[ "$(id -u)" -eq 0 ]]; then
    echo "run as the Podman user, not as root (sudo is called where needed)" >&2
    exit 1
fi
if [[ -e /run/.containerenv ]]; then
    echo "run on the host, not inside the container" >&2
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "${HERE}")"
USER_NAME="$(id -un)"
USER_CONFIG="${XDG_CONFIG_HOME:-${HOME}/.config}"
UNIT_DIR="${USER_CONFIG}/containers/systemd"
DROP_BIN=/usr/local/sbin/atlas-drop-caches
SUDOERS=/etc/sudoers.d/atlas-drop-caches
MEMLOCK_DROPIN=/etc/systemd/system/user@.service.d/atlas-memlock.conf
MEMLOCK_LOGIN=/etc/security/limits.d/90-atlas-memlock.conf
HUGEPAGES_CONF=/etc/sysctl.d/90-atlas-hugepages.conf
NVIDIA_POOLS_CONF=/etc/modprobe.d/atlas-nvidia-memory-pools.conf
NVIDIA_PARAMS=/proc/driver/nvidia/params
ENV_FILE="${HERE}/atlas-gb10.env"

# Huge-page pool size: required, no implicit default.
HUGEPAGES_GB="$(sed -n 's/^ATLAS_HUGEPAGES_GB=//p' "${ENV_FILE}")"
if [[ ! "${HUGEPAGES_GB}" =~ ^[0-9]+$ ]]; then
    echo "ATLAS_HUGEPAGES_GB must be set to a whole number of GiB in ${ENV_FILE} (0 = no pool)" >&2
    exit 1
fi

# NVIDIA driver memory pools: required, no implicit default.
NVIDIA_POOLS="$(sed -n 's/^ATLAS_NVIDIA_SYSTEM_MEMORY_POOLS=//p' "${ENV_FILE}")"
if [[ ! "${NVIDIA_POOLS}" =~ ^([0-9]+|driver)$ ]]; then
    echo "ATLAS_NVIDIA_SYSTEM_MEMORY_POOLS must be set in ${ENV_FILE}: 0 (the driver returns freed memory to the kernel) or 'driver' (keep the driver default)" >&2
    exit 1
fi

TMP="$(mktemp)"
trap 'rm -f "${TMP}"' EXIT

# install_file DEST MODE [sudo]: install ${TMP} at DEST only when the content
# differs. Prints "updated" or "unchanged".
install_file() {
    local dest="$1" mode="$2" run="${3:-}"
    if ${run} cmp -s "${TMP}" "${dest}" 2>/dev/null; then
        echo "    unchanged"
        return
    fi
    if [[ -n "${run}" ]]; then
        sudo install -D -o root -g root -m "${mode}" "${TMP}" "${dest}"
    else
        install -D -m "${mode}" "${TMP}" "${dest}"
    fi
    echo "    updated"
}

echo "==> [1/8] quadlet unit -> ${UNIT_DIR}/atlas-gb10.container"
sed "s|@REPO@|${REPO}|g" "${HERE}/atlas-gb10.container.in" > "${TMP}"
install_file "${UNIT_DIR}/atlas-gb10.container" 0644

echo "==> [2/8] cache-drop helper -> ${DROP_BIN} (root-owned)"
# Copied out of the checkout on purpose: the sudoers rule must never point at
# a file the unprivileged user (or the container) can rewrite.
cat "${HERE}/atlas-drop-caches.sh" > "${TMP}"
install_file "${DROP_BIN}" 0755 sudo

echo "==> [3/8] sudoers rule -> ${SUDOERS}"
echo "${USER_NAME} ALL=(root) NOPASSWD: ${DROP_BIN}" > "${TMP}"
sudo visudo -cf "${TMP}" >/dev/null
install_file "${SUDOERS}" 0440 sudo
# Prove the rule works the way systemd will use it: -k ignores the sudo
# password cache from the steps above, -n forbids a prompt. This performs
# one real cache drop.
if sudo -k -n "${DROP_BIN}"; then
    echo "    passwordless cache drop ok"
else
    echo "    passwordless cache drop FAILED - check ${SUDOERS}" >&2
    exit 1
fi

echo "==> [4/8] memlock limit for user services -> ${MEMLOCK_DROPIN}"
# Rootless Podman cannot raise RLIMIT_MEMLOCK above the hard limit of the
# systemd user manager (default 8 MiB); the quadlet's `Ulimit=memlock=-1:-1`
# has no effect without this. Applies to a fresh user manager, i.e. after
# a reboot.
printf '[Service]\nLimitMEMLOCK=infinity\n' > "${TMP}"
install_file "${MEMLOCK_DROPIN}" 0644 sudo
sudo systemctl daemon-reload

echo "==> [5/8] memlock limit for login sessions -> ${MEMLOCK_LOGIN}"
# `podman exec` inherits the rlimits of the calling login shell (PAM), not of
# the user manager; without this crun cannot apply the container's
# memlock=-1 on exec. Applies to new login sessions.
printf '%s - memlock unlimited\n' "${USER_NAME}" > "${TMP}"
install_file "${MEMLOCK_LOGIN}" 0644 sudo

echo "==> [6/8] huge-page pool (ATLAS_HUGEPAGES_GB=${HUGEPAGES_GB}) -> ${HUGEPAGES_CONF}"
# Written as a boot-time sysctl only, never applied live: a pool appearing
# under a running server changes its free-memory accounting, and a large
# pool is only reliably available before memory fragments.
if [[ "${HUGEPAGES_GB}" -eq 0 ]]; then
    if [[ -e "${HUGEPAGES_CONF}" ]]; then
        sudo rm -f "${HUGEPAGES_CONF}"
        echo "    removed"
    else
        echo "    unchanged (no reservation)"
    fi
else
    PAGE_KB="$(awk '/^Hugepagesize:/ {print $2}' /proc/meminfo)"
    if [[ "${PAGE_KB}" != 2048 ]]; then
        echo "    default huge page size is ${PAGE_KB:-unknown} kB, expected 2048 - not writing a pool" >&2
        exit 1
    fi
    printf '# Atlas self-managed memory block: %s GiB of 2 MiB pages (.devpod/install.sh)\nvm.nr_hugepages = %s\n' \
        "${HUGEPAGES_GB}" "$((HUGEPAGES_GB * 512))" > "${TMP}"
    install_file "${HUGEPAGES_CONF}" 0644 sudo
fi

echo "==> [7/8] NVIDIA driver memory pools (ATLAS_NVIDIA_SYSTEM_MEMORY_POOLS=${NVIDIA_POOLS}) -> ${NVIDIA_POOLS_CONF}"
# With its system memory pools on, the driver keeps freed GPU allocations
# cached instead of returning the pages to the kernel. On GB10 that memory
# then belongs to no process and appears in no free-memory reading, so the
# server sizes its KV pool (hence every session's context window) from a
# number that is too small by whatever the last model load staged. A module
# option: the driver reads it when it loads, i.e. after a reboot.
POOLS_CHANGED=0
if [[ "${NVIDIA_POOLS}" == driver ]]; then
    if [[ -e "${NVIDIA_POOLS_CONF}" ]]; then
        sudo rm -f "${NVIDIA_POOLS_CONF}"
        POOLS_CHANGED=1
        echo "    removed"
    else
        echo "    unchanged (driver default)"
    fi
else
    printf '# Atlas: freed GPU memory goes back to the kernel (.devpod/install.sh)\noptions nvidia NVreg_EnableSystemMemoryPools=%s\n' \
        "${NVIDIA_POOLS}" > "${TMP}"
    install_file "${NVIDIA_POOLS_CONF}" 0644 sudo
fi
# The initramfs carries its own copy of /etc/modprobe.d; refresh it when it
# is older than the option (also catches a refresh that failed last time).
INITRD="/boot/initrd.img-$(uname -r)"
if command -v update-initramfs >/dev/null && [[ -e "${INITRD}" ]] \
    && { [[ "${POOLS_CHANGED}" -eq 1 ]] || [[ "${NVIDIA_POOLS_CONF}" -nt "${INITRD}" ]]; }; then
    sudo update-initramfs -u
    echo "    initramfs refreshed"
fi

echo "==> [8/8] linger (start at boot without login) + user daemon-reload"
sudo loginctl enable-linger "${USER_NAME}"
systemctl --user daemon-reload
QUADLET=/usr/libexec/podman/quadlet
[[ -x "${QUADLET}" ]] || QUADLET=/usr/lib/podman/quadlet
if [[ -x "${QUADLET}" ]]; then
    "${QUADLET}" -dryrun -user >/dev/null && echo "    quadlet dry-run ok"
else
    echo "    quadlet binary not found - check: systemctl --user cat atlas-gb10"
fi

# ── What is not live yet? ──
REBOOT_REASONS=()
MANAGER_PID="$(systemctl --user show --property MainPID --value 2>/dev/null || true)"
[[ "${MANAGER_PID}" =~ ^[1-9][0-9]*$ ]] || MANAGER_PID="$(pgrep -u "${USER_NAME}" -xo systemd || true)"
MEMLOCK_HARD="$(awk '/^Max locked memory/ {print $5}' "/proc/${MANAGER_PID:-0}/limits" 2>/dev/null || true)"
if [[ "${MEMLOCK_HARD}" != unlimited ]]; then
    REBOOT_REASONS+=("memlock limit of the user manager is '${MEMLOCK_HARD:-unknown}', wanted 'unlimited'")
fi
WANT_PAGES=0
[[ -e "${HUGEPAGES_CONF}" ]] && WANT_PAGES="$(awk -F= '/^vm.nr_hugepages/ {gsub(/ /, "", $2); print $2}' "${HUGEPAGES_CONF}")"
HAVE_PAGES="$(cat /proc/sys/vm/nr_hugepages)"
if [[ "${WANT_PAGES}" != "${HAVE_PAGES}" ]]; then
    REBOOT_REASONS+=("huge-page pool is ${HAVE_PAGES} pages, configured ${WANT_PAGES}")
fi
if [[ "${NVIDIA_POOLS}" != driver ]]; then
    LIVE_POOLS="$(awk '/^EnableSystemMemoryPools:/ {print $2}' "${NVIDIA_PARAMS}" 2>/dev/null || true)"
    if [[ "${LIVE_POOLS}" != "${NVIDIA_POOLS}" ]]; then
        REBOOT_REASONS+=("NVIDIA driver memory pools are '${LIVE_POOLS:-unknown}' (${NVIDIA_PARAMS}), configured ${NVIDIA_POOLS}")
    fi
elif [[ "${POOLS_CHANGED}" -eq 1 ]]; then
    REBOOT_REASONS+=("NVIDIA driver memory pools go back to the driver default")
fi

cat <<EOF

Installed.
  Apply unit/env changes without a reboot (ends every shell / Claude session
  in the container, rebuilds the image first):
    systemctl --user restart atlas-gb10
  Watch:
    systemctl --user status atlas-gb10
    journalctl --user -u atlas-gb10 -f
  Environment changes: edit ${HERE}/atlas-gb10.env, then restart the unit.
  Unit changes: edit atlas-gb10.container.in and re-run this script.
EOF

if [[ ${#REBOOT_REASONS[@]} -eq 0 ]]; then
    echo
    if [[ "$(ulimit -Hl)" == unlimited ]]; then
        echo "No reboot required - everything is live."
    else
        echo "No reboot required, but this login session still has the old memlock"
        echo "limit - log out and back in before using 'podman exec'."
    fi
    exit 0
fi

echo
echo "A host reboot is required:"
printf '  - %s\n' "${REBOOT_REASONS[@]}"
echo "After it: atlas-gb10 comes up on its own."
if [[ ! -t 0 ]]; then
    echo "Not rebooting (no terminal). Run: sudo systemctl reboot"
    exit 0
fi
read -r -p "Reboot now? [y/N] " ANSWER
if [[ ! "${ANSWER}" =~ ^[yYjJ]$ ]]; then
    echo "Not rebooting. Run: sudo systemctl reboot"
    exit 0
fi
sudo systemctl reboot
