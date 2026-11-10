#!/usr/bin/env bash
# One-time host setup for the Atlas quadlet. Run ON THE HOST as the user that
# owns the rootless Podman store (NOT as root, NOT inside the container):
#
#   ./.devpod/install.sh
#
# It installs files only. It neither stops nor starts the unit - the next
# steps are printed at the end.
set -euo pipefail

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
UNIT_DIR="${XDG_CONFIG_HOME:-${HOME}/.config}/containers/systemd"
DROP_BIN=/usr/local/sbin/atlas-drop-caches
SUDOERS=/etc/sudoers.d/atlas-drop-caches
MEMLOCK_DROPIN=/etc/systemd/system/user@.service.d/atlas-memlock.conf

echo "==> [1/5] quadlet unit -> ${UNIT_DIR}/atlas-gb10.container"
mkdir -p "${UNIT_DIR}"
sed "s|@REPO@|${REPO}|g" "${HERE}/atlas-gb10.container.in" \
    > "${UNIT_DIR}/atlas-gb10.container"

echo "==> [2/5] cache-drop helper -> ${DROP_BIN} (root-owned)"
# Copied out of the checkout on purpose: the sudoers rule must never point at
# a file the unprivileged user (or the container) can rewrite.
# -D: /usr/local/sbin does not exist on every host.
sudo install -D -o root -g root -m 0755 "${HERE}/atlas-drop-caches.sh" "${DROP_BIN}"

echo "==> [3/5] sudoers rule -> ${SUDOERS}"
TMP="$(mktemp)"
trap 'rm -f "${TMP}"' EXIT
echo "${USER_NAME} ALL=(root) NOPASSWD: ${DROP_BIN}" > "${TMP}"
sudo visudo -cf "${TMP}" >/dev/null
sudo install -o root -g root -m 0440 "${TMP}" "${SUDOERS}"
# Prove the rule works the way systemd will use it: -k ignores the sudo
# password cache from the steps above, -n forbids a prompt. This performs
# one real cache drop.
if sudo -k -n "${DROP_BIN}"; then
    echo "    passwordless cache drop ok"
else
    echo "    passwordless cache drop FAILED - check ${SUDOERS}" >&2
    exit 1
fi

echo "==> [4/5] memlock limit for user services -> ${MEMLOCK_DROPIN}"
# Rootless Podman cannot raise RLIMIT_MEMLOCK above the hard limit of the
# systemd user manager (default 8 MiB), so `--ulimit memlock=-1` would fail
# at container start without this.
sudo mkdir -p "$(dirname "${MEMLOCK_DROPIN}")"
printf '[Service]\nLimitMEMLOCK=infinity\n' | sudo tee "${MEMLOCK_DROPIN}" >/dev/null
sudo systemctl daemon-reload

echo "==> [5/5] linger (start at boot without login) + user daemon-reload"
sudo loginctl enable-linger "${USER_NAME}"
systemctl --user daemon-reload

echo "    generated unit check:"
QUADLET=/usr/libexec/podman/quadlet
[[ -x "${QUADLET}" ]] || QUADLET=/usr/lib/podman/quadlet
if [[ -x "${QUADLET}" ]]; then
    "${QUADLET}" -dryrun -user >/dev/null && echo "    quadlet dry-run ok"
else
    echo "    quadlet binary not found - check: systemctl --user cat atlas-gb10"
fi

cat <<EOF

Installed. The unit is used from the next container start on. A (re)start
ends every shell / Claude session in the container.

  1. First install only: the memlock limit applies to a fresh user manager -
     simplest: reboot (the unit then starts on its own). Check afterwards:
       systemd-run --user --pipe sh -c 'ulimit -l'    # expect: unlimited
  2. Otherwise, to apply unit changes right away (builds the image first and
     replaces an existing container of the same name):
       systemctl --user restart atlas-gb10
  3. Watch:
       systemctl --user status atlas-gb10
       journalctl --user -u atlas-gb10 -f

Environment changes: edit ${HERE}/atlas-gb10.env, then
  systemctl --user restart atlas-gb10
Unit changes: edit atlas-gb10.container.in and re-run this script.
EOF
