#!/usr/bin/env bash
# Set up a new DGX Spark as an Atlas node, the same way as this one. Run ON
# THE MAIN SPARK'S HOST as the Podman user (like install.sh):
#
#   ./.devpod/cluster/provision.sh <ssh-target> --allow-from <cidr>
#
#   <ssh-target>  user@host of the node, e.g. jgroth@192.168.4.95; that user
#                 becomes the node's Podman user (not root; needs sudo).
#                 The SSH password is asked once (all steps share one
#                 connection), the sudo password per step that needs it.
#                 Optional: ssh-copy-id <ssh-target> saves the SSH password.
#   --allow-from  network allowed to reach the API port through the node's
#                 firewall, e.g. 192.168.4.0/24 (the server runs without
#                 --require-auth).
#   -h, --help    this text
#
# Steps:
#   1. prerequisites on the node (grow the LVM volume holding $HOME into the
#      free NVMe space, podman/git/rsync, CDI spec for the GPU, subuid
#      range); compares driver and Podman with this host
#   2. rsync the HF hub (~/.cache/huggingface/hub) to the node
#   3. git clone this checkout's origin + branch to the same path below $HOME
#   4. .devpod/host.env with ATLAS_PUBLISH_ADDR=0.0.0.0 (the API port is
#      published on the network, not on 127.0.0.1 as behind a local proxy)
#   5. firewall: allow <cidr> to the API port (ufw or firewalld, if active)
#   6. .devpod/install.sh on the node (sudo; on a fresh host it offers the
#      reboot the memlock limit needs - the unit then comes up on its own)
#   7. record the node in .devpod/cluster/nodes (update.sh's list) and start
#      the unit; the first start builds the image and the release binary, and
#      writes one snapshot per model - that takes a while
#
# Runs inside the tmux session atlas-provision-<host> (started on its own;
# detach with Ctrl-b d - Ctrl-C stops it -, come back with tmux attach -t
# atlas-provision-<host>), so a dropped SSH connection to this host does not
# stop it. The output also goes to logs/cluster/atlas-provision-<host>.log.
# Pushes to the ntfy channel of atlas-gb10.env (ATLAS_NTFY_TOPIC): models
# synced, input needed (sudo password), done, failed.
#
# Idempotent: re-running it on a provisioned node only applies what differs.
# What the node runs is origin/<branch>: commit and push first - uncommitted
# or unpushed changes on this host do not reach the node.
set -euo pipefail

usage() {
    # The header comment above, up to `set -euo pipefail`.
    sed -n '2,/^set -euo pipefail/s/^# \{0,1\}//p' "${BASH_SOURCE[0]}" | sed '/^#/d'
}

ORIG_ARGS=("$@")
TARGET=""
ALLOW_FROM=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --allow-from) ALLOW_FROM="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*) usage >&2; exit 2 ;;
        *) [[ -z "${TARGET}" ]] || { usage >&2; exit 2; }; TARGET="$1"; shift ;;
    esac
done
if [[ -z "${TARGET}" || -z "${ALLOW_FROM}" ]]; then
    usage >&2
    exit 2
fi
if [[ ! "${ALLOW_FROM}" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}/[0-9]{1,2}$ ]]; then
    echo "--allow-from must be an IPv4 network like 192.168.4.0/24, got '${ALLOW_FROM}'" >&2
    exit 2
fi

# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
JOB_NAME="provision ${TARGET}"
cluster_init "atlas-provision-${TARGET#*@}" "${ORIG_ARGS[@]}"

ORIGIN_URL="$(git -C "${REPO}" remote get-url origin)"
BRANCH="$(git -C "${REPO}" rev-parse --abbrev-ref HEAD)"
if [[ "${BRANCH}" == HEAD ]]; then
    echo "this checkout is on a detached HEAD - check out the branch the nodes should follow" >&2
    exit 1
fi

step "[1/7] prerequisites on ${TARGET}"
remote "${TARGET}" '
if [[ "$(id -u)" -eq 0 ]]; then
    echo "    log in as the Podman user, not as root" >&2
    exit 1
fi
# Disk: the Ubuntu installer leaves most of the NVMe unused (a 100 GiB root
# LV in a 3.7 TB volume group). Grow the LV holding $HOME (models, images,
# snapshots) into all free space of its VG; -r grows ext4/xfs online.
home_dev="$(findmnt -no SOURCE -T "$HOME")"
home_fs="$(findmnt -no FSTYPE -T "$HOME")"
if [[ "$home_dev" == /dev/mapper/* ]] && sudo lvs "$home_dev" >/dev/null 2>&1; then
    vg="$(sudo lvs --noheadings -o vg_name "$home_dev" | tr -d " ")"
    free="$(sudo vgs --noheadings --units b --nosuffix -o vg_free "$vg" | tr -d " ")"
    if (( free < 1073741824 )); then
        echo "    disk: $home_dev already uses its whole volume group"
    elif [[ "$home_fs" != ext4 && "$home_fs" != xfs ]]; then
        echo "    disk: $(( free >> 30 )) GiB free in $vg, but $home_fs is not grown online - left as is" >&2
    else
        echo "    disk: growing $home_dev by $(( free >> 30 )) GiB (all free space of $vg)"
        sudo lvextend -r -l +100%FREE "$home_dev"
    fi
else
    echo "    disk: $home_dev is no LVM volume - not resized"
fi
df -h "$HOME" | sed "s/^/    /"
# Command -> package. Quiet apt: no debconf dialog, no needrestart scan.
declare -A pkg=([podman]=podman [git]=git [rsync]=rsync [nvidia-ctk]=nvidia-container-toolkit)
apt_install() {
    sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_SUSPEND=1 apt-get install -y -qq "$@"
}
missing=()
for c in podman git rsync nvidia-ctk; do
    command -v "$c" >/dev/null || missing+=("${pkg[$c]}")
done
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "    installing: ${missing[*]}"
    sudo apt-get update -qq
    # The container toolkit comes from the NVIDIA apt repository; add it
    # (the documented install path) when no configured source offers it.
    if [[ " ${missing[*]} " == *" nvidia-container-toolkit "* ]] &&
        ! apt-cache policy nvidia-container-toolkit | grep -q "Candidate: [^(]"; then
        echo "    adding the NVIDIA container toolkit apt repository"
        keyring=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |
            sudo gpg --batch --yes --dearmor -o "$keyring"
        curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
            sed "s#deb https://#deb [signed-by=$keyring] https://#g" |
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
        sudo apt-get update -qq
    fi
    apt_install "${missing[@]}"
fi
if ! grep -q "^$(id -un):" /etc/subuid; then
    echo "    no /etc/subuid range for $(id -un) - rootless Podman needs one:" >&2
    echo "    sudo usermod --add-subuids 100000-165535 --add-subgids 100000-165535 $(id -un)" >&2
    exit 1
fi
if ! nvidia-ctk cdi list 2>/dev/null | grep -qx "nvidia.com/gpu=all"; then
    echo "    generating the CDI spec for the GPU (/etc/cdi/nvidia.yaml)"
    sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
fi
echo "    ok"
'
HERE_DRIVER="$(driver_version)"
NODE_DRIVER="$(driver_version "${TARGET}")"
HERE_PODMAN="$(podman --version)"
NODE_PODMAN="$(remote "${TARGET}" 'podman --version' </dev/null | tr -d '\r')"
echo "    driver: here ${HERE_DRIVER:-unknown}, node ${NODE_DRIVER:-unknown}"
echo "    podman: here ${HERE_PODMAN}, node ${NODE_PODMAN}"
if [[ "${HERE_DRIVER}" != "${NODE_DRIVER}" ]]; then
    echo "    WARNING: different drivers - the server is tested with this host's (${HERE_DRIVER:-unknown})" >&2
fi
if [[ "${NODE_DRIVER}" == 610.* ]]; then
    echo "    WARNING: the 610 branch under-reports free memory, the KV pool comes out far too small - update the node's driver (README, Quick start)" >&2
fi
if [[ "${HERE_PODMAN}" != "${NODE_PODMAN}" ]]; then
    echo "    WARNING: different Podman versions - the unit is only tested with this host's" >&2
fi

step "[2/7] HF models ${HF_HUB} -> ${TARGET}:~/${HF_HUB_REL}"
SYNC_START=${SECONDS}
sync_models "${TARGET}"
notify "Atlas ${JOB_NAME}: models synced" "HF hub copied in $(( (SECONDS - SYNC_START) / 60 )) min, continuing with the checkout" default package

step "[3/7] checkout ${ORIGIN_URL} (${BRANCH}) -> ${TARGET}:~/${REPO_REL}"
remote "${TARGET}" '
dir="$HOME/$1" url="$2" branch="$3"
if [[ -d "$dir/.git" ]]; then
    have="$(git -C "$dir" remote get-url origin)"
    if [[ "$have" != "$url" ]]; then
        echo "    $dir exists with origin $have, expected $url - not touching it" >&2
        exit 1
    fi
    echo "    exists - pulling"
    git -C "$dir" pull --ff-only origin "$branch"
elif [[ -e "$dir" ]]; then
    echo "    $dir exists and is not a git checkout - not touching it" >&2
    exit 1
else
    mkdir -p "$(dirname "$dir")"
    git clone --branch "$branch" "$url" "$dir"
fi
' "${REPO_REL}" "${ORIGIN_URL}" "${BRANCH}"

step "[4/7] ${TARGET}:~/${REPO_REL}/.devpod/host.env (API port on 0.0.0.0)"
remote "${TARGET}" 'echo "ATLAS_PUBLISH_ADDR=0.0.0.0" > "$HOME/$1/.devpod/host.env"' "${REPO_REL}"

# sudo caches the password per terminal, and every remote call gets a new
# one: each sudo step asks again, so each one pushes "input needed".
need_input "${TARGET}" "sudo password for the firewall rule"
step "[5/7] firewall: ${ALLOW_FROM} -> port ${API_PORT}/tcp"
remote "${TARGET}" '
port="$1" cidr="$2"
if command -v ufw >/dev/null && sudo ufw status | grep -q "^Status: active"; then
    sudo ufw allow from "$cidr" to any port "$port" proto tcp comment atlas-gb10
elif systemctl is-active --quiet firewalld; then
    rule="rule family=ipv4 source address=$cidr port port=$port protocol=tcp accept"
    if sudo firewall-cmd --permanent --query-rich-rule="$rule" >/dev/null; then
        echo "    unchanged"
    else
        sudo firewall-cmd --permanent --add-rich-rule="$rule"
        sudo firewall-cmd --reload
    fi
else
    echo "    no active firewall (ufw/firewalld) - port $port is reachable from every network the node is on"
fi
' "${API_PORT}" "${ALLOW_FROM}"

need_input "${TARGET}" "sudo password for install.sh"
step "[6/7] install.sh on ${TARGET}"
# Bind-mount sources of the unit that must exist before the first start.
remote "${TARGET}" 'mkdir -p "$HOME/.claude" "$HOME/$1/logs"' "${REPO_REL}"
REBOOTED=0
remote_install "${TARGET}" || {
    rc=$?
    # 255 = the SSH connection dropped: install.sh rebooted the node.
    [[ ${rc} -eq 255 ]] || exit "${rc}"
    REBOOTED=1
}

step "[7/7] register in ${NODES_FILE} and start"
if [[ ! -e "${NODES_FILE}" ]] || ! grep -qxF "${TARGET}" "${NODES_FILE}"; then
    echo "${TARGET}" >> "${NODES_FILE}"
    echo "    added"
else
    echo "    already listed"
fi
if [[ ${REBOOTED} -eq 1 ]]; then
    echo "    node is rebooting - ${UNIT} starts on its own (linger)"
else
    remote "${TARGET}" "systemctl --user reset-failed ${UNIT} 2>/dev/null || true; systemctl --user start --no-block ${UNIT}"
    echo "    started"
fi

notify "Atlas ${JOB_NAME}: done" "Node provisioned$([[ ${REBOOTED} -eq 1 ]] && echo ", rebooting"). First start builds image, binary and snapshots." default white_check_mark
NODE_HOST="${TARGET#*@}"
cat <<EOF

Provisioned ${TARGET}. The first start builds the image, the release binary
and one snapshot per model - follow it with:
    ssh ${TARGET} journalctl --user -u ${UNIT} -f
Then:
    curl -s http://${NODE_HOST}:${API_PORT}/v1/models
Later updates of all nodes: ./.devpod/cluster/update.sh
EOF
