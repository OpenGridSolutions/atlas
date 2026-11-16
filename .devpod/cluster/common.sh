# Shared pieces of provision.sh and update.sh - sourced, not run.
#
# Both scripts run ON THE MAIN SPARK'S HOST as the Podman user (the one that
# owns the Atlas checkout and ~/.cache/huggingface), never in the container:
# the container has neither SSH keys nor the host's systemd.

if [[ "$(id -u)" -eq 0 ]]; then
    echo "run as the Podman user, not as root" >&2
    exit 1
fi
if [[ -e /run/.containerenv ]]; then
    echo "run on the host, not inside the container" >&2
    exit 1
fi

# Tools this host needs itself (the node's are checked by provision.sh).
for tool in git ssh rsync; do
    if ! command -v "${tool}" >/dev/null; then
        echo "${tool} is missing on this host - install it first: sudo apt install ${tool}" >&2
        exit 1
    fi
done

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVPOD_DIR="$(dirname "${CLUSTER_DIR}")"
REPO="$(dirname "${DEVPOD_DIR}")"
NODES_FILE="${CLUSTER_DIR}/nodes"
UNIT=atlas-gb10.service

# The checkout sits at the same place below $HOME on every node.
if [[ "${REPO}" != "${HOME}"/* ]]; then
    echo "checkout ${REPO} is not below ${HOME} - cannot derive the path on the nodes" >&2
    exit 1
fi
REPO_REL="${REPO#"${HOME}"/}"

# The HF cache the quadlet mounts (%h/.cache/huggingface). Only hub/ is
# synced: no token, no xet chunk cache.
HF_HUB_REL=.cache/huggingface/hub
HF_HUB="${HOME}/${HF_HUB_REL}"

# Host port of the API: read from the unit template, the one place it is set.
API_PORT="$(sed -n 's/^PublishPort=@PUBLISH_ADDR@:\([0-9]*\):.*/\1/p' "${DEVPOD_DIR}/atlas-gb10.container.in")"
if [[ ! "${API_PORT}" =~ ^[0-9]+$ ]]; then
    echo "no 'PublishPort=@PUBLISH_ADDR@:<port>:...' line in ${DEVPOD_DIR}/atlas-gb10.container.in" >&2
    exit 1
fi

# One SSH connection per node, reused by every ssh/rsync call to it.
SSH_CTL="$(mktemp -d)"
trap 'ssh_close_all; rm -rf "${SSH_CTL}"' EXIT
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=${SSH_CTL}/%C" -o ControlPersist=120)
ssh_close_all() {
    local sock
    for sock in "${SSH_CTL}"/*; do
        [[ -S "${sock}" ]] && ssh -o "ControlPath=${sock}" -O exit dummy 2>/dev/null || true
    done
}

# remote TARGET SCRIPT [ARGS...]: run a bash SCRIPT on TARGET with ARGS as
# $1.. . Gets a terminal when we have one, so sudo can ask for a password.
remote() {
    local target="$1" script="$2"
    shift 2
    local tty=-T
    [[ -t 0 ]] && tty=-t
    # Shipped base64-encoded: survives any login shell (sh, bash, zsh)
    # without quoting trouble. ARGS must be plain words (paths, URLs).
    local a b64
    for a in "$@"; do
        if [[ ! "${a}" =~ ^[A-Za-z0-9@%+=:,./_~-]+$ ]]; then
            echo "remote: argument '${a}' is not a plain word" >&2
            return 2
        fi
    done
    b64="$(printf 'set -euo pipefail\n%s\n' "${script}" | base64 -w0)"
    # shellcheck disable=SC2029  # expanded locally on purpose
    ssh "${SSH_OPTS[@]}" "${tty}" "${target}" \
        "bash -c \"\$(echo ${b64} | base64 -d)\" atlas-remote $*"
}

# sync_models TARGET [--delete]: copy the main spark's HF hub to TARGET.
# Resumable (--partial), keeps the cache's symlinks (snapshots -> blobs).
# --delete also removes models on TARGET that are gone here.
sync_models() {
    local target="$1"
    shift
    local extra=("$@")
    if [[ ! -d "${HF_HUB}" ]]; then
        echo "    no HF hub at ${HF_HUB}" >&2
        return 1
    fi
    remote "${target}" 'mkdir -p "$HOME/$1"' "${HF_HUB_REL}"
    rsync -a --partial --info=progress2,stats1 --human-readable \
        --exclude '.locks/' --exclude '*.incomplete' \
        "${extra[@]}" \
        -e "ssh ${SSH_OPTS[*]}" \
        "${HF_HUB}/" "${target}:${HF_HUB_REL}/"
}

# Files whose change needs install.sh on a node (unit, root-owned helper,
# the installer itself); everything else only needs a unit restart.
INSTALL_INPUTS=(.devpod/install.sh .devpod/atlas-gb10.container.in .devpod/atlas-drop-caches.sh)

# remote_install TARGET: run install.sh on TARGET (asks for sudo, may offer
# the reboot a fresh host needs for the memlock limit - the unit comes up on
# its own after it).
remote_install() {
    remote "$1" 'cd "$HOME/$1" && ./.devpod/install.sh' "${REPO_REL}"
}

# driver_version [TARGET]: NVIDIA driver here or on TARGET.
driver_version() {
    local q='nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1'
    if [[ $# -eq 0 ]]; then
        bash -c "${q}" || true
    else
        remote "$1" "${q}" </dev/null 2>/dev/null | tr -d '\r' || true
    fi
}
