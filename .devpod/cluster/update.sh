#!/usr/bin/env bash
# Roll the main spark's state out to every provisioned node. Run ON THE MAIN
# SPARK'S HOST as the Podman user (like install.sh):
#
#   ./.devpod/cluster/update.sh [--delete-models] [--allow-unpushed] [node...]
#
#   node...           only these ssh targets (default: every line of
#                     .devpod/cluster/nodes, which provision.sh maintains)
#   --delete-models   also delete models on the nodes that are gone here
#   --allow-unpushed  roll out although this checkout has commits that are
#                     not on origin (the nodes then run an older state)
#   -h, --help        this text
#
# Per node:
#   1. rsync the HF hub (~/.cache/huggingface/hub) to the node
#   2. git pull --ff-only in the node's checkout
#   3. install.sh again if the pull changed the unit template, the cache-drop
#      helper or install.sh itself (sudo: the password of such nodes is asked
#      once at the start; a reboot it needs is reported, not done)
#   4. systemctl --user restart atlas-gb10 - the restart rebuilds the image if
#      the Containerfile changed, and serve.sh rebuilds the binary
#
# The nodes get origin/<branch>, not this working tree: commit and push first.
# A failing node does not stop the others; the summary lists it.
# Runs inside the tmux session atlas-update (started on its own; detach with
# Ctrl-b d - Ctrl-C stops it -, back with tmux attach -t atlas-update; output
# also in logs/cluster/atlas-update.log) and pushes to the ntfy
# channel of atlas-gb10.env: input needed, node failed, done.
set -euo pipefail

usage() {
    # The header comment above, up to `set -euo pipefail`.
    sed -n '2,/^set -euo pipefail/s/^# \{0,1\}//p' "${BASH_SOURCE[0]}" | sed '/^#/d'
}

ORIG_ARGS=("$@")
DELETE_MODELS=()
ALLOW_UNPUSHED=0
NODES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --delete-models) DELETE_MODELS=(--delete); shift ;;
        --allow-unpushed) ALLOW_UNPUSHED=1; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) usage >&2; exit 2 ;;
        *) NODES+=("$1"); shift ;;
    esac
done

# shellcheck source=common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
JOB_NAME="update"
cluster_init atlas-update "${ORIG_ARGS[@]}"

if [[ ${#NODES[@]} -eq 0 ]]; then
    if [[ ! -r "${NODES_FILE}" ]]; then
        echo "no ${NODES_FILE} - provision a node first (provision.sh adds it)" >&2
        exit 1
    fi
    mapfile -t NODES < <(sed -e 's/#.*//' -e 's/[[:space:]]//g' -e '/^$/d' "${NODES_FILE}")
    if [[ ${#NODES[@]} -eq 0 ]]; then
        echo "${NODES_FILE} lists no nodes" >&2
        exit 1
    fi
fi

BRANCH="$(git -C "${REPO}" rev-parse --abbrev-ref HEAD)"
step "state to roll out: origin/${BRANCH}"
check_pushed "${BRANCH}" "${ALLOW_UNPUSHED}"

# update_node TARGET: steps 1-4 for one node.
update_node() {
    local target="$1" old new
    echo "  -> [1/4] HF models"
    sync_models "${target}" "${DELETE_MODELS[@]}"

    echo "  -> [2/4] git pull"
    old="$(remote "${target}" 'git -C "$HOME/$1" rev-parse HEAD' "${REPO_REL}" </dev/null | tr -d '\r')"
    remote "${target}" 'git -C "$HOME/$1" pull --ff-only origin "$2"' "${REPO_REL}" "${BRANCH}"
    new="$(remote "${target}" 'git -C "$HOME/$1" rev-parse HEAD' "${REPO_REL}" </dev/null | tr -d '\r')"
    echo "     ${old:0:9} -> ${new:0:9}"
    if [[ "${new}" != "${TARGET_SHA}" ]]; then
        echo "     WARNING: node is at ${new:0:9}, origin/${BRANCH} is ${TARGET_SHA:0:9}" >&2
    fi

    echo "  -> [3/4] install.sh"
    if [[ "${old}" != "${new}" ]] && ! git -C "${REPO}" diff --quiet "${old}" "${new}" -- "${INSTALL_INPUTS[@]}"; then
        need_input "${target}" "sudo password for install.sh on ${target}"
        remote_install "${target}" no
    else
        echo "     not needed (unit and helper unchanged)"
    fi

    echo "  -> [4/4] restart ${UNIT}"
    remote "${target}" "systemctl --user reset-failed ${UNIT} 2>/dev/null || true
systemctl --user restart ${UNIT}
systemctl --user is-active ${UNIT}"
}

# Nodes whose pull will change the installer inputs need sudo for
# install.sh: ask their passwords now, so the rest runs unattended.
step "sudo passwords (only nodes that need install.sh)"
for node in "${NODES[@]}"; do
    old="$(remote "${node}" 'git -C "$HOME/$1" rev-parse HEAD' "${REPO_REL}" </dev/null 2>/dev/null | tr -d '\r')" || continue
    [[ -n "${old}" && "${old}" != "${TARGET_SHA}" ]] || continue
    if ! git -C "${REPO}" diff --quiet "${old}" "${TARGET_SHA}" -- "${INSTALL_INPUTS[@]}" 2>/dev/null; then
        sudo_setup "${node}" || echo "    no sudo for ${node} - install.sh there will ask" >&2
    fi
done

FAILED=()
for node in "${NODES[@]}"; do
    step "${node}"
    # Subshell: set -e aborts this node only. Not inside `if`/`||`, where
    # bash would switch set -e off for the whole function.
    set +e
    (set -e; update_node "${node}")
    rc=$?
    set -e
    if [[ ${rc} -ne 0 ]]; then
        echo "  !! ${node} failed (exit ${rc})" >&2
        notify "Atlas update: ${node} failed" "exit ${rc} - see tmux attach -t atlas-update on $(hostname)" high warning
        FAILED+=("${node}")
    fi
done

echo
echo "Updated $(( ${#NODES[@]} - ${#FAILED[@]} )) of ${#NODES[@]} node(s) to ${TARGET_SHA:0:9}."
if [[ ${#FAILED[@]} -gt 0 ]]; then
    printf 'Failed: %s\n' "${FAILED[@]}" >&2
    CURRENT_STEP="failed nodes: ${FAILED[*]}"
    exit 1
fi
notify "Atlas update: done" "${#NODES[@]} node(s) at ${TARGET_SHA:0:9}, servers restarting" default white_check_mark
echo "serve.sh now rebuilds the binary on each node; follow one with:"
echo "    ssh <node> journalctl --user -u ${UNIT} -f"
