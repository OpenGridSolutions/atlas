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
#      helper or install.sh itself (asks for sudo there)
#   4. systemctl --user restart atlas-gb10 - the restart rebuilds the image if
#      the Containerfile changed, and serve.sh rebuilds the binary
#
# The nodes get origin/<branch>, not this working tree: commit and push first.
# A failing node does not stop the others; the summary lists it.
set -euo pipefail

usage() {
    # The header comment above, up to `set -euo pipefail`.
    sed -n '2,/^set -euo pipefail/s/^# \{0,1\}//p' "${BASH_SOURCE[0]}" | sed '/^#/d'
}

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
echo "==> state to roll out: origin/${BRANCH}"
git -C "${REPO}" fetch --quiet origin "${BRANCH}"
TARGET_SHA="$(git -C "${REPO}" rev-parse "origin/${BRANCH}")"
echo "    $(git -C "${REPO}" log -1 --format='%h %s' "${TARGET_SHA}")"
UNPUSHED="$(git -C "${REPO}" rev-list --count "origin/${BRANCH}..HEAD")"
if [[ "${UNPUSHED}" -gt 0 ]]; then
    if [[ ${ALLOW_UNPUSHED} -eq 0 ]]; then
        echo "    ${UNPUSHED} local commit(s) not on origin - push first, or pass --allow-unpushed" >&2
        exit 1
    fi
    echo "    WARNING: ${UNPUSHED} local commit(s) not on origin are NOT rolled out" >&2
fi
if [[ -n "$(git -C "${REPO}" status --porcelain --untracked-files=no)" ]]; then
    echo "    note: uncommitted changes in this checkout are NOT rolled out"
fi

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
        remote_install "${target}"
    else
        echo "     not needed (unit and helper unchanged)"
    fi

    echo "  -> [4/4] restart ${UNIT}"
    remote "${target}" "systemctl --user reset-failed ${UNIT} 2>/dev/null || true
systemctl --user restart ${UNIT}
systemctl --user is-active ${UNIT}"
}

FAILED=()
for node in "${NODES[@]}"; do
    echo "==> ${node}"
    # Subshell: set -e aborts this node only. Not inside `if`/`||`, where
    # bash would switch set -e off for the whole function.
    set +e
    (set -e; update_node "${node}")
    rc=$?
    set -e
    if [[ ${rc} -ne 0 ]]; then
        echo "  !! ${node} failed (exit ${rc})" >&2
        FAILED+=("${node}")
    fi
done

echo
echo "Updated $(( ${#NODES[@]} - ${#FAILED[@]} )) of ${#NODES[@]} node(s) to ${TARGET_SHA:0:9}."
if [[ ${#FAILED[@]} -gt 0 ]]; then
    printf 'Failed: %s\n' "${FAILED[@]}" >&2
    exit 1
fi
echo "serve.sh now rebuilds the binary on each node; follow one with:"
echo "    ssh <node> journalctl --user -u ${UNIT} -f"
