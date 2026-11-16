#!/usr/bin/env bash
# Stop the Atlas unit, rebuild its image, start it again. Run ON THE HOST as
# the user that owns the rootless Podman store (like install.sh):
#
#   ./.devpod/build.sh          # reinstall Claude Code (current release),
#                                 # everything else from the layer cache
#   ./.devpod/build.sh --full   # rebuild every layer: apt packages, Rust
#                                 # toolchain, sccache, pdfium, Claude Code
#
# How a rebuild is forced: the Containerfile COPYs two stamp files,
# .stamp-full right after FROM and .stamp-claude right before the Claude Code
# install. This script rewrites them, and a changed file invalidates its
# layer and every layer below it. The build command itself is exactly the
# unit's ExecStartPre, so the build the unit repeats on every start resolves
# to the image built here.
#
# Do NOT force a rebuild with `podman build --no-cache` instead: a later
# cached build (the unit's ExecStartPre) matches the OLDEST fitting layers in
# the store and moves the tag back to the old image.
#
# Model lists, server code and env settings need none of this: the image only
# holds the toolchain. For those, `systemctl --user restart atlas-gb10`.
set -euo pipefail

usage() {
    echo "usage: $(basename "$0") [--full]"
    echo "  (no option)  reinstall Claude Code, everything else from the layer cache"
    echo "  --full       rebuild every layer"
}

FULL=0
if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
fi
case "${1:-}" in
    "") ;;
    --full) FULL=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

if [[ "$(id -u)" -eq 0 ]]; then
    echo "run as the Podman user, not as root" >&2
    exit 1
fi
if [[ -e /run/.containerenv ]]; then
    echo "run on the host, not inside the container" >&2
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT=atlas-gb10.service
CONTAINER=atlas-gb10
IMAGE=localhost/atlas-gb10-dev:local
CONTAINERFILE="${HERE}/Containerfile"
STAMPS=(.stamp-full .stamp-claude)

# Without the COPY lines the stamps change nothing and the build would be a
# silent layer-cache no-op.
for stamp in "${STAMPS[@]}"; do
    if ! grep -Eq "^COPY[[:space:]]+${stamp//./\\.}[[:space:]]" "${CONTAINERFILE}"; then
        echo "${CONTAINERFILE} has no 'COPY ${stamp} ...' line - this script needs the Containerfile that goes with it" >&2
        exit 1
    fi
done

image_id() {
    podman image inspect --format '{{.Id}}' "${IMAGE}" 2>/dev/null || true
}

# claude_version IMAGE_ID: version of the Claude Code inside that image.
claude_version() {
    local version=""
    if [[ -n "$1" ]]; then
        version="$(podman run --rm --network none --entrypoint /usr/local/bin/claude "$1" --version 2>/dev/null | head -n 1 || true)"
    fi
    echo "${version:-unknown}"
}

echo "==> [1/4] stop ${UNIT}"
# No-op when already stopped. On a running unit this also triggers the
# ExecStopPost cache drop, so the build starts with the memory handed back.
systemctl --user stop "${UNIT}"

OLD_ID="$(image_id)"
OLD_CLAUDE="$(claude_version "${OLD_ID}")"

# Rewrite the stamps. If the build does not succeed they are put back, so
# that the unit's own build stays a cache hit on the previous image.
BACKUP="$(mktemp -d)"
BUILD_OK=0
restore_stamps() {
    if [[ "${BUILD_OK}" -eq 0 ]]; then
        for stamp in "${STAMPS[@]}"; do
            if [[ -e "${BACKUP}/${stamp}" ]]; then
                cat "${BACKUP}/${stamp}" > "${HERE}/${stamp}"
            else
                rm -f "${HERE}/${stamp}"
            fi
        done
    fi
    rm -rf "${BACKUP}"
}
trap restore_stamps EXIT
for stamp in "${STAMPS[@]}"; do
    [[ ! -e "${HERE}/${stamp}" ]] || cp "${HERE}/${stamp}" "${BACKUP}/${stamp}"
done

NOW="$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"
if [[ "${FULL}" -eq 1 || ! -e "${HERE}/.stamp-full" ]]; then
    echo "${NOW}" > "${HERE}/.stamp-full"
fi
echo "${NOW}" > "${HERE}/.stamp-claude"

if [[ "${FULL}" -eq 1 ]]; then
    echo "==> [2/4] build ${IMAGE} (every layer)"
else
    echo "==> [2/4] build ${IMAGE} (Claude Code layer; rest from the cache)"
fi
# Keep identical to ExecStartPre in atlas-gb10.container.in.
if ! podman build --tag "${IMAGE}" --file "${CONTAINERFILE}" "${HERE}"; then
    cat >&2 <<EOF

Build FAILED - ${UNIT} stays stopped, the stamp files are back to their
previous content. Bring the server back on the previous image with:
    systemctl --user start ${UNIT}
EOF
    exit 1
fi
BUILD_OK=1
NEW_ID="$(image_id)"
NEW_CLAUDE="$(claude_version "${NEW_ID}")"
OLD_SHORT="${OLD_ID:0:12}"
echo "    image:       ${OLD_SHORT:-none} -> ${NEW_ID:0:12}"
echo "    Claude Code: ${OLD_CLAUDE} -> ${NEW_CLAUDE}"

echo "==> [3/4] start ${UNIT}"
# A manual rebuild is a deliberate start: clear a tripped crash-loop brake
# (StartLimitBurst) so systemd does not refuse it.
systemctl --user reset-failed "${UNIT}" 2>/dev/null || true
# The unit's ExecStartPre build runs again here and is a layer-cache no-op.
systemctl --user start "${UNIT}"

echo "==> [4/4] check that the container runs the new image"
RUNNING_ID="$(podman container inspect --format '{{.Image}}' "${CONTAINER}" 2>/dev/null || true)"
if [[ "${RUNNING_ID}" != "${NEW_ID}" ]]; then
    cat >&2 <<EOF
    MISMATCH: container ${CONTAINER} runs image '${RUNNING_ID:0:12}', built was ${NEW_ID:0:12}.
    The unit's own build (ExecStartPre) produced a different image - compare
    its command in atlas-gb10.container.in with the build in this script and
    check: journalctl --user -u ${UNIT} -n 50
EOF
    exit 1
fi
echo "    ok (${NEW_ID:0:12})"

# The superseded image is untagged now. Remove it: it only costs disk space,
# and its layers are what a later cached build could match by mistake.
if [[ -n "${OLD_ID}" && "${OLD_ID}" != "${NEW_ID}" ]]; then
    if [[ "$(podman image inspect --format '{{len .RepoTags}}' "${OLD_ID}" 2>/dev/null || true)" == 0 ]]; then
        if podman rmi "${OLD_ID}" >/dev/null 2>&1; then
            echo "    removed superseded image ${OLD_ID:0:12}"
        else
            echo "    superseded image ${OLD_ID:0:12} is still in use - not removed"
        fi
    fi
fi

echo
echo "Follow the log: journalctl --user -u ${UNIT} -f"
