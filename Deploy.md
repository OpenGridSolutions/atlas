podman build -f docker/gb10/Dockerfile -t atlas-gb10:latest .

podman login git.intelliprocess.dev

podman tag atlas-gb10:latest git.intelliprocess.dev/heldvonkosmos/atlas-gb10:latest

podman push git.intelliprocess.dev/heldvonkosmos/atlas-gb10:latest
