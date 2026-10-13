FROM python:3.12-slim

# The runner starts the submission as an unmapped, non-root UID. The writable
# /tmp directory is supplied as a tmpfs by DockerRunner.
RUN useradd --system --uid 65532 --no-create-home sandbox
USER sandbox
WORKDIR /sandbox
