# The CLI environment: the pinned Ubuntu base plus libfaketime, so commands
# read a fixed clock instead of the host's. ContainerRuntime passes the base
# image and sets LD_PRELOAD and FAKETIME at launch.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
# An exact version from the signed jammy archive resolves to exactly one
# file, so this layer is the same on every build.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libfaketime=0.9.8-9 \
 && rm -rf /var/lib/apt/lists/* \
 && ln -s "$(dpkg -L libfaketime | grep '/libfaketime.so.1$')" /usr/local/lib/libfaketime.so.1
