#!/bin/bash
#
# Build manylinux1 wheels. Based on the example at
# <https://github.com/pypa/python-manylinux-demo>
#
# Run this within the repository root:
#   docker run --rm -v $(pwd):/io quay.io/pypa/manylinux1_x86_64 /io/buildwheels.sh
#
# The wheels will be put into the wheelhouse/ subdirectory.
#
# For interactive tests:
#   docker run -it -v $(pwd):/io quay.io/pypa/manylinux1_x86_64 /bin/bash

set -xeuo pipefail

# For convenience, if this script is called from outside of a docker container,
# it starts a container and runs itself inside of it.
if ! grep -q docker /proc/1/cgroup; then
  # We are not inside a container
  docker pull quay.io/pypa/manylinux1_x86_64
  exec docker run --rm -v $(pwd):/io quay.io/pypa/manylinux1_x86_64 /io/$0
fi

if ! test -d /io/dist; then
  mkdir /io/dist
  chown --reference=/io/setup.py /io/dist
fi

# Strip binaries (copied from multibuild)
STRIP_FLAGS=${STRIP_FLAGS:-"-Wl,-strip-all"}
export CFLAGS="${CFLAGS:-$STRIP_FLAGS}"
export CXXFLAGS="${CXXFLAGS:-$STRIP_FLAGS}"

for PYBIN in /opt/python/cp3[678]-*/bin; do
    ${PYBIN}/pip wheel /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/tinyalign-*.whl; do
    auditwheel repair "$whl" -w repaired/
done

# Created files are owned by root, so fix permissions.
chown -R --reference=/io/setup.py repaired/
mv repaired/*.whl /io/dist/
