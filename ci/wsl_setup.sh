# Runs as root inside a fresh WSL2 Ubuntu 24.04 distro: install Docker Engine the way a
# Windows user following "Docker Engine in WSL" guides does, and start the daemon.
set -eu
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq docker.io curl openssh-client python3 ca-certificates iproute2 >/dev/null
docker --version
if ! docker info >/dev/null 2>&1; then
    nohup dockerd > /var/log/dockerd.log 2>&1 &
    for i in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 2; done
fi
docker info 2>&1 | grep -E 'Server Version|Docker Root Dir|Cgroup|Operating System|Total Memory|CPUs'
df -h / | tail -1
