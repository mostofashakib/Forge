#!/usr/bin/env bash
# Prepare a fresh Ubuntu 24.04 VM for Forge. Safe to run more than once.
#
#   sudo ./provision-vm.sh base
#       Docker Engine and Compose, Docker log rotation, unattended upgrades,
#       key-only SSH with root login off, and the Tailscale package.
#
#   sudo --preserve-env=SSH_CONNECTION ./provision-vm.sh lockdown <tailnet-host>
#       Firewall (only tailscale0 may connect), the tailnet TLS certificate with a
#       monthly renewal, and a weekly Docker prune. Refuses to run unless this SSH
#       session arrives over Tailscale, so it cannot cut off the session running it.
set -euo pipefail

FORGE_DIR=/opt/forge
CERT_DIR=/etc/forge/certs
LOGIN_USER="${SUDO_USER:-ubuntu}"

say() { echo "[provision] $*"; }
has_systemd() { [[ -d /run/systemd/system ]]; }

write_if_changed() {  # $1: path, stdin: content. Returns 0 when the file changed.
    local tmp; tmp=$(mktemp)
    cat >"$tmp"
    if [[ -f "$1" ]] && cmp -s "$tmp" "$1"; then rm -f "$tmp"; return 1; fi
    install -m 0644 "$tmp" "$1"; rm -f "$tmp"; return 0
}

base() {
    export DEBIAN_FRONTEND=noninteractive
    say "updating packages"
    apt-get update -q
    apt-get upgrade -yq
    apt-get install -yq ca-certificates curl gnupg ufw unattended-upgrades sqlite3

    say "installing Docker Engine and the Compose plugin"
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    # shellcheck disable=SC1091
    codename=$(. /etc/os-release && echo "$VERSION_CODENAME")
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $codename stable" \
        | write_if_changed /etc/apt/sources.list.d/docker.list || true
    apt-get update -q
    apt-get install -yq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    usermod -aG docker "$LOGIN_USER"

    say "Docker log rotation: 50 MB per file, 5 files per container"
    mkdir -p /etc/docker
    if printf '{\n  "log-driver": "json-file",\n  "log-opts": { "max-size": "50m", "max-file": "5" }\n}\n' \
        | write_if_changed /etc/docker/daemon.json && has_systemd; then
        systemctl restart docker
    fi

    say "unattended security upgrades"
    printf 'APT::Periodic::Update-Package-Lists "1";\nAPT::Periodic::Unattended-Upgrade "1";\n' \
        | write_if_changed /etc/apt/apt.conf.d/20auto-upgrades || true

    say "SSH: keys only, no root login"
    mkdir -p /etc/ssh/sshd_config.d
    if printf 'PasswordAuthentication no\nKbdInteractiveAuthentication no\nPermitRootLogin no\n' \
        | write_if_changed /etc/ssh/sshd_config.d/60-forge.conf && has_systemd; then
        sshd -t
        systemctl reload ssh
    fi

    say "installing Tailscale"
    curl -fsSL "https://pkgs.tailscale.com/stable/ubuntu/$codename.noarmor.gpg" -o /usr/share/keyrings/tailscale-archive-keyring.gpg
    curl -fsSL "https://pkgs.tailscale.com/stable/ubuntu/$codename.tailscale-keyring.list" \
        | write_if_changed /etc/apt/sources.list.d/tailscale.list || true
    apt-get update -q
    apt-get install -yq tailscale

    mkdir -p "$CERT_DIR" "$FORGE_DIR"
    chown "$LOGIN_USER:$LOGIN_USER" "$FORGE_DIR"
    has_systemd || say "no systemd here (a container?), so services were not restarted"
    say "base done. Next: tailscale up, then lockdown over Tailscale SSH."
}

require_tailscale_session() {
    local server_ip
    server_ip=$(awk '{ print $3 }' <<<"${SSH_CONNECTION:-}")
    if [[ -z "$server_ip" ]]; then
        echo "SSH_CONNECTION is empty. Run: sudo --preserve-env=SSH_CONNECTION $0 lockdown <tailnet-host>" >&2
        exit 1
    fi
    # Tailscale addresses come from 100.64.0.0/10.
    if ! [[ "$server_ip" =~ ^100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\. ]]; then
        echo "This session reached the VM at $server_ip, not over Tailscale. Reconnect with 'ssh forge' first." >&2
        exit 1
    fi
    tailscale status >/dev/null || { echo "tailscale is not running" >&2; exit 1; }
}

install_timer() {  # $1: name, $2: OnCalendar, $3: command
    printf '[Unit]\nDescription=Forge %s\n\n[Service]\nType=oneshot\nExecStart=/bin/sh -c %q\n' "$1" "$3" \
        | write_if_changed "/etc/systemd/system/forge-$1.service" || true
    printf '[Unit]\nDescription=Forge %s timer\n\n[Timer]\nOnCalendar=%s\nPersistent=true\n\n[Install]\nWantedBy=timers.target\n' "$1" "$2" \
        | write_if_changed "/etc/systemd/system/forge-$1.timer" || true
    systemctl daemon-reload
    systemctl enable --now "forge-$1.timer"
}

lockdown() {
    local host="${1:?usage: lockdown <tailnet-host>, for example forge.tail1234.ts.net}"
    require_tailscale_session

    say "firewall: deny all inbound except on tailscale0"
    ufw default deny incoming
    ufw default allow outgoing
    ufw allow in on tailscale0
    ufw --force enable

    say "TLS certificate for $host"
    local cert_cmd="tailscale cert --cert-file $CERT_DIR/cert.pem --key-file $CERT_DIR/key.pem $host"
    $cert_cmd
    local compose="docker compose -f $FORGE_DIR/deploy/docker-compose.yml --env-file $FORGE_DIR/deploy/.env.production"
    install_timer cert-renew monthly "$cert_cmd && ($compose exec -T caddy caddy reload --config /etc/caddy/Caddyfile || true)"

    say "weekly prune of unused images and build cache older than 7 days"
    install_timer docker-prune weekly 'docker image prune -af --filter "until=168h" && docker builder prune -af --filter "until=168h"'

    say "lockdown done"
}

[[ $EUID -eq 0 ]] || { echo "run as root: sudo $0 $*" >&2; exit 1; }
case "${1:-}" in
    base) base ;;
    lockdown) shift; lockdown "$@" ;;
    *) echo "usage: $0 base | lockdown <tailnet-host>" >&2; exit 2 ;;
esac
