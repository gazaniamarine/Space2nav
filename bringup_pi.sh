#!/usr/bin/env bash
# Bring up the rover's Pi-side services (rover-camera, rover-agent,
# rover-zenoh) over SSH so its Zenoh topics (image_raw, depth_raw,
# image_raw/camera_info, rover/rpm) actually exist before
# gui_rover.py / run_stagewise_demo.py --pi-ip tries to subscribe to them.
#
# Mirrors LemNAV's LAUNCH/lib/backend.sh::_bringup_rover (same physical
# rover, same systemd service names) -- restart, don't reinvent: this
# doesn't run the optional UDP camera sender path (start_udp_camera.sh);
# rover-camera already publishes over Zenoh on its own, which is all
# RoverMotionAdapter subscribes to.
#
# Usage:
#   PI_PASS=... ./bringup_pi.sh <PI_IP>
set -uo pipefail

PI_IP="${1:?Usage: PI_PASS=... ./bringup_pi.sh <PI_IP>}"
PI_USER="${PI_USER:-pi}"
PI_PASS="${PI_PASS:?export PI_PASS first}"

ssh_pi() { sshpass -p "$PI_PASS" ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$PI_USER@$PI_IP" "$@"; }

echo "[bringup] pinging $PI_IP ..."
ping -c5 -i0.5 -W2 "$PI_IP" >/dev/null 2>&1 || { echo "[bringup] FAIL: Pi unreachable -- is it ON and on this network?"; exit 1; }
echo "[bringup] Pi reachable."

ssh_pi 'echo ssh_ok' 2>/dev/null | grep -q ssh_ok || { echo "[bringup] FAIL: SSH failed (user=$PI_USER, check PI_PASS)"; exit 1; }
echo "[bringup] SSH OK."

services="rover-camera rover-agent rover-zenoh"
n=3
# A cold restart of an already-streaming D435i can crash-loop for 20-30s --
# leave a healthy camera alone (same guard as LemNAV's launcher).
if ssh_pi "journalctl -u rover-camera -n 6 --no-pager 2>/dev/null | grep -q '\[STATUS\]'" 2>/dev/null; then
    echo "[bringup] rover-camera already streaming -- not restarting it."
    services="rover-agent rover-zenoh"
    n=2
fi

echo "[bringup] restarting: $services"
active_count=$(ssh_pi "echo $PI_PASS | sudo -S -p '' systemctl restart $services 2>/dev/null; sleep 4; systemctl is-active $services" | grep -c active)
if [[ "$active_count" == "$n" ]]; then
    echo "[bringup] services active: $services"
else
    echo "[bringup] WARNING: not all services came up active -- check: ssh $PI_USER@$PI_IP 'journalctl -u rover-camera -u rover-agent -u rover-zenoh -n 30'"
fi

echo "[bringup] waiting for camera to stream (up to 40s)..."
cam_ok=false
for i in $(seq 1 20); do
    sleep 2
    if ssh_pi "journalctl -u rover-camera -n 4 --no-pager 2>/dev/null | grep -q 'color_published='" 2>/dev/null; then
        cam_ok=true
        echo "[bringup] camera streaming [$((i*2))s]"
        break
    fi
done
$cam_ok || echo "[bringup] WARNING: camera not confirmed streaming -- check: ssh $PI_USER@$PI_IP 'journalctl -u rover-camera -n 20'"

echo "[bringup] checking ESP32 heartbeat (/rover/rpm, up to 25s)..."
if ssh_pi 'bash -lc "source /opt/ros/humble/setup.bash; timeout 25 ros2 topic echo /rover/rpm --once 2>/dev/null"' 2>/dev/null \
    | grep -q "layout\|data"; then
    echo "[bringup] ESP32 alive (/rover/rpm publishing)."
else
    echo "[bringup] WARNING: no /rover/rpm -- check: ssh $PI_USER@$PI_IP 'journalctl -u rover-agent -n 20'"
fi

echo "[bringup] done."
