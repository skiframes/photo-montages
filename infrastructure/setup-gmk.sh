#!/bin/bash
# GMK Edge Device Full Setup Script
# Run this on a fresh Ubuntu 20.04 installation to set up everything
#
# Usage: curl -sL <raw_github_url> | bash
#    or: bash setup-gmk.sh
#
# Prerequisites:
#   - Ubuntu 20.04 (amd64) booted and connected to internet
#   - Run as user 'paul' (with sudo access)
#   - Router configured at 192.168.2.1 (P2 subnet)

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  GMK Edge Device Setup (P2/Pelican 2)  ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# ============================================================
# 1. System packages
# ============================================================
echo -e "${YELLOW}[1/8] Installing system packages...${NC}"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip python3-venv ffmpeg git openssh-server

# ============================================================
# 2. Python virtual environment + dependencies
# ============================================================
echo -e "${YELLOW}[2/8] Setting up Python environment...${NC}"
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip
pip install flask opencv-python-headless numpy Pillow boto3 awscli

# Also install awscli in user space (for sync script which uses system python)
pip3 install --user awscli

# ============================================================
# 3. Clone the repo
# ============================================================
echo -e "${YELLOW}[3/8] Cloning skiframes repo...${NC}"
mkdir -p ~/skiframes
if [ ! -d ~/skiframes/photo-montages/.git ]; then
    git clone git@github.com:skiframes/photo-montages.git ~/skiframes/photo-montages
else
    cd ~/skiframes/photo-montages && git pull
fi

# Create output directory (owned by paul, NOT root)
mkdir -p ~/skiframes/photo-montages/output

# ============================================================
# 4. AWS credentials
# ============================================================
echo -e "${YELLOW}[4/8] Setting up AWS credentials...${NC}"
mkdir -p ~/.aws
if [ ! -f ~/.aws/credentials ]; then
    echo -e "${YELLOW}  AWS credentials not found. Run 'aws configure' to set them up.${NC}"
    echo -e "${YELLOW}  Contact admin for access key and secret.${NC}"
else
    echo -e "${GREEN}  AWS credentials already exist${NC}"
fi

if [ ! -f ~/.aws/config ]; then
    cat > ~/.aws/config << 'CONF'
[default]
region = us-east-1
CONF
fi

# ============================================================
# 5. Systemd services (system-level, need sudo)
# ============================================================
echo -e "${YELLOW}[5/8] Installing systemd services...${NC}"

# Edge service (Flask app)
sudo tee /etc/systemd/system/skiframes-edge.service > /dev/null << 'EOF'
[Unit]
Description=SkiFrames Edge Calibration & Detection
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=paul
WorkingDirectory=/home/paul/skiframes/photo-montages/edge
ExecStart=/home/paul/venv/bin/python3 -u app.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=SKIFRAMES_DEVICE_ID=gmk
Environment=R1_RTSP_URL=rtsp://j40:J40j40j40@192.168.2.101/h264Preview_01_main
Environment=R2_RTSP_URL=rtsp://j40:J40j40j40@192.168.2.102/h264Preview_01_main
Environment=AXIS_RTSP_URL=rtsp://j40:j40@192.168.2.100/axis-media/media.amp
Environment=R3_RTSP_URL=rtsp://j40:J40j40j40@192.168.2.103/h265Preview_01_main

[Install]
WantedBy=multi-user.target
EOF

# Sync service
sudo tee /etc/systemd/system/skiframes-sync.service > /dev/null << 'EOF'
[Unit]
Description=Sync skiframes montages to S3

[Service]
Type=oneshot
User=paul
ExecStart=/home/paul/skiframes/photo-montages/infrastructure/sync-montages.sh --auto
EOF

# Sync timer (every 30 seconds)
sudo tee /etc/systemd/system/skiframes-sync.timer > /dev/null << 'EOF'
[Unit]
Description=Sync skiframes montages every 30 seconds

[Timer]
OnBootSec=60
OnUnitActiveSec=30

[Install]
WantedBy=timers.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable --now skiframes-edge.service
sudo systemctl enable --now skiframes-sync.timer
echo -e "${GREEN}  skiframes-edge.service: $(systemctl is-active skiframes-edge.service)${NC}"
echo -e "${GREEN}  skiframes-sync.timer: $(systemctl is-active skiframes-sync.timer)${NC}"

# ============================================================
# 6. Tailscale VPN
# ============================================================
echo -e "${YELLOW}[6/8] Installing Tailscale...${NC}"
if ! command -v tailscaled &>/dev/null; then
    curl -fsSL https://tailscale.com/install.sh | sh
fi

# Tailscale needs userspace networking on GMK (QEMU/kernel tun device issue)
sudo tee /etc/systemd/system/tailscaled.service > /dev/null << 'EOF'
[Unit]
Description=Tailscale Daemon
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/local/sbin/tailscaled --state=/var/lib/tailscale/tailscaled.state --tun=userspace-networking
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now tailscaled.service

# Check if already authenticated
if ! tailscale status &>/dev/null; then
    echo -e "${YELLOW}  Run 'sudo tailscale up' to authenticate with Tailscale${NC}"
else
    echo -e "${GREEN}  Tailscale: $(tailscale status | head -1)${NC}"
fi

# ============================================================
# 7. Cloudflare Tunnel (user-level service)
# ============================================================
echo -e "${YELLOW}[7/8] Installing Cloudflare Tunnel...${NC}"
if [ ! -f ~/.local/bin/cloudflared ]; then
    mkdir -p ~/.local/bin
    curl -fsSL -o ~/.local/bin/cloudflared \
        https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x ~/.local/bin/cloudflared
fi

# Cloudflare tunnel token for p2.skiframes.com
TUNNEL_TOKEN="eyJhIjoiY2I2YTgxZTUwYjBkMzVmOTMwODFiNjg0ZjI2ODNhM2QiLCJ0IjoiYjcyYWIwOTAtNjMzMC00NDBkLWFjNTgtMDVkMTczNDNkMzNmIiwicyI6Ik1qWTBOamRsTVRVdE0yVXpaUzAwWmpVM0xXRTJZak10TkRreU5qZGxObVUzTnpJdyJ9"

mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/cloudflared.service << CFEOF
[Unit]
Description=Cloudflare Tunnel (gmk-edge)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/home/paul/.local/bin/cloudflared tunnel --no-autoupdate run --token ${TUNNEL_TOKEN}
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
CFEOF

# Enable lingering so user services start on boot without login
sudo loginctl enable-linger paul

systemctl --user daemon-reload
systemctl --user enable --now cloudflared.service
echo -e "${GREEN}  cloudflared: $(systemctl --user is-active cloudflared.service)${NC}"

# ============================================================
# 8. Verification
# ============================================================
echo -e "${YELLOW}[8/8] Verifying setup...${NC}"
echo ""
echo -e "${GREEN}Services:${NC}"
echo "  skiframes-edge  : $(systemctl is-active skiframes-edge.service)"
echo "  skiframes-sync  : $(systemctl is-active skiframes-sync.timer)"
echo "  tailscaled      : $(systemctl is-active tailscaled.service)"
echo "  cloudflared     : $(systemctl --user is-active cloudflared.service)"
echo ""
echo -e "${GREEN}Network:${NC}"
echo "  Local IP  : $(hostname -I | awk '{print $1}')"
echo "  Tailscale : $(tailscale ip -4 2>/dev/null || echo 'not connected')"
echo "  Tunnel    : p2.skiframes.com"
echo ""
echo -e "${GREEN}Paths:${NC}"
echo "  Code    : ~/skiframes/photo-montages/"
echo "  Output  : ~/skiframes/photo-montages/output/"
echo "  AWS creds: ~/.aws/credentials"
echo "  Configs  : ~/skiframes/photo-montages/edge/config/zones/"
echo ""
echo -e "${GREEN}Camera URLs (P2 subnet 192.168.2.x):${NC}"
echo "  R1   : 192.168.2.101 (h264)"
echo "  R2   : 192.168.2.102 (h264)"
echo "  Axis : 192.168.2.100"
echo "  R3   : 192.168.2.103 (h265)"
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup complete!                       ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Calibration UI: http://$(hostname -I | awk '{print $1}'):5000"
echo "Public URL:     https://p2.skiframes.com"
echo ""
echo "All services are enabled and will auto-start on reboot."
