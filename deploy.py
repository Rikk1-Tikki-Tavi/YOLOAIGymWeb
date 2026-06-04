"""
AIGym deployment script — uploads files and configures the VPS via SSH/SFTP.
Run with the backend venv Python.
"""
import os
import sys
import time
import pathlib
import paramiko
from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient  # installed via paramiko extras

HOST     = "43.157.201.110"
USER     = "ubuntu"
PASSWORD = "panda-86$-thunder"
APP_DIR  = "/var/www/aigym"

ROOT      = pathlib.Path(__file__).parent
FRONTEND  = ROOT / "frontend" / "dist"
BACKEND   = ROOT / "backend"

NGINX_CONF = r"""
server {
    listen 80;
    server_name _;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name _;

    ssl_certificate     /etc/ssl/aigym/cert.pem;
    ssl_certificate_key /etc/ssl/aigym/key.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    root /var/www/aigym/frontend;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /ws/ {
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_read_timeout 3600;
    }

    location ~ ^/(health|exercises) {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
    }
}
"""

SYSTEMD_SERVICE = r"""
[Unit]
Description=AIGym FastAPI Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/var/www/aigym/backend
Environment="MODEL_PATH=yolo11s-pose.pt"
Environment="NUM_THREADS=4"
ExecStart=/var/www/aigym/backend/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""

SETUP_SCRIPT = f"""#!/bin/bash
set -e

echo "=== Creating app directories ==="
sudo mkdir -p {APP_DIR}/frontend {APP_DIR}/backend
sudo chown -R ubuntu:ubuntu {APP_DIR}

echo "=== Installing system packages ==="
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip nginx libgl1 libglib2.0-0

echo "=== Setting up Python venv ==="
cd {APP_DIR}/backend
python3 -m venv venv
venv/bin/pip install --quiet --upgrade pip
venv/bin/pip install --quiet -r requirements.txt

echo "=== Writing nginx config ==="
sudo mkdir -p /etc/ssl/aigym
cat > /tmp/aigym_nginx.conf << 'NGINXEOF'
{NGINX_CONF}
NGINXEOF
sudo cp /tmp/aigym_nginx.conf /etc/nginx/sites-available/aigym
sudo ln -sf /etc/nginx/sites-available/aigym /etc/nginx/sites-enabled/aigym
sudo rm -f /etc/nginx/sites-enabled/default

echo "=== Writing systemd service ==="
cat > /tmp/aigym.service << 'SVCEOF'
{SYSTEMD_SERVICE}
SVCEOF
sudo cp /tmp/aigym.service /etc/systemd/system/aigym.service
sudo systemctl daemon-reload
sudo systemctl enable aigym

echo "=== Opening firewall ==="
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

echo "=== Done with setup script ==="
"""


def ssh_run(client, cmd, desc=""):
    if desc:
        print(f"  >> {desc}")
    stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    exit_code = stdout.channel.recv_exit_status()
    if out.strip():
        for line in out.strip().splitlines()[-6:]:
            print(f"     {line}")
    if exit_code != 0 and err.strip():
        print(f"  [STDERR] {err.strip()[:300]}")
    return exit_code


def upload_dir(scp, local_dir: pathlib.Path, remote_dir: str):
    """Recursively upload a directory via SCP."""
    print(f"  Uploading {local_dir.name}/ → {remote_dir}/")
    scp.put(str(local_dir), remote_path=remote_dir, recursive=True)


def main():
    print(f"\n{'='*60}")
    print(f"  AIGym Deployment -> {HOST}")
    print(f"{'='*60}\n")

    # ── 1. Connect ────────────────────────────────────────────────
    print("[1/6] Connecting to VPS...")
    client = SSHClient()
    client.set_missing_host_key_policy(AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=30)
    print("      Connected.\n")

    # ── 2. Create directories + install deps ──────────────────────
    print("[2/6] Running server setup script...")
    ssh_run(client, f"echo '{SETUP_SCRIPT}' > /tmp/aigym_setup.sh && bash /tmp/aigym_setup.sh",
            "apt + venv + nginx + systemd")
    print()

    # ── 3. Upload frontend ────────────────────────────────────────
    print("[3/6] Uploading frontend dist/...")
    with SCPClient(client.get_transport(), progress=lambda f,s,t: None) as scp:
        # Upload contents of dist/ into frontend/
        for item in FRONTEND.iterdir():
            dest = f"{APP_DIR}/frontend"
            if item.is_dir():
                scp.put(str(item), remote_path=dest, recursive=True)
            else:
                scp.put(str(item), remote_path=f"{dest}/{item.name}")
    print("      Frontend uploaded.\n")

    # ── 4. Upload backend ─────────────────────────────────────────
    print("[4/6] Uploading backend files...")
    SKIP = {"venv", "__pycache__", ".pyc"}
    with SCPClient(client.get_transport()) as scp:
        for item in BACKEND.iterdir():
            if item.name in SKIP or item.suffix == ".pyc":
                continue
            dest = f"{APP_DIR}/backend"
            if item.is_dir():
                scp.put(str(item), remote_path=dest, recursive=True)
            else:
                scp.put(str(item), remote_path=f"{dest}/{item.name}")
    print("      Backend uploaded.\n")

    # ── 5. Re-install Python deps (in case requirements changed) ──
    print("[5/6] Installing Python requirements on server...")
    ssh_run(client,
            f"cd {APP_DIR}/backend && venv/bin/pip install --quiet -r requirements.txt",
            "pip install")
    print()

    # ── 6. Restart services ───────────────────────────────────────
    print("[6/6] Starting/restarting services...")
    ssh_run(client, "sudo systemctl restart aigym", "restart aigym")
    time.sleep(3)
    ssh_run(client, "sudo systemctl status aigym --no-pager -l", "aigym status")
    ssh_run(client, "sudo nginx -t && sudo systemctl restart nginx", "restart nginx")
    print()

    # ── Summary ───────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"  Deployment complete!")
    print(f"  VPS IP : {HOST}")
    print(f"  HTTP   : http://{HOST}  (redirects to HTTPS once cert is in place)")
    print(f"\n  NOTE: HTTPS/WSS requires the SSL cert files at:")
    print(f"    /etc/ssl/aigym/cert.pem")
    print(f"    /etc/ssl/aigym/key.pem")
    print(f"  Upload them with:")
    print(f"    scp your_cert.pem ubuntu@{HOST}:/tmp/cert.pem")
    print(f"    scp your_key.pem  ubuntu@{HOST}:/tmp/key.pem")
    print(f"    ssh ubuntu@{HOST} 'sudo mv /tmp/cert.pem /etc/ssl/aigym/ && sudo mv /tmp/key.pem /etc/ssl/aigym/'")
    print(f"\n  DNS record to add:")
    print(f"    A   <your-subdomain>   {HOST}   TTL 300")
    print(f"{'='*60}\n")

    client.close()


if __name__ == "__main__":
    main()
