"""Fix incomplete deployment: wait for apt lock, create venv, install service."""
import time
import sys
import paramiko
from paramiko import SSHClient, AutoAddPolicy

HOST     = "43.157.201.110"
USER     = "ubuntu"
PASSWORD = "panda-86$-thunder"
APP_DIR  = "/var/www/aigym"

SYSTEMD_SERVICE = """[Unit]
Description=AIGym FastAPI Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/var/www/aigym/backend
Environment=MODEL_PATH=yolo11s-pose.pt
Environment=NUM_THREADS=4
ExecStart=/var/www/aigym/backend/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""

def run(client, cmd, desc="", ok_to_fail=False):
    if desc:
        print(f"\n  >> {desc}")
    _, stdout, stderr = client.exec_command(cmd, get_pty=False, timeout=300)
    out = stdout.read().decode(errors="replace").strip()
    err = stderr.read().decode(errors="replace").strip()
    code = stdout.channel.recv_exit_status()
    for line in (out + "\n" + err).strip().splitlines()[-10:]:
        if line.strip():
            print(f"     {line}")
    if code != 0 and not ok_to_fail:
        print(f"  [!] Command exited {code}")
    return code

def main():
    print(f"\n{'='*55}")
    print(f"  AIGym Fix Deployment -> {HOST}")
    print(f"{'='*55}\n")

    client = SSHClient()
    client.set_missing_host_key_policy(AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=30)
    print("Connected.\n")

    # Wait for apt lock to clear
    print("[1] Waiting for apt lock to clear...")
    for attempt in range(20):
        code = run(client,
            "sudo flock -n /var/lib/dpkg/lock-frontend echo free 2>/dev/null && echo LOCK_FREE || echo LOCKED",
            ok_to_fail=True)
        _, stdout, _ = client.exec_command(
            "sudo flock -n /var/lib/dpkg/lock-frontend echo LOCK_FREE 2>/dev/null || echo LOCKED")
        result = stdout.read().decode().strip()
        if "LOCK_FREE" in result:
            print("     apt lock is free.")
            break
        print(f"     apt locked, waiting 10s (attempt {attempt+1}/20)...")
        time.sleep(10)

    # Install required system packages
    run(client,
        "sudo apt-get install -y -qq nginx python3-venv python3-pip libgl1 libglib2.0-0",
        "Install nginx + python3-venv + libgl1")

    # Create Python venv and install deps
    run(client,
        f"cd {APP_DIR}/backend && python3 -m venv venv",
        "Create Python venv")
    run(client,
        f"cd {APP_DIR}/backend && venv/bin/pip install --quiet --upgrade pip",
        "Upgrade pip")
    run(client,
        f"cd {APP_DIR}/backend && venv/bin/pip install --quiet -r requirements.txt",
        "Install requirements (may take 2-3 mins for ultralytics/torch)")

    # Write systemd service
    run(client,
        f"cat > /tmp/aigym.service << 'EOF'\n{SYSTEMD_SERVICE}\nEOF",
        "Write systemd service")
    run(client,
        "sudo cp /tmp/aigym.service /etc/systemd/system/aigym.service && "
        "sudo systemctl daemon-reload && sudo systemctl enable aigym",
        "Install systemd service")
    run(client,
        "sudo systemctl restart aigym && sleep 3 && sudo systemctl status aigym --no-pager",
        "Start aigym service")

    # Fix nginx — write site config
    nginx_conf = r"""server {
    listen 80 default_server;
    root /var/www/aigym/frontend;
    index index.html;
    location / { try_files $uri $uri/ /index.html; }
    location /ws/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 3600;
    }
    location ~ ^/(health|exercises) { proxy_pass http://127.0.0.1:8000; }
}"""
    run(client,
        f"echo '{nginx_conf}' | sudo tee /etc/nginx/sites-available/aigym > /dev/null && "
        "sudo ln -sf /etc/nginx/sites-available/aigym /etc/nginx/sites-enabled/aigym && "
        "sudo rm -f /etc/nginx/sites-enabled/default",
        "Write nginx config (HTTP only — add SSL cert separately)")
    run(client, "sudo nginx -t && sudo systemctl restart nginx", "Test + restart nginx")

    # Verify
    run(client, "sudo systemctl status aigym --no-pager -l", "Final aigym status")
    run(client, "curl -s http://127.0.0.1:8000/health", "Health check")

    print(f"\n{'='*55}")
    print("  Fix complete!\n")
    print("  Site (HTTP):  http://43.157.201.110")
    print("  Backend WS:   ws://43.157.201.110/ws/<exercise>/<foot>")
    print()
    print("  DNS record to add:")
    print("    A   <your-subdomain>   43.157.201.110   TTL 300")
    print()
    print("  For HTTPS, upload your brevonsolutions cert files:")
    print("    scp cert.pem ubuntu@43.157.201.110:/tmp/cert.pem")
    print("    scp key.pem  ubuntu@43.157.201.110:/tmp/key.pem")
    print("    Then tell me and I will update the nginx config for SSL.")
    print(f"{'='*55}\n")

    client.close()

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
