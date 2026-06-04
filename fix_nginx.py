"""Add HTTP fallback so the site is accessible by IP on port 80."""
import paramiko
from paramiko import SSHClient, AutoAddPolicy
import sys

HOST, USER, PASSWORD = "43.157.201.110", "ubuntu", "panda-86$-thunder"
DOMAIN = "aigym.brevonsolutions.com"

# Two server blocks:
# 1. HTTP default_server (IP access) — serves site directly, no redirect
# 2. HTTPS for the real domain — uses Let's Encrypt cert
NGINX_CONF = f"""# HTTP — serves by IP directly (no SSL redirect)
server {{
    listen 80 default_server;
    server_name _;
    root /var/www/aigym/frontend;
    index index.html;

    location / {{
        try_files $uri $uri/ /index.html;
    }}
    location /ws/ {{
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_read_timeout 3600;
    }}
    location ~ ^/(health|exercises) {{
        proxy_pass http://127.0.0.1:8000;
    }}
}}

# HTTPS — serves by domain with Let's Encrypt cert
server {{
    listen 443 ssl;
    server_name {DOMAIN};

    ssl_certificate     /etc/letsencrypt/live/brevonsolutions.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/brevonsolutions.com/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;

    root /var/www/aigym/frontend;
    index index.html;

    location / {{
        try_files $uri $uri/ /index.html;
    }}
    location /ws/ {{
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_read_timeout 3600;
    }}
    location ~ ^/(health|exercises) {{
        proxy_pass http://127.0.0.1:8000;
    }}
}}
"""

def run(client, cmd, desc=""):
    if desc:
        print(f"  >> {desc}")
    _, stdout, stderr = client.exec_command(cmd, timeout=30)
    out = stdout.read().decode(errors="replace").strip()
    err = stderr.read().decode(errors="replace").strip()
    code = stdout.channel.recv_exit_status()
    for line in (out + "\n" + err).strip().splitlines()[-6:]:
        if line.strip():
            print(f"     {line}")
    return code

def main():
    client = SSHClient()
    client.set_missing_host_key_policy(AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=30)
    print("Connected.\n")

    # Write the new config
    run(client,
        f"sudo tee /etc/nginx/sites-available/aigym > /dev/null << 'NGINXEOF'\n{NGINX_CONF}\nNGINXEOF",
        "Write nginx config")

    run(client,
        "sudo ln -sf /etc/nginx/sites-available/aigym /etc/nginx/sites-enabled/aigym",
        "Enable site")

    code = run(client, "sudo nginx -t", "Test config")
    if code != 0:
        print("\n  [!] nginx config test failed — check errors above")
        client.close()
        return

    run(client, "sudo systemctl reload nginx", "Reload nginx")
    run(client, "curl -s -o /dev/null -w '%{http_code}' http://localhost/", "Test HTTP locally")

    print(f"""
{'='*55}
  Done!

  By IP (works now):
    http://43.157.201.110        <- site
    ws://43.157.201.110/ws/...   <- WebSocket

  By domain (works after DNS):
    https://{DOMAIN}
    wss://{DOMAIN}/ws/...

  DNS record still needed:
    A   aigym   43.157.201.110   TTL 300
{'='*55}
""")
    client.close()

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
