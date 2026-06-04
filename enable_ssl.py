"""Update nginx to use the existing brevonsolutions.com Let's Encrypt cert."""
import paramiko
from paramiko import SSHClient, AutoAddPolicy
import sys

HOST, USER, PASSWORD = "43.157.201.110", "ubuntu", "panda-86$-thunder"
DOMAIN = "aigym.brevonsolutions.com"

NGINX_CONF = f"""server {{
    listen 80;
    server_name {DOMAIN};
    return 301 https://$host$request_uri;
}}

server {{
    listen 443 ssl;
    server_name {DOMAIN};

    ssl_certificate     /etc/letsencrypt/live/brevonsolutions.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/brevonsolutions.com/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

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
        proxy_set_header Host $host;
    }}
}}
"""

def run(client, cmd, desc=""):
    if desc:
        print(f"\n  >> {desc}")
    _, stdout, stderr = client.exec_command(cmd, timeout=60)
    out = stdout.read().decode(errors="replace").strip()
    err = stderr.read().decode(errors="replace").strip()
    code = stdout.channel.recv_exit_status()
    combined = "\n".join(filter(None, [out, err]))
    for line in combined.splitlines()[-8:]:
        if line.strip():
            print(f"     {line}")
    return code

def main():
    client = SSHClient()
    client.set_missing_host_key_policy(AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=30)
    print("Connected.\n")

    # Write nginx config via heredoc
    escaped = NGINX_CONF.replace("'", "'\\''")
    run(client,
        f"sudo tee /etc/nginx/sites-available/aigym > /dev/null << 'NGINXEOF'\n{NGINX_CONF}\nNGINXEOF",
        f"Write nginx config for {DOMAIN}")

    run(client,
        "sudo ln -sf /etc/nginx/sites-available/aigym /etc/nginx/sites-enabled/aigym",
        "Enable site")

    run(client, "sudo nginx -t", "Test nginx config")
    run(client, "sudo systemctl reload nginx", "Reload nginx")
    run(client, "sudo systemctl status aigym --no-pager | head -5", "aigym service status")

    print(f"""
{'='*55}
  SSL enabled!

  URL : https://{DOMAIN}
  WSS : wss://{DOMAIN}/ws/<exercise>/<foot>

  DNS record to add:
    Type : A
    Name : aigym
    Value: 43.157.201.110
    TTL  : 300

  Add that record in your DNS provider for brevonsolutions.com,
  then the site will be live at https://{DOMAIN}
{'='*55}
""")
    client.close()

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
