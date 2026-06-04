"""Find existing SSL certs on the VPS."""
import paramiko
from paramiko import SSHClient, AutoAddPolicy
import sys

HOST, USER, PASSWORD = "43.157.201.110", "ubuntu", "panda-86$-thunder"

def run(client, cmd):
    _, stdout, stderr = client.exec_command(cmd, timeout=30)
    out = stdout.read().decode(errors="replace").strip()
    err = stderr.read().decode(errors="replace").strip()
    return out, err

client = SSHClient()
client.set_missing_host_key_policy(AutoAddPolicy())
client.connect(HOST, username=USER, password=PASSWORD, timeout=30)
print("Connected.\n")

print("=== Searching for cert/key files ===")
out, _ = run(client, "sudo find /etc/ssl /etc/nginx /etc/letsencrypt /home -name '*.pem' -o -name '*.crt' -o -name '*.key' -o -name '*.cer' 2>/dev/null | sort")
print(out or "(none found in those paths)")

print("\n=== /etc/ssl contents ===")
out, _ = run(client, "sudo find /etc/ssl -type f 2>/dev/null")
print(out or "(empty)")

print("\n=== /etc/nginx ssl references ===")
out, _ = run(client, "sudo grep -r ssl_certificate /etc/nginx/ 2>/dev/null")
print(out or "(no ssl_certificate directives)")

print("\n=== Let's Encrypt ===")
out, _ = run(client, "sudo ls /etc/letsencrypt/live/ 2>/dev/null")
print("Live certs:", out or "(none)")

print("\n=== snap/certbot ===")
out, _ = run(client, "which certbot 2>/dev/null || snap list certbot 2>/dev/null || echo 'certbot not installed'")
print(out)

client.close()
