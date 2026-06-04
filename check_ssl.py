import paramiko; from paramiko import SSHClient, AutoAddPolicy; import sys
HOST, USER, PASSWORD = "43.157.201.110", "ubuntu", "panda-86$-thunder"
def run(c, cmd, title=""):
    if title: print(f"\n=== {title} ===")
    _, o, e = c.exec_command(cmd, timeout=30)
    print(o.read().decode(errors="replace").strip() or e.read().decode(errors="replace").strip())
client = SSHClient(); client.set_missing_host_key_policy(AutoAddPolicy())
client.connect(HOST, username=USER, password=PASSWORD, timeout=30)

run(client, "sudo ss -tlnp | grep -E ':80|:443'", "Ports 80 and 443 listening")
run(client, "sudo ufw status", "UFW firewall status")
run(client, "curl -vk -s -o /dev/null https://127.0.0.1/ 2>&1 | head -20", "HTTPS curl to localhost")
run(client, "sudo nginx -T 2>/dev/null | grep -A3 'listen 443'", "nginx listen 443 blocks")
client.close()
