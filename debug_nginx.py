import paramiko; from paramiko import SSHClient, AutoAddPolicy; import sys
HOST, USER, PASSWORD = "43.157.201.110", "ubuntu", "panda-86$-thunder"
def run(c, cmd):
    _, o, e = c.exec_command(cmd, timeout=30)
    print(o.read().decode(errors="replace").strip() or e.read().decode(errors="replace").strip())
client = SSHClient(); client.set_missing_host_key_policy(AutoAddPolicy())
client.connect(HOST, username=USER, password=PASSWORD, timeout=30)

print("=== carwash-api config (port 80?) ===")
run(client, "sudo cat /etc/nginx/sites-available/carwash-api | head -30")

print("\n=== carwash.brevonsolutions.com config (port 80?) ===")
run(client, "sudo cat '/etc/nginx/sites-available/carwash.brevonsolutions.com' | head -20")

print("\n=== What's actually bound on port 80 ===")
run(client, "sudo ss -tlnp | grep ':80'")

print("\n=== Curl test to IP port 80 with verbose ===")
run(client, "curl -v -s -o /dev/null http://43.157.201.110/ 2>&1 | head -20")

print("\n=== Curl test to localhost port 80 ===")
run(client, "curl -v -s -o /dev/null http://127.0.0.1/ 2>&1 | head -20")

client.close()
