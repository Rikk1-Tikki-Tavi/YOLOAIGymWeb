import paramiko; from paramiko import SSHClient, AutoAddPolicy; import sys
HOST, USER, PASSWORD = "43.157.201.110", "ubuntu", "panda-86$-thunder"
def run(c, cmd):
    _, o, e = c.exec_command(cmd, timeout=30)
    out = o.read().decode(errors="replace").strip()
    print(out or e.read().decode(errors="replace").strip())
client = SSHClient(); client.set_missing_host_key_policy(AutoAddPolicy())
client.connect(HOST, username=USER, password=PASSWORD, timeout=30)
print("=== sites-enabled ==="); run(client, "ls -la /etc/nginx/sites-enabled/")
print("\n=== port 80 listeners (which has default_server) ===")
run(client, "sudo grep -r 'listen 80' /etc/nginx/sites-enabled/")
print("\n=== aigym config ==="); run(client, "cat /etc/nginx/sites-available/aigym")
client.close()
