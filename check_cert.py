import paramiko; from paramiko import SSHClient, AutoAddPolicy; import sys
HOST, USER, PASSWORD = "43.157.201.110", "ubuntu", "panda-86$-thunder"
def run(c, cmd, title=""):
    if title: print(f"\n=== {title} ===")
    _, o, e = c.exec_command(cmd, timeout=30)
    print(o.read().decode(errors="replace").strip() or e.read().decode(errors="replace").strip())
client = SSHClient(); client.set_missing_host_key_policy(AutoAddPolicy())
client.connect(HOST, username=USER, password=PASSWORD, timeout=30)
run(client,
    "sudo openssl x509 -in /etc/letsencrypt/live/brevonsolutions.com/cert.pem -noout -text "
    "| grep -A5 'Subject Alternative'",
    "Cert SAN — domains it covers")
run(client, "sudo openssl x509 -in /etc/letsencrypt/live/brevonsolutions.com/cert.pem -noout -dates", "Cert expiry")
client.close()
