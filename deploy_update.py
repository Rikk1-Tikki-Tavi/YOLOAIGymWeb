"""Push updated frontend + backend to VPS and restart PM2."""
import time, sys, pathlib
import paramiko
from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient

HOST, USER, PASSWORD = "43.157.201.110", "ubuntu", "panda-86$-thunder"
APP_DIR = "/var/www/aigym"
ROOT    = pathlib.Path(__file__).parent

def run(c, cmd, desc=""):
    if desc: print(f"  >> {desc}")
    _, o, e = c.exec_command(cmd, timeout=120)
    out = o.read().decode(errors="replace").strip()
    err = e.read().decode(errors="replace").strip()
    code = o.channel.recv_exit_status()
    for line in (out + "\n" + err).strip().splitlines()[-6:]:
        if line.strip(): print(f"     {line}")
    return code

def main():
    client = SSHClient()
    client.set_missing_host_key_policy(AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=30)
    print("Connected.\n")

    # Upload frontend dist
    print("[1/4] Uploading frontend dist/...")
    FRONTEND = ROOT / "frontend" / "dist"
    with SCPClient(client.get_transport()) as scp:
        for item in FRONTEND.iterdir():
            dest = f"{APP_DIR}/frontend"
            if item.is_dir():
                scp.put(str(item), remote_path=dest, recursive=True)
            else:
                scp.put(str(item), remote_path=f"{dest}/{item.name}")
    print("     Frontend uploaded.\n")

    # Upload backend (exclude venv, cache)
    print("[2/4] Uploading backend files...")
    SKIP = {"venv", "__pycache__", ".pyc"}
    BACKEND = ROOT / "backend"
    with SCPClient(client.get_transport()) as scp:
        for item in BACKEND.iterdir():
            if item.name in SKIP or item.suffix == ".pyc": continue
            dest = f"{APP_DIR}/backend"
            if item.is_dir():
                scp.put(str(item), remote_path=dest, recursive=True)
            else:
                scp.put(str(item), remote_path=f"{dest}/{item.name}")
    print("     Backend uploaded.\n")

    # Restart PM2
    print("[3/4] Restarting PM2 aigym process...")
    run(client, "pm2 restart aigym && sleep 2 && pm2 show aigym | grep -E 'status|uptime|pid'",
        "pm2 restart")

    # Verify
    print("\n[4/4] Verifying...")
    run(client, "pm2 list | grep aigym", "pm2 status")
    print()
    print("="*50)
    print("  Deploy complete!")
    print("  http://43.157.201.110")
    print("  https://aigym.brevonsolutions.com  (after DNS)")
    print("="*50)
    client.close()

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
