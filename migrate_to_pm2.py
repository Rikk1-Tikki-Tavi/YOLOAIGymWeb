"""Stop systemd aigym service and re-register it under PM2."""
import paramiko; from paramiko import SSHClient, AutoAddPolicy; import sys

HOST, USER, PASSWORD = "43.157.201.110", "ubuntu", "panda-86$-thunder"

def run(c, cmd, title=""):
    if title: print(f"\n  >> {title}")
    _, o, e = c.exec_command(cmd, timeout=60)
    out = o.read().decode(errors="replace").strip()
    err = e.read().decode(errors="replace").strip()
    code = o.channel.recv_exit_status()
    for line in (out + "\n" + err).strip().splitlines()[-8:]:
        if line.strip(): print(f"     {line}")
    return code

def main():
    client = SSHClient()
    client.set_missing_host_key_policy(AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=30)
    print("Connected.\n")

    # 1. Stop and disable the systemd service
    run(client, "sudo systemctl stop aigym && sudo systemctl disable aigym",
        "Stop + disable systemd aigym.service")

    # 2. Check if PM2 is installed
    run(client, "which pm2 || npm list -g pm2 2>/dev/null | head -3",
        "Check PM2 location")

    # 3. Kill any existing PM2 aigym entry
    run(client, "pm2 delete aigym 2>/dev/null || true",
        "Remove old PM2 aigym entry (if any)")

    # 4. Start uvicorn under PM2
    run(client,
        "cd /var/www/aigym/backend && "
        "MODEL_PATH=yolo11s-pose.pt NUM_THREADS=4 "
        "pm2 start venv/bin/uvicorn "
        "--name aigym "
        "--interpreter none "
        "-- main:app --host 127.0.0.1 --port 8000 --workers 1",
        "Start aigym via PM2")

    # 5. Save PM2 process list so it survives reboots
    run(client, "pm2 save", "Save PM2 process list")

    # 6. Ensure PM2 startup hook is registered
    run(client, "pm2 startup systemd -u ubuntu --hp /home/ubuntu 2>&1 | tail -5",
        "PM2 startup hook")

    # 7. Show final PM2 list
    run(client, "pm2 list", "PM2 process list")

    # 8. Quick health check
    import time; time.sleep(3)
    run(client, "pm2 show aigym | grep -E 'status|uptime|memory|pid'",
        "aigym process details")

    print("\n" + "="*55)
    print("  Done! aigym is now managed by PM2.")
    print("  Your friend can see it with:  pm2 list")
    print("  Logs:                         pm2 logs aigym")
    print("  Restart:                      pm2 restart aigym")
    print("="*55 + "\n")
    client.close()

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
