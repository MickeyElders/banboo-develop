import json
import shutil
import subprocess
from typing import Dict, Any


def _run(cmd, timeout=10):
    """Run a command and return (rc, out, err)."""
    try:
        p = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as e:
        return 1, "", str(e)


def reboot():
    return _run(["systemctl", "reboot"], timeout=2)


def shutdown():
    return _run(["systemctl", "poweroff"], timeout=2)


def restart_service(service="bamboo-vision.service"):
    return _run(["systemctl", "restart", service], timeout=5)


def has_nmcli() -> bool:
    return shutil.which("nmcli") is not None


def wifi_status() -> Dict[str, Any]:
    if not has_nmcli():
        return {"ok": False, "error": "nmcli not found"}
    rc, out, err = _run(["nmcli", "-t", "-f", "DEVICE,STATE,CONNECTION", "dev", "status"])
    active = {}
    if rc == 0:
        for line in out.splitlines():
            parts = line.split(":")
            if len(parts) >= 3 and parts[1] == "connected":
                active = {"device": parts[0], "connection": parts[2]}
                break
    rc2, out2, err2 = _run(["nmcli", "-t", "-f", "ACTIVE,SSID,SIGNAL,SECURITY", "dev", "wifi"])
    wifi = {}
    if rc2 == 0:
        for line in out2.splitlines():
            parts = line.split(":")
            if len(parts) >= 4 and parts[0] == "yes":
                wifi = {"ssid": parts[1], "signal": parts[2], "security": parts[3]}
                break
    conn = active.get("connection")
    ip_info = {}
    if conn:
        rc3, out3, err3 = _run(["nmcli", "-t", "-f", "ipv4.address,ipv4.gateway,ipv4.dns,ipv4.method", "connection", "show", conn])
        if rc3 == 0:
            for line in out3.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    ip_info[k] = v
    return {
        "ok": True,
        "device": active.get("device"),
        "connection": conn,
        "wifi": wifi,
        "ip": ip_info.get("ipv4.address", ""),
        "gateway": ip_info.get("ipv4.gateway", ""),
        "dns": ip_info.get("ipv4.dns", ""),
        "method": ip_info.get("ipv4.method", ""),
    }


def wifi_apply(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    cfg: {ssid, password, mode('dhcp'|'static'), ip, mask, gateway, dns, iface(optional)}
    """
    if not has_nmcli():
        return {"ok": False, "error": "nmcli not found"}
    ssid = cfg.get("ssid", "")
    password = cfg.get("password", "")
    iface = cfg.get("iface")
    mode = cfg.get("mode", "dhcp")
    conn_name = cfg.get("connection", "bamboo-wifi")
    if not ssid:
        return {"ok": False, "error": "ssid required"}
    # create or modify connection
    args = ["nmcli", "dev", "wifi", "connect", ssid, "name", conn_name]
    if password:
        args += ["password", password]
    if iface:
        args += ["ifname", iface]
    rc, out, err = _run(args, timeout=15)
    if rc != 0:
        return {"ok": False, "error": err or out}
    if mode == "static":
        ip = cfg.get("ip", "")
        mask = cfg.get("mask", "")
        gateway = cfg.get("gateway", "")
        dns = cfg.get("dns", "")
        addr = ip + ("/" + mask if mask else "")
        _run(["nmcli", "con", "mod", conn_name, "ipv4.addresses", addr])
        if gateway:
            _run(["nmcli", "con", "mod", conn_name, "ipv4.gateway", gateway])
        if dns:
            _run(["nmcli", "con", "mod", conn_name, "ipv4.dns", dns])
        _run(["nmcli", "con", "mod", conn_name, "ipv4.method", "manual"])
    else:
        _run(["nmcli", "con", "mod", conn_name, "ipv4.method", "auto"])
    _run(["nmcli", "con", "up", conn_name])
    status = wifi_status()
    status["ok"] = True
    status["message"] = "Wi-Fi applied"
    return status


def wifi_restart() -> Dict[str, Any]:
    if not has_nmcli():
        return {"ok": False, "error": "nmcli not found"}
    _run(["nmcli", "networking", "off"])
    _run(["nmcli", "networking", "on"])
    return {"ok": True, "message": "Networking restarted"}
