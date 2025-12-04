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


def has_connmanctl() -> bool:
    return shutil.which("connmanctl") is not None


def _connman_wifi_status() -> Dict[str, Any]:
    rc, out, err = _run(["connmanctl", "services"])
    if rc != 0:
        return {"ok": False, "error": err or "connmanctl services failed"}
    active_line = ""
    for line in out.splitlines():
        if line.strip().startswith("*"):
            active_line = line.strip()
            break
    if not active_line:
        return {"ok": False, "error": "wifi not connected (connmanctl)"}
    parts = active_line.split()
    service_id = parts[1] if len(parts) > 1 else ""
    ssid = parts[-1] if len(parts) >= 3 else ""
    ip = gateway = dns = ""
    if service_id:
        rc2, out2, _ = _run(["connmanctl", "services", service_id])
        if rc2 == 0:
            for l in out2.splitlines():
                if "IPv4 =" in l:
                    # Format: IPv4 = <ip>/<mask>/<gateway>
                    seg = l.split("=", 1)[1].strip()
                    parts_ipv4 = seg.split("/")
                    if len(parts_ipv4) >= 1:
                        ip = parts_ipv4[0]
                    if len(parts_ipv4) >= 2:
                        gateway = parts_ipv4[2] if len(parts_ipv4) >= 3 else ""
                if "IPv4.DNS" in l:
                    dns = l.split("=", 1)[1].strip()
    return {
        "ok": True,
        "device": service_id,
        "connection": ssid,
        "wifi": {"ssid": ssid, "signal": "", "security": ""},
        "ip": ip,
        "gateway": gateway,
        "dns": dns,
        "method": "",
    }


def wifi_status() -> Dict[str, Any]:
    if has_nmcli():
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
    if has_connmanctl():
        return _connman_wifi_status()
    return {"ok": False, "error": "nmcli/connmanctl not found"}


def wifi_apply(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    cfg: {ssid, password, mode('dhcp'|'static'), ip, mask, gateway, dns, iface(optional)}
    """
    if not has_nmcli() and not has_connmanctl():
        return {"ok": False, "error": "nmcli/connmanctl not found"}
    ssid = cfg.get("ssid", "")
    password = cfg.get("password", "")
    iface = cfg.get("iface")
    mode = cfg.get("mode", "dhcp")
    conn_name = cfg.get("connection", "bamboo-wifi")
    if not ssid:
        return {"ok": False, "error": "ssid required"}
    if has_connmanctl() and not has_nmcli():
        # Lightweight connmanctl workflow
        _run(["connmanctl", "enable", "wifi"])
        rc, out, err = _run(["connmanctl", "services"])
        if rc != 0:
            return {"ok": False, "error": err or "connmanctl services failed"}
        service_id = ""
        for line in out.splitlines():
            if ssid in line:
                parts = line.strip().split()
                if len(parts) >= 2:
                    service_id = parts[1]
                    break
        if not service_id:
            return {"ok": False, "error": f"SSID {ssid} not found via connmanctl"}
        if password:
            _run(["connmanctl", "config", service_id, f"--passphrase={password}"])
        if mode == "static":
            ip = cfg.get("ip", "")
            mask = cfg.get("mask", "")
            gateway = cfg.get("gateway", "")
            _run(["connmanctl", "config", service_id, "--ipv4", "manual", ip, mask, gateway])
        else:
            _run(["connmanctl", "config", service_id, "--ipv4", "dhcp"])
        rc_conn, out_conn, err_conn = _run(["connmanctl", "connect", service_id], timeout=20)
        if rc_conn != 0:
            return {"ok": False, "error": err_conn or out_conn or "connmanctl connect failed"}
        status = _connman_wifi_status()
        status["message"] = "Wi-Fi applied via connmanctl"
        return status

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
    if has_nmcli():
        _run(["nmcli", "networking", "off"])
        _run(["nmcli", "networking", "on"])
        return {"ok": True, "message": "Networking restarted"}
    if has_connmanctl():
        _run(["connmanctl", "disable", "wifi"])
        _run(["connmanctl", "enable", "wifi"])
        return {"ok": True, "message": "Wi-Fi restarted via connmanctl"}
    return {"ok": False, "error": "nmcli/connmanctl not found"}
