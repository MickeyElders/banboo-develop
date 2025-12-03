import logging
import struct
import time

from pymodbus.client import ModbusTcpClient

from .shared_state import SharedState


def float_to_regs_be(value: float) -> tuple[int, int]:
    packed = struct.pack(">f", float(value))
    high = int.from_bytes(packed[0:2], byteorder="big", signed=False)
    low = int.from_bytes(packed[2:4], byteorder="big", signed=False)
    return high, low


def regs_to_float_be(high: int, low: int) -> float:
    packed = high.to_bytes(2, "big") + low.to_bytes(2, "big")
    return struct.unpack(">f", packed)[0]


class ModbusBridge:
    """Lightweight Modbus TCP client matching PLC.md map."""

    def __init__(self, cfg: dict, state: SharedState):
        mcfg = cfg.get("modbus", {})
        self.host = mcfg.get("host", "127.0.0.1")
        self.port = int(mcfg.get("port", 502))
        self.slave_id = int(mcfg.get("slave_id", 1))
        self.poll_ms = int(mcfg.get("poll_ms", 50))
        self.hb_ms = int(mcfg.get("write_heartbeat_ms", 20))
        self.addr_cam = mcfg.get("addr_cam_to_plc", {})
        self.addr_plc = mcfg.get("addr_plc_to_cam", {})
        self.client = ModbusTcpClient(host=self.host, port=self.port, unit_id=self.slave_id, timeout=1)
        self.connected = False
        self.last_poll = 0.0
        self.last_hb = 0.0
        self.plc_ready = False
        self.plc_state = 0
        self.plc_pos = 0.0
        self.hb_local = 0
        self.state = state

    def ensure_connected(self) -> bool:
        if self.connected and self.client.connected:
            return True
        self.connected = self.client.connect()
        if not self.connected:
            logging.warning("Modbus connect failed %s:%s", self.host, self.port)
        return self.connected

    def close(self):
        try:
            self.client.close()
        except Exception:
            pass
        self.connected = False

    def step(self, now: float):
        if not self.ensure_connected():
            return

        # Heartbeat/communication ack
        if now - self.last_hb >= self.hb_ms / 1000.0:
            ack_addr = self.addr_cam.get("comm", 0x07D0)
            status_addr = self.addr_cam.get("status", 0x07D1)
            self.client.write_register(address=ack_addr, value=1, slave=self.slave_id)
            self.client.write_register(address=status_addr, value=1, slave=self.slave_id)  # 1=normal
            self.last_hb = now
            self.hb_local = (self.hb_local + 1) & 0xFFFF

        # Poll PLC state/position
        if now - self.last_poll >= self.poll_ms / 1000.0:
            start = self.addr_plc.get("heartbeat", 0x0834)
            count = 4  # 0834..0837 covers heartbeat/state/pos(float)
            resp = self.client.read_holding_registers(address=start, count=count, slave=self.slave_id)
            if not resp.isError():
                hb_plc, state, pos_hi, pos_lo = resp.registers
                self.plc_state = state
                self.plc_pos = regs_to_float_be(pos_hi, pos_lo)
                self.plc_ready = state == 1  # 1=ready to receive coordinate
                self.state.update_plc(self.plc_state, self.plc_ready, self.plc_pos, self.hb_local)
            else:
                logging.warning("Modbus read error: %s", resp)
            self.last_poll = now

    def publish_detection(self, x_mm: float | None, result_code: int):
        """
        Write detection to PLC if connected.
        result_code: 1=success, 2=fail/no target (per PLC.md D2004)
        """
        if not self.ensure_connected():
            return
        if not self.plc_ready:
            logging.debug("PLC not ready (state=%s), skip publish", self.plc_state)
            return
        coord_addr = self.addr_cam.get("coord", 0x07D2)
        result_addr = self.addr_cam.get("result", 0x07D4)
        coord_val = x_mm if x_mm is not None else 0.0
        hi, lo = float_to_regs_be(coord_val)
        self.client.write_registers(address=coord_addr, values=[hi, lo], slave=self.slave_id)
        self.client.write_register(address=result_addr, value=result_code, slave=self.slave_id)
        logging.info("Published to PLC: x=%.2f mm result=%d", coord_val, result_code)
