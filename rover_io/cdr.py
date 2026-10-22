"""ROS2 CDR (DDS wire format) encode/decode for the rover's Zenoh topics.

Pure byte-level protocol code, no navigation logic -- vendored unchanged
from /home/gpu/Desktop/Nihil/LemNAV/nav_pipeline/zenoh_node.py (same
sensor_msgs/Image, sensor_msgs/CameraInfo, std_msgs/Float32MultiArray,
geometry_msgs/Twist wire contract the rover firmware/bringup already
speaks -- this has to match byte-for-byte, so it is not something to
"reimplement fresh"; the actual navigation/motion-control logic that uses
it lives in odometry.py and zenoh_motion.py instead, both written fresh
for Space2nav).
"""
from __future__ import annotations

import struct
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class CDRReader:
    def __init__(self, data: bytes):
        self.data = data
        self.le = data[1] in (0x01, 0x11)
        self.end = "<" if self.le else ">"
        self.offset = 4
        self.base = 4

    def _align(self, n: int):
        rem = (self.offset - self.base) % n
        if rem:
            self.offset += n - rem

    def read_uint8(self) -> int:
        v = self.data[self.offset]
        self.offset += 1
        return v

    def read_int32(self) -> int:
        self._align(4)
        (v,) = struct.unpack_from(self.end + "i", self.data, self.offset)
        self.offset += 4
        return v

    def read_uint32(self) -> int:
        self._align(4)
        (v,) = struct.unpack_from(self.end + "I", self.data, self.offset)
        self.offset += 4
        return v

    def read_float32(self) -> float:
        self._align(4)
        (v,) = struct.unpack_from(self.end + "f", self.data, self.offset)
        self.offset += 4
        return v

    def read_float64(self) -> float:
        self._align(8)
        (v,) = struct.unpack_from(self.end + "d", self.data, self.offset)
        self.offset += 8
        return v

    def read_string(self) -> str:
        length = self.read_uint32()
        s = self.data[self.offset: self.offset + length - 1].decode("utf-8", errors="replace")
        self.offset += length
        return s

    def read_sequence_uint8(self) -> bytes:
        count = self.read_uint32()
        data = self.data[self.offset: self.offset + count]
        self.offset += count
        return data


class CDRWriter:
    def __init__(self):
        self.buf = bytearray(b"\x00\x01\x00\x00")  # CDR LE encapsulation
        self.base = 4

    def _align(self, n: int):
        rem = (len(self.buf) - self.base) % n
        if rem:
            self.buf += b"\x00" * (n - rem)

    def write_int32(self, v: int):
        self._align(4)
        self.buf += struct.pack("<i", v)

    def write_uint32(self, v: int):
        self._align(4)
        self.buf += struct.pack("<I", v)

    def write_float64(self, v: float):
        self._align(8)
        self.buf += struct.pack("<d", v)

    def write_string(self, s: str):
        encoded = s.encode("utf-8") + b"\x00"
        self.write_uint32(len(encoded))
        self.buf += encoded

    def to_bytes(self) -> bytes:
        return bytes(self.buf)


def parse_image(cdr_data: bytes) -> Optional[np.ndarray]:
    """sensor_msgs/Image CDR -> RGB uint8 (H,W,3) or float32 depth (H,W)."""
    r = CDRReader(cdr_data)
    r.read_int32(); r.read_uint32(); r.read_string()  # header
    height = r.read_uint32()
    width = r.read_uint32()
    encoding = r.read_string()
    r.read_uint8(); r._align(4); r.read_uint32()  # is_bigendian, step
    pixel_data = r.read_sequence_uint8()

    enc = encoding.lower()
    if enc in ("32fc1", "depth32f"):
        img = np.frombuffer(pixel_data, dtype=np.float32)
        try:
            return img.reshape(height, width).copy()
        except ValueError:
            return None
    if enc == "16uc1":  # depth in mm
        img = np.frombuffer(pixel_data, dtype=np.uint16)
        try:
            return img.reshape(height, width).astype(np.float32) / 1000.0
        except ValueError:
            return None
    if enc == "png16":
        # PNG-compressed uint16 depth (mm), published at HALF color's pixel
        # dimensions on this rig (see LemNAV's
        # scripts/realsense_direct_publisher.py) -- height/width read above
        # are the encoded (halved) dims, not the final ones. Upsample back
        # to color's resolution with nearest-neighbor (no invented values
        # across real depth edges). 640x480 is this rig's fixed color
        # profile -- update both sides together if that ever changes.
        if cv2 is None:
            return None
        arr = np.frombuffer(pixel_data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.shape != (480, 640):
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_NEAREST)
        return img.astype(np.float32) / 1000.0

    img = np.frombuffer(pixel_data, dtype=np.uint8)
    try:
        img = img.reshape(height, width, -1)
    except ValueError:
        return None
    if enc == "bgr8":
        return img[:, :, :3][:, :, ::-1].copy()
    if img.shape[2] >= 3:
        return img[:, :, :3]
    return None


def parse_compressed_image(cdr_data: bytes) -> Optional[np.ndarray]:
    """sensor_msgs/CompressedImage CDR -> RGB uint8 (H,W,3)."""
    r = CDRReader(cdr_data)
    r.read_int32(); r.read_uint32(); r.read_string()  # header
    r.read_string()  # format
    jpeg_bytes = r.read_sequence_uint8()
    try:
        if cv2 is not None:
            arr = np.frombuffer(bytes(jpeg_bytes), dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        import io
        from PIL import Image
        return np.array(Image.open(io.BytesIO(bytes(jpeg_bytes))).convert("RGB"), dtype=np.uint8)
    except Exception:
        return None


def parse_camera_info(cdr_data: bytes) -> Optional[Tuple[float, float, float, float]]:
    """sensor_msgs/CameraInfo CDR -> (fx, fy, cx, cy) from the K[9] matrix."""
    r = CDRReader(cdr_data)
    r.read_int32(); r.read_uint32(); r.read_string()  # header
    r.read_uint32()  # height
    r.read_uint32()  # width
    r.read_string()  # distortion_model
    d_count = r.read_uint32()
    r.offset += d_count * 8
    try:
        k = [r.read_float64() for _ in range(9)]
    except (struct.error, IndexError):
        return None
    fx, cx, fy, cy = k[0], k[2], k[4], k[5]
    if fx <= 0 or fy <= 0:
        return None
    return fx, fy, cx, cy


def parse_float32_multiarray(cdr_data: bytes) -> List[float]:
    """std_msgs/Float32MultiArray CDR -> list[float] (the .data field).
    Rover's rpm topic layout (esp32/rover_6wd_complete.ino): [left_rpm,
    right_rpm, imu_heading?, imu_calib?, lateral_m_s?]."""
    r = CDRReader(cdr_data)
    dim_count = r.read_uint32()
    for _ in range(dim_count):
        r.read_string()
        r.read_uint32()
        r.read_uint32()
    r.read_uint32()  # data_offset
    n = r.read_uint32()
    return [r.read_float32() for _ in range(n)]


def serialize_twist(linear_x: float, angular_z: float) -> bytes:
    w = CDRWriter()
    w.write_float64(linear_x); w.write_float64(0.0); w.write_float64(0.0)
    w.write_float64(0.0); w.write_float64(0.0); w.write_float64(angular_z)
    return w.to_bytes()


def serialize_string(text: str) -> bytes:
    w = CDRWriter()
    w.write_string(text)
    return w.to_bytes()
