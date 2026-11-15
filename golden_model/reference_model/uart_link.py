"""
uart_link.py - stage 6: FPGA -> laptop over the Nexys4 DDR's USB-UART (rtl/uart_tx.sv)

2 000 000 baud, 8N1 (10 bits per byte). One packet per processed camera frame:
    A5 5A 46 50 | version | flags | camera no | ToF no | U-Net cycles | post cycles | n_obj
    | n_obj x 12-byte objects | 64 x u16 ToF distance | 64 x u8 ToF status
    | 4 800-byte object map (80 x 60) | [19 200-byte camera image, switch SW0 on] | sum16
Full definition: fpga_model.uart_packet / parse_uart_packet.
"""
from . import fpga_model as fm

BAUD = 2_000_000
BITS_PER_BYTE = 10


def build_packet(cam_no, tof_no, tof_ok, infer_cycles, post_cycles, post, dist, status, frame=None):
    return fm.uart_packet(cam_no, tof_no, tof_ok, infer_cycles, post_cycles, post.objects,
                          post.object_map.ravel(), dist, status, frame, flags_extra=4 * int(post.overflow))


def parse_packet(buf):
    pkt, n = fm.parse_uart_packet(buf)
    if pkt is None:
        raise ValueError(f"incomplete packet: {len(buf)} of {n} bytes")
    return pkt, n


def transfer_time_s(n_bytes):
    return n_bytes * BITS_PER_BYTE / BAUD


def packet_bytes(n_obj, with_image):
    return fm.UART_HDR + n_obj * fm.OBJ_BYTES + 192 + fm.NPIX + (fm.IMG_W * fm.IMG_H if with_image else 0) + 2
