"""
Standalone MPU6050 test: connects over I2C and prints accelerometer and
gyroscope readings to the terminal. Ctrl+C to quit.

    python mpu6050_test.py                 # bus 1, address 0x68, 10 Hz
    python mpu6050_test.py --hz 50 --address 0x69
"""

import argparse
import struct
import sys
import time

from smbus2 import SMBus

PWR_MGMT_1 = 0x6B
ACCEL_CONFIG = 0x1C
GYRO_CONFIG = 0x1B
ACCEL_XOUT_H = 0x3B  # accel(6) + temp(2) + gyro(6), read in one burst
WHO_AM_I = 0x75

ACCEL_LSB_PER_G = 16384.0  # ±2 g range
GYRO_LSB_PER_DPS = 131.0   # ±250 °/s range


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--bus", type=int, default=1, help="I2C bus number (default 1)")
    p.add_argument("--address", type=lambda s: int(s, 0), default=0x68,
                   help="I2C address (default 0x68; 0x69 if AD0 is high)")
    p.add_argument("--hz", type=float, default=10.0, help="print rate (default 10)")
    args = p.parse_args()

    try:
        bus = SMBus(args.bus)
    except FileNotFoundError:
        print(f"/dev/i2c-{args.bus} not found. Enable I2C: sudo raspi-config "
              "-> Interface Options -> I2C, then reboot.", file=sys.stderr)
        return 1

    with bus:
        try:
            who = bus.read_byte_data(args.address, WHO_AM_I)
            bus.write_byte_data(args.address, PWR_MGMT_1, 0x00)  # wake up
            bus.write_byte_data(args.address, ACCEL_CONFIG, 0x00)  # ±2 g
            bus.write_byte_data(args.address, GYRO_CONFIG, 0x00)   # ±250 °/s
            time.sleep(0.1)  # let the first conversion finish after wake-up
        except OSError as e:
            print(f"No response from 0x{args.address:02x} on bus {args.bus}: {e}\n"
                  "Check wiring (VCC, GND, SDA->GPIO2/pin 3, SCL->GPIO3/pin 5) "
                  f"and run: i2cdetect -y {args.bus}", file=sys.stderr)
            return 1

        print(f"Connected: bus {args.bus}, address 0x{args.address:02x}, "
              f"WHO_AM_I=0x{who:02x}. Ctrl+C to quit.\n")
        print(f"{'ax (g)':>8} {'ay (g)':>8} {'az (g)':>8}   "
              f"{'gx (°/s)':>9} {'gy (°/s)':>9} {'gz (°/s)':>9}")

        period = 1.0 / args.hz
        try:
            while True:
                try:
                    raw = bytes(bus.read_i2c_block_data(args.address, ACCEL_XOUT_H, 14))
                except OSError as e:
                    print(f"read error: {e}", file=sys.stderr)
                    time.sleep(period)
                    continue
                ax, ay, az, _temp, gx, gy, gz = struct.unpack(">7h", raw)
                print(f"{ax / ACCEL_LSB_PER_G:8.3f} {ay / ACCEL_LSB_PER_G:8.3f} "
                      f"{az / ACCEL_LSB_PER_G:8.3f}   "
                      f"{gx / GYRO_LSB_PER_DPS:9.2f} {gy / GYRO_LSB_PER_DPS:9.2f} "
                      f"{gz / GYRO_LSB_PER_DPS:9.2f}", flush=True)
                time.sleep(period)
        except KeyboardInterrupt:
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
