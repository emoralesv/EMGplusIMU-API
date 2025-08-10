
from __future__ import print_function, division, unicode_literals, absolute_import
import struct

class SerialCoder():
    
    @staticmethod
    def read_s08(s):
        return struct.unpack('<b', bytearray(s.read(1)))[0]

    @staticmethod
    def read_s16(s):
        return struct.unpack('<h', bytearray(s.read(2)))[0]

    @staticmethod
    def read_32(s):
        return struct.unpack('<l', bytearray(s.read(4)))[0]

    @staticmethod
    def read_u08(s):
        return struct.unpack('<B', bytearray(s.read(1)))[0]

    @staticmethod
    def read_u16(s):
        return struct.unpack('<H', bytearray(s.read(2)))[0]

    @staticmethod
    def read_u32(s):
        return struct.unpack('<L', bytearray(s.read(4)))[0]

    @staticmethod
    def write_s08(s, value):
        if -128 <= value <= 127:
            s.write(struct.pack('<b', value))
        else:
            print("Value error:{}".format(value))
    
    @staticmethod
    def write_s16(s, value):
        s.write(struct.pack('<h', value))

    @staticmethod
    def write_s32(s, value):
        s.write(struct.pack('<l', value))

    @staticmethod
    def write_u08(s, value):
        s.write(struct.pack('<B', value))

    @staticmethod
    def write_u16(s, value):
        s.write(struct.pack('<H', value))

    @staticmethod
    def write_u32(s, value):
        s.write(struct.pack('<L', value))
