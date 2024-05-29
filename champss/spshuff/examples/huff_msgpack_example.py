#! /usr/bin/env python
from spshuff.huff_utils import convert_msgpack_to_huff, read_huff_msgpack

# Chitrang's example msgpack/IntensityFile conversion routines

if __name__ == "__main__":
    filename = "/data/frb-archiver/2020/04/27/astro_82243100/intensity/raw/0214/astro_82243100_20200427010452418153_beam0214_00100276_01.msgpack"
    new_file = convert_msgpack_to_huff(filename)
    header_info, intensities, rfi_mask, means, variance = read_huff_msgpack(new_file)
    print("\n\nOutput: ")
    print(header_info)
    print(intensities, rfi_mask, means, variance)
