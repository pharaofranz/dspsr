import numpy as np


def dump_data():

    header = {
      "HDR_SIZE": "4096",
      "BW": "40",
      "DEC": "-45:59:09.5",
      "DSB": "0",
      "FREQ": "1405.000000",
      "HDR_VERSION": "1.0",
      "INSTRUMENT": "dspsr",
      "MODE": "PSR",
      "NBIT": "32",
      "NCHAN": "16",
      "NDIM": "1",
      "NPOL": "2",
      "OBS_OFFSET": "0",
      "PRIMARY": "dspsr",
      "RA": "16:44:49.28",
      "SOURCE": "J1644-4559",
      "TELESCOPE": "PKS",
      "TSAMP": "0.025",
      "UTC_START": "2019-08-29-04:42:20"
    }

    header_str = "\n".join(["{} {}".format(key, val)
                            for key, val in header.items()])
    header_bytes = str.encode(header_str)
    remaining_bytes = 4096 - len(header_bytes)
    header_bytes += str.encode(
        "".join(["\0" for i in range(remaining_bytes)]))

    arr = np.ones((16*2*100, ), dtype=np.float32) # this is numpy's ``float`` type.

    with open("test_data.dump", "wb") as fd:
        fd.write(header_bytes)
        fd.write(arr.tobytes())


if __name__ == "__main__":
    dump_data()
