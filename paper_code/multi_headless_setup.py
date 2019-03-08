import json

TEMPLATE_1 = """
Section "Device"
    Identifier     "Device{0}"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
    BoardName      "{1}"
    BusID          "PCI:{2}"
EndSection
"""
TEMPLATE_2 = """
Section "ServerLayout"
    Identifier     "Layout{0}"
    Screen      0  "Screen{0}"
EndSection

Section "Screen"
    Identifier     "Screen{0}"
    Device         "Device{0}"
    Monitor        "Monitor{0}"
    DefaultDepth    24
    Option         "UseDisplayDevice" "None"
    SubSection     "Display"
        Virtual     1280 1024
        Depth       24
    EndSubSection
EndSection
"""

# nvidia-xconfig --query-gpu-info
gpu_info = json.load(open("gpu_info.json"))

TEMPLATE_2_GPUS = {"GeForce GTX 1080 Ti"}

for (gpu, info) in enumerate(gpu_info):
    (BOARDNAME, BUSID) = (info["BOARDNAME"], info["BUSID"])
    conf = TEMPLATE_1.format(gpu, BOARDNAME, BUSID)
    if BOARDNAME in TEMPLATE_2_GPUS:
        conf = TEMPLATE_2.format(gpu) + conf

    conf_f = open("/etc/X11/xorg.conf.{0}".format(gpu), "w")
    conf_f.write(conf)
    conf_f.close()