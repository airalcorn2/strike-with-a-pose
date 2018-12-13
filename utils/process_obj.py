# Get all vns first.
vns = []
lines = open("Jeep.obj").read().strip().split("\n")
for line in lines:
    parts = line.split()
    if parts[0] == "vn":
        vns.append(line)

vs = []
vts = []
fs = []

total_vs = 0
total_vts = 0
total_fs = 0

cur_obj = 0

lines = open("Jeep.obj").read().strip().split("\n")
for line in lines:

    parts = line.split()

    if parts[0] == "v":

        if len(fs) > 0:

            total_vs += len(vs)
            total_vts += len(vts)
            total_fs += len(fs)

            out_f = open("obj_{0}.obj".format(cur_obj), "w")
            print("\n".join(vs), file=out_f)
            print("\n".join(vts), file=out_f)
            print("\n".join(vns), file=out_f)
            print("\n".join(fs), file=out_f)
            out_f.close()

            cur_obj += 1
            vs = []
            vts = []
            fs = []

        vs.append(line)

    elif parts[0] == "vt":

        vts.append(line)

    elif parts[0] == "f":

        old_f = parts[1:4]
        new_f = []

        for fv in old_f:

            (v, vt, vn) = fv.split("/")
            v = str(int(v) - total_vs)
            assert "-" not in v
            vt = str(int(vt) - total_vts)
            assert "-" not in vt
            new_f.append("{0}/{1}/{2}".format(v, vt, vn))

        fs.append("f " + " ".join(new_f))

out_f = open("obj_{0}.obj".format(cur_obj), "w")
print("\n".join(vs), file=out_f)
print("\n".join(vts), file=out_f)
print("\n".join(vns), file=out_f)
print("\n".join(fs), file=out_f)
out_f.close()