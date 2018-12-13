input_f = "frame.obj"
output_f = "frame_new.obj"

lines = open(input_f).read().strip().split("\n")
out = open(output_f, "w")
for line in lines:
    if line[:2] == "vn":
        parts = line.split()
        # Play around with this line.
        (parts[2], parts[3]) = (parts[3], str(-float(parts[2])))
        line = " ".join(parts)

    print(line, file=out)

out.close()