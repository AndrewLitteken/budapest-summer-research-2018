import os

base = os.getcwd()
read_base = base + "/omniglot"
write_base = base + "/omniglot-rotate"

if not os.path.exists(write_base):
    os.mkdir(write_base)
for directory in os.listdir(read_base):
    read_path = read_base + "/" + directory
    write_path = write_base + "/" + directory
    if os.path.isdir(read_path) and read_path[0] != ".":
        if not os.path.exists(write_path):
            os.mkdir(write_path)
        for alphabet in os.listdir(read_path):
            read_alpha = read_path + "/" + alphabet
            write_alpha = write_path + "/" + alphabet
            if os.path.isdir(read_alpha) and read_alpha[0] != ".":
                if not os.path.exists(write_alpha):
                    os.mkdir(write_alpha)
                for character in os.listdir(read_alpha):
                    read_char = read_alpha + "/" + character
                    write_char = write_alpha + "/" + character
                    if os.path.isdir(read_char) and read_char[0] != ".":
                        if not os.path.exists(write_char):
                            os.mkdir(write_char)
                        for image in os.listdir(read_char):
                            read_img = read_char + "/" + image
                            write_img = write_char + "/" + image
                            if os.path.isfile(read_img):
                                read_img = read_img.replace("(", "\(")
                                read_img = read_img.replace(")", "\)")
                                write_img = write_img.replace("(", "\(")
                                write_img = write_img.replace(")", "\)")
                                os.system("cp {} {}".format(read_img, write_img))
                                for amount in ["90", "180", "270"]:
                                    os.system("convert -rotate \"{}\" {} {}".format(amount, read_img, "/".join(write_img.split("/")[:-1])+"/"+write_img.split("/")[-1].split(".")[0]+"_"+amount+".png"))
