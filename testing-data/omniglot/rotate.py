import os

base = os.getcwd()

for directory in os.listdir("."):
    path = base + "/" + directory
    if os.path.isdir(path) and path[0] != ".":
        for alphabet in os.listdir(path):
            alpha = path + "/" + alphabet
            if os.path.isdir(alpha) and path[0] != ".":
                for character in os.listdir(alpha):
                    char = alpha + "/" + character
                    if os.path.isdir(char) and path[0] != ".":
                        for image in os.listdir(char):
                            img = char + "/" + image
                            if os.path.isfile(img):
                                img = img.replace("(", "\(")
                                img = img.replace(")", "\)")
                                for amount in ["90", "180", "270"]:
                                    os.system("convert -rotate \"{}\" {} {}".format(amount, img, img.split(".")[0]+"_"+amount+".png"))
