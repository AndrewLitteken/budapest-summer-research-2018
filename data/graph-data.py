import sys
import os

def gnuplot_file(data_file, class_dict, planes_or_supports, path):
    base = data_file.split("/")[-1].split(".dat")[0]
    plt_name = path + "/" + base + ".plt"
    png_name = path + "/" + base + ".png"
    plt_obj = open(plt_name, "w")
    plt_obj.write("set terminal pngcairo size 800, 640\n")
    plt_obj.write("set title \"Accuracy vs "+planes_or_supports+"\n")
    plt_obj.write("set output \""+png_name+"\"\n")
    plt_obj.write("set key below left\n")
    plt_obj.write("set xlabel \""+planes_or_supports+"\"\n")
    plt_obj.write("set ylabel \"Accuracy\"\n")
    plt_obj.write("plot ")
    
    dt = ""
    for index, classes in enumerate(class_dict.keys(),start=1):
        if 2*index > 9:
           dt = "dt 2" 
        cos_title = classes+" Classes, Cos Acc"
        plt_obj.write("\""+data_file+"\" using 1:"+str(2*index)+" with lines "+dt +" title \""+cos_title+"\", ")
        lsh_title = classes+" Classes, LSH Acc"
        plt_obj.write("\""+data_file+"\" using 1:"+str(2*index+1)+" with lines "+dt+" title \""+lsh_title+"\", ")
    plt_obj.write("\n")

    plt_obj.close()
    return plt_name

file_name = sys.argv[1]

file_obj = open(file_name, "r")

base_name = file_name.split("/")[-1].split(".csv")[0]
base_path = os.path.split(os.path.abspath(file_name))[0]
first_line = file_obj.readline()

line = file_obj.readline()
plane_dict = {}
support_dict = {}

while line:
    split = line.split(",")
    if "cosine" in base_name:
        if "one_rest" in base_name:
            classes = split[6]
            supports = split[7]
            cos = split[10]
            planes = classes
            true_lsh = split[11]
            sig_lsh = split[12]
        else:
            classes = split[6]
            supports = split[7]
            planes = split[8]
            cos = split[11]
            true_lsh = split[12]
            sig_lsh = split[13]
    else:
        if "one_rest" in base_name:
            classes = split[8]
            supports = split[9]
            planes = classes
            cos = split[12]
            true_lsh = split[13]
            sig_lsh = split[14]
        else:
            classes = split[8]
            supports = split[9]
            planes = split[10]
            cos = split[13]
            true_lsh = split[14]
            sig_lsh = split[15]

    class_dict = {"line": line, "cos": cos, "true_lsh": true_lsh, "sig_lsh": sig_lsh}

    if planes not in plane_dict.keys():
        plane_dict[planes] = {}
    if supports not in support_dict.keys():
        support_dict[supports] = {}

    if planes not in support_dict[supports].keys():
        support_dict[supports][planes] = {}
    if supports not in plane_dict[planes].keys():
        plane_dict[planes][supports] = {}

    plane_dict[planes][supports][classes] = class_dict
    support_dict[supports][planes][classes] = class_dict

    line = file_obj.readline()

file_obj.close()

if not os.path.exists(base_path + "/supports"):
    os.mkdir(base_path + "/supports")
if not os.path.exists(base_path + "/planes"):
    os.mkdir(base_path + "/planes")

for supports in support_dict.keys():
    if not os.path.exists(base_path + "/supports/"+supports+"-supports"):
        os.mkdir(base_path + "/supports/"+supports+"-supports")
    
    support_path = base_path + "/supports/"+supports+"-supports/"
    csv_file_name = support_path + base_name + "_" + supports + "_supports.csv"
    csv_file_obj = open(support_path + base_name + "_" + supports + "_supports.csv", "w")
    data_file_name = support_path + base_name + "_"+supports+"_supports.dat"
    data_file_obj = open(support_path + base_name + "_"+supports+"_supports.dat", "w")

    csv_file_obj.write(first_line)
    for planes in support_dict[supports].keys():
        data_file_obj.write("#planes ")
        for classes in support_dict[supports][planes].keys():
            data_file_obj.write(classes+"-cos-acc ")
            data_file_obj.write(classes+"-lsh-acc ")
        data_file_obj.write("\n")
        break

    for planes in support_dict[supports].keys():
        data_file_obj.write(planes+" ")
        for index, classes in enumerate(support_dict[supports][planes].keys()):
            class_dict = support_dict[supports][planes][classes]
            csv_file_obj.write(class_dict["line"])
            cos_acc = support_dict[supports][planes][classes]["cos"]
            lsh_acc = support_dict[supports][planes][classes]["true_lsh"]
            data_file_obj.write(cos_acc+" "+lsh_acc+" ")
        data_file_obj.write("\n")
    csv_file_obj.close()
    data_file_obj.close()
    for item in support_dict[supports].keys():
        plt_file = gnuplot_file(data_file_name, support_dict[supports][item], "Number of Planes", support_path)
        break
    os.system("gnuplot "+plt_file)

for planes in plane_dict.keys():
    if not os.path.exists(base_path + "/planes/"+planes+"-planes"):
        os.mkdir(base_path + "/planes/"+planes+"-planes")
    
    plane_path = base_path + "/planes/"+planes+"-planes/"
    csv_file_name = plane_path + base_name + "_"+planes+"_planes.csv"
    csv_file_obj = open(plane_path + base_name + "_"+planes+"_planes.csv", "w")
    data_file_name = plane_path + base_name + "_"+planes+"_planes.dat"
    data_file_obj = open(plane_path + base_name + "_"+planes+"_planes.dat", "w")

    csv_file_obj.write(first_line)
    for supports in plane_dict[planes].keys():
        data_file_obj.write("#supports ")
        for classes in plane_dict[planes][supports].keys():
            data_file_obj.write(classes+"-cos-acc ")
            data_file_obj.write(classes+"-lsh-acc ")
        data_file_obj.write("\n")
        break

    for supports in plane_dict[planes].keys():
        data_file_obj.write(supports+" ")
        for classes in plane_dict[planes][supports].keys():
            class_dict = plane_dict[planes][supports][classes]
            csv_file_obj.write(class_dict["line"])
            cos_acc = plane_dict[planes][supports][classes]["cos"]
            lsh_acc = plane_dict[planes][supports][classes]["true_lsh"]
            data_file_obj.write(cos_acc+" "+lsh_acc+" ")
        data_file_obj.write("\n")
    csv_file_obj.close()
    data_file_obj.close()
    for item in plane_dict[planes].keys():
        plt_file = gnuplot_file(data_file_name, plane_dict[planes][item], "Number of Supports", plane_path)
        break
    os.system("gnuplot "+plt_file)
