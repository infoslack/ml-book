ga_tag = open("ga_tag.txt", "r").read().strip()
ga_tag_lines = ga_tag.split("\n")
ga_tag_lines = [f"{line}\n" for line in ga_tag_lines]
ga_tag_lines[0] = "        " + ga_tag_lines[0]

from os import listdir
from os.path import isfile, join
path = "_book/contents"
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
onlyfiles = [f"{path}/{file}" for file in onlyfiles]
onlyfiles.append("_book/index.html")

n = len(" · HonKit</title>")
for file in onlyfiles:
    print(file)
    if not file.endswith("html"):
        print(file)
        continue
    correct_lines = []
    prev = None
    with open(file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if prev == "<head>":
                correct_lines.extend(ga_tag_lines)
            if line.strip().endswith(" · HonKit</title>"):
                line = line[:-n-1] + " · MLIB</title>\n"
                correct_lines.append(line)
                break
            correct_lines.append(line)
            prev = line.strip()
        correct_lines.extend(lines[i + 1:])
    with open(file, "w") as f:
        text = "".join(correct_lines)
        f.write(text)