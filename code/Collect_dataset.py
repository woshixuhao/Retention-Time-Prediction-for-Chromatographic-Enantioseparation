#program 1
import os
# 存储文件
def save_to_file(file_name, contents):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()

# 文件位置
path = "C:\\Users\\mofan\\Documents\\My Document\\07 Papers\\Zhang Wanbin 2018\\"
# filename = "Zhang Wanbin_xxx_10.1039-C8SC04626C.txt"
files = os.listdir(path)

for filename in files:
    # 将所有行合并成一行
    with open(path + filename, encoding='gbk') as f:  # 从TXT文件中读出数据
        data = f.readlines()  # 读取每一行
    a = str()  # 定义一个空字符串，用于临时存放
    for i in data:
        a += i.replace("\n", " ")  # 把每一行的换行改为空格，并且合并为一行 （保存到变量a中，变量a是str形式）

    # 去除一些可能造成影响的字符
    for i in range(1, 100):  # 这里去除的是pdf copy过程中的页码，如S10
        a = a.replace('S' + str(i), '')
    #     a = a.replace(' ' + str(i) + ' ', ' ')


    b = a.encode('GBK', 'ignore')  # 忽略不能解码的字符
    c = str(b).replace('b\'', '')  # 再做调整，删除b‘
    c = str(c).replace('v/cm', '')
    save_to_file(path + filename, c)  # 存回原来的文件

c = str(c).replace('. Allyl', '. \nAllyl')
save_to_file(path + filename, c)  # 存回原来的文件

e = '(E)'
c = str(c).replace(e, '\n' + e)
save_to_file(path + filename, c)  # 存回原来的文件

#program2

import pandas as pd
import os

path = "C:\\Users\mofan\Documents\\My Document\\07 Papers\\test\\"  # 文件夹目录
files = os.listdir(path)

# 手工调整完之后，运行这个文件

def save_to_file(file_name, contents):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()

for filename in files:
    with open(path + filename, encoding='utf-8') as f:  # 从TXT文件中读出数据
        row = f.readlines()  # 读取每一行

name = list()
HPLC = list()

for i in range(len(row)):
    row[i] = row[i].strip()
    name.append(row[i].split('?')[0])  # 这里的符号根据需要调整
    HPLC.append(row[i].split('?')[1])

dataframe = pd.DataFrame({'name': name, 'HPLC': HPLC})
dataframe.to_csv(path + filename[:-4] + ".csv", index=False, sep=',')

#program3

import numpy as np
import pandas as pd
import re
import os

# 可能用到的一些函数
def find_word(word, text):
    return word in text

def get_index(word, text):
    return text.index(word)

def get_index1(lst=None, item=''):
    return [index for (index, value) in enumerate(lst) if value == item]

def get_word(text, rank):
    return text[rank]

def concat_words(text, n, m):
    words = text.split()
    word_1 = words[n]
    word_2 = words[m]
    return word_1 + word_2

def getmidstring(string, start_str, end):
    start = string.find(start_str)
    if start >= 0:
        start += len(start_str)
        end = string.find(end, start)
        if end >= 0:
            return string[start:end].strip()

# 调用HPLC关键词数据库，持续更新中
columns = list()
models = list()
solvents = list()
with open(r"C:\Users\mofan\Documents\My Document\07 Papers\origin data\HPLC_info.txt") as f:
    row = f.readlines()
for item in row[0].split(','):  # 第0行是柱子信息
    columns.append(item)
for item in row[1].split(','):  # 第1行是柱子型号信息
    models.append(item)
for item in row[2].split(','):  # 第2行是溶剂信息
    solvents.append(item)

# 声明一个新的dataframe
dataframe = pd.DataFrame(
    columns=['No.', 'Name', 'Sub_brand', 'Model', 'solvent_1', 'solvent_2', 'n-Hexane', 'i-PrOH', 'Speed'
            , 't1 (min) (major)', 't2 (min)'
            , 'ee (%)', 'Optical rotation', 'Concentration', 'Solvent', 'Detector'
            , 'Literature', 'Author', 'Instrument'])

# 载入文件夹，对文件夹中的所有文件进行处理
path = "C:\\Users\mofan\Documents\\My Document\\07 Papers\\test\\"  # 文件夹目录
files = os.listdir(path)
print(files)

n = 0
for filename in files:
    print(filename)
    df = pd.read_csv(path + filename)
    m = 0
    for i in range(len(df)):
        # 取化合物名称
        compound_name = str(df.iloc[i][0])
        # 取作者名
        author = filename.split('_')[0]
        # 取文献名
        literature = filename[:-4].split('_')[2].replace('-', '/')
        # 取仪器名
        instrument = filename.split('_')[1]

        # 将HPLC信息预处理
        data_origin = df.iloc[i][1].replace(',', '').replace(')', '').replace(';', '').replace('=', ' ').split()
        data = list()
        for item in data_origin:
            if bool(re.search(r'\d', item)):
                pass
            else:
                item = item.replace('.', '')
            data.append(item)

        # 取手性柱信息
        column = str()
        model = str()
        for item in columns:
            if find_word(item, data):
                column = item
        for item in models:
            if find_word(item, data):
                model = item

        # 取溶剂及比例
        solvent = list()
        ratio = np.full(4, np.nan)
        ratio_1 = np.full(4, np.nan)
        ratio_2 = np.full(4, np.nan)
        for item in solvents:
            if find_word(item, data):
                solvent.append(item)
        solvent.extend(["null"] * 2)

        for item in data:
            if '/' in item:
                if bool(re.search(r'\d', item)):
                    if item[-1] != ':':
                        ratio = item
                        # print(ratio)
        ratio_1 = ratio.split('/')[0]
        ratio_2 = ratio.split('/')[1]


        # 取流速
        speed = str()
        speed = get_word(data, get_index('mL/min', data) - 1)

        # 取保留时间
        minor_index = int()
        for item in data:
            if 'minor' in item:
                minor_index = get_index(item, data)
        major_index = int()
        for item in data:
            if 'major' in item:
                major_index = get_index(item, data)

        tr = np.full(4, np.nan)
        tr_index = np.full(4, np.nan)
        if len(get_index1(data, 'min')) >= 2:
            tr[0] = (get_word(data, get_index1(data, 'min')[0] - 1))
            tr[1] = (get_word(data, get_index1(data, 'min')[1] - 1))

        if minor_index < major_index:
            tr[0], tr[1] = tr[1], tr[0]

        # 取ee值
        ee_index = int()
        ee = str()
        for item in data:
            if 'ee' in item:
                ee_index = get_index(item, data)
        ee = get_word(data, ee_index - 1)

        # 取旋光值
        rotation_index = int()
        rotation = str()
        c = str()
        rotation_solvent = str()
        for item in data:
            if '(c' in item:
                rotation_index = get_index(item, data)
                if get_word(data, rotation_index - 2) == '-':
                    rotation = '-' + get_word(data, rotation_index - 1)
                else:
                    rotation = get_word(data, rotation_index - 1)
                c = get_word(data, rotation_index + 1)
                rotation_solvent = get_word(data, rotation_index + 2)

        # 取检测器波长
        detector = str()
        if find_word('nm', data):
            detector = get_word(data, get_index('nm', data) - 1)

        dataframe = dataframe.append({'No.': i+1, 'Name': compound_name
                                         , 'Sub_brand': column
                                         , 'Model': model
                                         , 'solvent_1': solvent[0]
                                         , 'solvent_2': solvent[1]
                                         , 'n-Hexane': ratio_1, 'i-PrOH': ratio_2
                                         , 'Speed': speed
                                         , 't1 (min) (major)': tr[0], 't2 (min)': tr[1]
                                         , 'ee (%)': ee
                                         , 'Optical rotation': rotation
                                         , 'Concentration': c
                                         , 'Solvent': rotation_solvent
                                         , 'Detector': detector
                                         , 'Instrument': instrument
                                         , 'Literature': literature
                                         , 'Author': author}
                                         , ignore_index=True)
        m += 1
        print('m =', m)
    n += 1
    print('n =', n)

dataframe.to_csv("1.csv", index=False, sep=',')

