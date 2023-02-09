from urllib.request import urlopen
from urllib.parse import quote
from openpyxl import load_workbook
from tqdm import tqdm
import time

start = time.time()
def CIRconvert(ids):
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(ids) + '/smiles'
        ans = urlopen(url).read().decode('utf8')
        return ans
    except:
        return 'Did not work'

path = "C:\\Users\\mofan\\Documents\\My Document\\"
filename = "0503.xlsx"

workbook = load_workbook(filename=path + filename)
sheet = workbook.active

for n in tqdm(range(1761, 3428)):
    sheet['U' + str(n)] = CIRconvert(sheet['B' + str(n)].value)
    if n % 10 == 0:
        workbook.save(path + filename)


workbook.save(path + filename)
end = time.time()
print(end-start)