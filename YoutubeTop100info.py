import requests as rq
import pandas as pd
import csv
from bs4 import BeautifulSoup

channal_name = []
channal_id = []
times= 1
with open('S:\\TaiwanSubTop100\TaiwanSubTop100list.csv') as csvfile:
    channal = csv.reader(csvfile)
    for x in channal:
        channal_name.append(x[0])
        channal_id.append(x[1])
date = pd.date_range(start='2019/10/26',end='2021/10/26').strftime('%Y-%m-%d').to_list()#選擇日期間隔
header = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36 Edg/95.0.1020.53'}
# ---------------------------------------爬取指定頻道的發布影片資訊---------------------------------------#
table = {}   
df = pd.DataFrame(table)
for id in channal_id[1:]:#從第1個channal_id開始,可自行調整從第幾個開始
    url2 = 'https://tw.noxinfluencer.com/api/video/list?channelId='+id+'&offset=0&sortField=pub_date&liveStream=false&pageSize=1000&isPredicted=0&r=mtRgFKUILnhK2z7lGovIFT44VprmMH6stefxgSs5RHzNYICAzaBaKcRHkZxAnmIeuh7VW4IfdvQ8asCQpqJNMD%2FsL1S6GDHZDxyLJs51PEVNfGgCoOnpg6BhT6S%2B0EhnA4JYvbVDem%2FaGSSGi8j6O%2Fd4%2FnI64G4rlFRasNb5a48%3D'
    res = rq.get(url2,headers=header)#爬的時候要注意,可能爬到一半被鎖就要重新調整網址,從被鎖的那一篇開始爬
    soup = BeautifulSoup(res.text,"html.parser")
    while 1:
        try:
            df =df.append(
                {'影片名稱':soup.find_all('div',class_='\\"detail\\"')[times].span.text.strip(),
                '觀看人次':soup.find_all('div',class_='\\"detail\\"')[times].find_all('div',class_='\\"subtitle\\"')[0].find_all('span')[0].text.strip()[:-1],
                '上傳日期':soup.find_all('div',class_='\\"detail\\"')[times].find_all('div',class_='\\"subtitle\\"')[0].find_all('span')[1].text.strip()},ignore_index=True)
            times += 1
        except:
            times =0
            with pd.ExcelWriter(engine='openpyxl', path="S:\\TaiwanSubTop100\\Top100info\\"+channal_name[channal_id.index(id)]+".xlsx", mode='a') as writer:
                try:
                    book = writer.book
                    book.remove(book['發布影片'])
                    df.to_excel(writer,sheet_name='發布影片')
                except:
                    df.to_excel(writer,sheet_name='發布影片')
            break
    df = pd.DataFrame(table)#檔案清空

# #---------------------------------------爬取指定頻道的資訊---------------------------------------#
        
    for x in date[::30]: 
        url = "https://tw.noxinfluencer.com/api/youtube/detail/dimension/?window=29&channelId="+id+"&startDate="+x+"&cpmMin=1&cpmMax=1.9&r=mFFO3VywH%2FMOQ4dk6IuX6UoIm38yC9lNej2aGfrgc06fYZhRK2ZJx%2Fh6ySA61DBtCBOzcBTIp5nGprE%2Fr2bgrf8M6KA%2B9U3cGzYdBTqFITNfnfO46pXfBvlO07YVbBG6nCKGBSk%2BgPuZp1x4Qqgw7doKDUmxJkodXXWkwGEJs%2Bc%3D"
        res = rq.get(url,headers=header)
        soup = BeautifulSoup(res.text,"html.parser")
        for y in range(30):
            try: 
                a =float(soup.find_all("tbody")[0].find_all("tr")[y].find_all("td")[2].find_all("span",class_='\\"kolicon')[0].text[:-1])*10000
            except:
                a = 0
            df = df.append(
                {"日期":soup.find_all("tbody")[0].find_all("tr")[y].find_all("td")[0].text,
                "訂閱數":float(soup.find_all("tbody")[0].find_all("tr")[y].find_all("td")[1].span.text[:-1])*10000,
                "總觀看次數":float(soup.find_all("tbody")[0].find_all("tr")[y].find_all("td")[2].find_all("span")[0].text[:-1])*10000000,
                "觀看成長量":a},ignore_index=True)     
    df.to_excel("S:\\TaiwanSubTop100\\Top100info\\"+channal_name[channal_id.index(id)]+".xlsx")
    df = pd.DataFrame(table)#檔案清空
