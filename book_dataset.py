import requests
import pandas as pd
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

headers = {
    "X-Naver-Client-Id": "id",
    "X-Naver-Client-Secret": "key"
}

# queries = ["가", "나", "다", "라", "마", "바", "사", "아", "자", "차", "카", "타", "파", "하"]
queries = [
    "소설","문학","자기계발", "심리", "인문학", "철학", "소설", "단편소설", "장편소설",
    "경제", "경영", "투자", "주식", "부동산", "회계",
    "에세이", "힐링", "감성", "사랑", "연애", "가족",
    "여행", "음식", "요리", "사진", "문화",
    "IT", "AI", "프로그래밍", "코딩", "디자인", "UX",
    "육아", "교육", "청소년", "어린이", "동화", "그림책"
]

all_books = []
query_stats = {}

outer = tqdm(queries, desc=" 쿼리 진행", position=0)

for query in outer:
    count = 0
    inner = tqdm(range(1, 1001, 100), desc=f"{query}", position=1, leave=False)
    for start in inner:
        params = {
            "query": query,
            "display": 100,
            "start": start
        }
        url = "https://openapi.naver.com/v1/search/book.json"
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                items = response.json().get("items", [])
                all_books.extend(items)
                count += len(items)
            else:
                print(f"실패: {query} start={start} status={response.status_code}")
        except Exception as e:
            print(f"예외: {query} start={start} → {e}")
        time.sleep(0.4)
    query_stats[query] = count

df = pd.DataFrame([{
    "title": item["title"].replace("<b>", "").replace("</b>", ""),
    "author": item["author"],
    "isbn": item["isbn"],
    "publisher": item["publisher"],
    "description": item["description"].replace("<b>", "").replace("</b>", ""),
} for item in all_books])

df.drop_duplicates(subset=["isbn"], inplace=True)

print(df.head())

df.to_csv("naver_books.csv", index=False)
