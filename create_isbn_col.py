import pandas as pd

df_user = pd.read_csv("peekabook_user_book_data.csv") # 유저 책장 데이터
df_meta = pd.read_csv("naver_books.csv") # title + isbn 포함

# 제목 기준 병합 (title 기준)
df_merged = df_user.merge(df_meta[["title", "isbn"]], on="title", how="left")

print(df_merged.head())

missing_rate = df_merged["isbn"].isnull().mean()
print(f"ISBN 매핑 실패율: {missing_rate * 100:.2f}%")

df_merged.to_csv("user_data_with_isbn.csv", index=False)