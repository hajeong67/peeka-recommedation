import pandas as pd

df = pd.read_csv("user_data_with_isbn.csv")

missing_isbn = df[df["isbn"].isnull()]

missing_titles = missing_isbn["title"].dropna().drop_duplicates().reset_index(drop=True)

print(f"ISBN이 없는 책 제목 수: {len(missing_titles)}")
print(missing_titles)
missing_titles.to_csv("missing_titles.txt", index=False, header=False)

missing_titles = missing_isbn["title"].dropna().drop_duplicates().reset_index(drop=True)
first_words = missing_titles.apply(lambda x: x.strip().split()[0] if isinstance(x, str) else "")
first_words.to_csv("missing_title_first_words.txt", index=False, header=False)
