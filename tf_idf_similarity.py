import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("naver_books.csv")

# 결측치 제거 (description 등 결측 가능성)
df = df.fillna("")

# 텍스트 파싱: title + description + author 를 하나로 묶기
df["parsed_text"] = df["title"] + " " + df["description"] + " " + df["author"]

# TF-IDF 벡터화
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
tfidf_matrix = vectorizer.fit_transform(df["parsed_text"])

# 유사도 계산 함수
def recommend_by_index(index, top_n=5):
    cosine_sim = cosine_similarity(tfidf_matrix[index], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]
    return df.iloc[similar_indices][["title", "author", "isbn"]]

# 예시: 0번 책과 유사한 책 5개 보기
print("기준 책:", df.iloc[0]["title"])
print(recommend_by_index(0))
