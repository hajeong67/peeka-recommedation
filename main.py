import pandas as pd
import os
import numpy as np

from kobert_model import load_kobert_model, encode_books
from data_utils import normalize_books, remap_logs_to_representative_isbn
from recommend_engine import hybrid_recommend_with_scores, get_global_top_picks

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# 책 메타데이터 로드 및 정규화
df_books = pd.read_csv("naver_books.csv", usecols=["title", "author", "isbn", "description"]).fillna("")
df_books = normalize_books(df_books)

# KoBERT 임베딩 로드 또는 생성
EMBEDDING_PATH = "bert_embeddings.npy"
if os.path.exists(EMBEDDING_PATH):
    print("캐시된 KoBERT 임베딩 불러오는 중...")
    bert_embeddings = np.load(EMBEDDING_PATH)
else:
    print("KoBERT 임베딩 생성 중...")
    model = load_kobert_model()
    bert_embeddings = encode_books(model, df_books["parsed_text"].tolist())
    np.save(EMBEDDING_PATH, bert_embeddings)
    print("임베딩 저장 완료: bert_embeddings.npy")

# 유저 로그 로드 및 처리
logs_df = pd.read_csv("user_data_with_isbn.csv", usecols=["user_id", "isbn", "pickIndex"])
logs_df = remap_logs_to_representative_isbn(logs_df, df_books)
logs_df = logs_df[logs_df["isbn"].isin(df_books["isbn"])]
logs_df["interact"] = 1

# 유저별 pick/shelf 딕셔너리, 사용자-아이템 매트릭스 생성
valid_pick_df = logs_df[logs_df["pickIndex"].isin([1, 2, 3])]
valid_shelf_df = logs_df[logs_df["pickIndex"] == 0]
user_pick_dict = valid_pick_df.groupby("user_id")["isbn"].apply(list).to_dict()
user_shelf_dict = valid_shelf_df.groupby("user_id")["isbn"].apply(list).to_dict()
user_item_df = logs_df.pivot_table(index="user_id", columns="isbn", values="interact", fill_value=0)

# 추천 실행
# user_id = 191
user_id = 296
result = hybrid_recommend_with_scores(
    user_id=user_id,
    user_item_df=user_item_df,
    user_pick_dict=user_pick_dict,
    user_shelf_dict=user_shelf_dict,
    df_books=df_books,
    bert_embeddings=bert_embeddings,
    alpha=0.7,
    top_n=10
)

# 결과 출력 및 저장
print(result)
result.to_csv(f"recommend_user_{user_id}.csv", index=False)
