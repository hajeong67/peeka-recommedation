import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

def get_user_profile_vector_kobert(pick_isbns, shelf_isbns, isbn_to_index, embeddings):
    pick_weight = 1.0
    shelf_weight = 0.3
    vectors, weights = [], []

    for isbn in pick_isbns:
        idx = isbn_to_index.get(isbn)
        if idx is not None:
            vectors.append(embeddings[idx])
            weights.append(pick_weight)
    for isbn in shelf_isbns:
        if isbn not in pick_isbns:
            idx = isbn_to_index.get(isbn)
            if idx is not None:
                vectors.append(embeddings[idx])
                weights.append(shelf_weight)

    if not vectors:
        return np.mean(embeddings, axis=0)

    return np.average(np.stack(vectors), axis=0, weights=weights)

def get_global_top_picks(user_pick_dict, df_books, top_n=3):
    all_picks = [isbn for picks in user_pick_dict.values() for isbn in picks]
    top_isbns = [isbn for isbn, _ in Counter(all_picks).most_common(top_n)]
    return df_books[df_books["isbn"].isin(top_isbns)][["title", "author", "isbn"]]

def hybrid_recommend_with_scores(user_id, user_item_df, user_pick_dict, user_shelf_dict, df_books, bert_embeddings, alpha=0.7, top_n=10):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    import pandas as pd

    isbn_to_index = {isbn: i for i, isbn in enumerate(df_books["isbn"])}

    if user_id not in user_item_df.index:
        print(f"[하드 콜드스타트] 유저 {user_id}")
        return get_global_top_picks(user_pick_dict, df_books, top_n)

    # Nearest Neighbors 기반 협업 필터링 준비
    knn = NearestNeighbors(n_neighbors=min(20, len(user_item_df)), metric="cosine")
    knn.fit(user_item_df)
    target_index = user_item_df.index.get_loc(user_id)
    distances, indices = knn.kneighbors([user_item_df.iloc[target_index]])

    neighbor_ids = user_item_df.index[indices[0]]
    similarities = 1 - distances[0]  # cosine distance → similarity (0~1)
    neighbors_matrix = user_item_df.loc[neighbor_ids].values  # shape: (k, num_books)

    # 개선된 collab_score: 유사도 기반 가중 합
    weighted_sum = np.dot(similarities, neighbors_matrix)  # shape: (num_books,)
    weighted_sum_series = pd.Series(weighted_sum, index=user_item_df.columns)

    # 내가 보지 않은 책 중에서만 추천
    my_vector = user_item_df.loc[user_id]
    recommend_candidates = weighted_sum_series[(my_vector == 0) & (weighted_sum_series > 0)]

    if recommend_candidates.empty:
        print("유효한 협업 후보 없음")
        return pd.DataFrame()

    candidate_isbns = recommend_candidates.index.tolist()
    collab_scores = recommend_candidates.to_dict()

    # 콘텐츠 기반 벡터
    user_profile_vector = get_user_profile_vector_kobert(
        pick_isbns=user_pick_dict.get(user_id, []),
        shelf_isbns=user_shelf_dict.get(user_id, []),
        isbn_to_index=isbn_to_index,
        embeddings=bert_embeddings
    ).reshape(1, -1)

    candidate_indices = [isbn_to_index[i] for i in candidate_isbns if i in isbn_to_index]
    if not candidate_indices:
        print("유효한 콘텐츠 후보 없음")
        return pd.DataFrame()

    similarities = cosine_similarity(user_profile_vector, bert_embeddings[candidate_indices]).flatten()
    content_scores = {candidate_isbns[i]: similarities[i] for i in range(len(candidate_indices))}

    result_rows = []
    for isbn in candidate_isbns:
        c_score = content_scores.get(isbn, 0)
        u_score = collab_scores.get(isbn, 0)
        final_score = alpha * u_score + (1 - alpha) * c_score
        result_rows.append({
            "isbn": isbn,
            "title": df_books.loc[df_books["isbn"] == isbn, "title"].values[0] if isbn in df_books["isbn"].values else "N/A",
            "author": df_books.loc[df_books["isbn"] == isbn, "author"].values[0] if isbn in df_books["isbn"].values else "N/A",
            "collab_score": round(u_score, 3),
            "content_score": round(c_score, 3),
            "final_score": round(final_score, 3)
        })

    sorted_df = pd.DataFrame(result_rows).sort_values(by="final_score", ascending=False).head(top_n)
    return sorted_df.reset_index(drop=True)

