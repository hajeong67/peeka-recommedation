import pandas as pd

def normalize_books(df_books):
    df_books["isbn"] = df_books["isbn"].astype(str).str.split(".").str[0]
    df_books = df_books.fillna("")
    df_books["parsed_text"] = df_books["title"] + " " + df_books["description"] + " " + df_books["author"]
    df_books_unique = df_books.groupby(["title", "author"]).first().reset_index()
    return df_books_unique

def remap_logs_to_representative_isbn(logs_df, df_books):
    logs_df["isbn"] = logs_df["isbn"].astype(str).str.split(".").str[0]
    df_books_unique = df_books.drop_duplicates(subset=["title", "author"], keep="first")
    title_author_to_isbn = df_books_unique.set_index(["title", "author"])["isbn"].to_dict()
    logs_df = logs_df.merge(df_books[["isbn", "title", "author"]], on="isbn", how="left")
    logs_df["isbn"] = logs_df[["title", "author"]].apply(
        lambda row: title_author_to_isbn.get((row["title"], row["author"])), axis=1
    )
    return logs_df.dropna(subset=["isbn"])
