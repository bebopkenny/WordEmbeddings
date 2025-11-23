import pandas as pd
import pandera.pandas as pa

aggregated_news_schema = pa.DataFrameSchema(
    {
        "date": pa.Column(pd.Timestamp, nullable=False),
        "symbol": pa.Column(str, nullable=False),
        "news": pa.Column(str, nullable=False),
    },
    coerce=True
)

vectorized_news_skipgram_schema = pa.DataFrameSchema(
    {
        "date": pa.Column(pd.Timestamp, nullable=False),
        "symbol": pa.Column(str, nullable=False),
        "news_vector": pa.Column(object, nullable=False),  # Assuming news_vector can be a list/array
        "impact_score": pa.Column(float, nullable=False),
    },
    coerce=True
)

vectorized_news_cbow_schema = pa.DataFrameSchema(
    {
        "date": pa.Column(pd.Timestamp, nullable=False),
        "symbol": pa.Column(str, nullable=False),
        "news_vector": pa.Column(object, nullable=False),  # Assuming news_vector can be a list/array
        "impact_score": pa.Column(float, nullable=False),
    },
    coerce=True
)

