from t2m import io


def test_download_url(tmp_path):
    url = "https://api.worldbank.org/v2/en/indicator/NY.GDP.MKTP.CD?downloadformat=csv"
    io.download.url(url, tmp_path)
