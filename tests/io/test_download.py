import shutil
import tempfile
import unittest

from t2m import io


class testDownload(unittest.TestCase):
    def setUp(self):
        self.tmp_path = tempfile.mkdtemp()

    def test_download_url(self):
        url = "https://api.worldbank.org/v2/en/indicator/NY.GDP.MKTP.CD?downloadformat=csv"
        io.download.url(url, self.tmp_path)

    def tearDown(self):
        shutil.rmtree(self.tmp_path)
