import os

class Cleaner():

    def __init__(self):
        self.path = "./static/images"
        self.files = os.listdir(self.path)

    def clean(self):
        for file in self.files:
            file_path = os.path.join(self.path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                pass

