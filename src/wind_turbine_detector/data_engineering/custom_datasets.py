from kedro.io import AbstractDataset
import glob
import os

class ImageFolderDataset(AbstractDataset):
    def __init__(self, path, image_extension='.tif'):
        self.path = path
        self.image_extension = image_extension
        super().__init__()

    def _load(self):
        images = glob.glob(os.path.join(self.path, f'*{self.image_extension}'))
        return images

    def _save(self, data):
        raise NotImplementedError("Saving is not supported for this dataset type")

    def _describe(self):
        return {
            "type": "ImageFolderDataset",
            "path": self.path,
            "image_extension": self.image_extension
        }
