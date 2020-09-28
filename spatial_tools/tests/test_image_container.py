import pytest

def _create_tiff(fname, h, w, c, num=1):
    """
    use tiffile to create a (multi-page) multi-channel tiff 
    """
    pass

def test_image_loading():
    """
    initialize ImageObject with tiff / multipagetiff / numpy array and check that loaded data 
    fits the expected shape + content
    """
    pass

def test_add_img():
    """
    add image to existing ImageObject and check result
    """
    pass
    
def test_crop():
    """
    crop different img_ids and check result
    """
    pass