"""從路徑取出房子名與影像識別碼。"""
import os

#: 全景影像副檔名（小寫）。比對時忽略大小寫。
IMAGE_EXTENSIONS = (".jpg", ".jpeg")


def house_of(path):
    """從影像路徑取出房子名。

    取影像的直接上層目錄名。資料集根目錄即使名稱含 DollhouseTask 或 direct_match，
    也不會蓋掉各房子自己的目錄名。路徑只剩檔名時回傳 unknown。
    """
    parts = [p for p in str(path).replace("\\", "/").split("/") if p not in ("", ".", "..")]
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


def image_stem(path):
    """影像檔名去掉副檔名。"""
    return os.path.splitext(os.path.basename(str(path)))[0]


def is_image_filename(name):
    """檔名是否為本專案接受的全景影像。"""
    return os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS


def dir_has_images(directory):
    """目錄當層是否含有全景影像。"""
    if not directory or not os.path.isdir(directory):
        return False
    try:
        for name in os.listdir(directory):
            if is_image_filename(name) and os.path.isfile(os.path.join(directory, name)):
                return True
    except OSError:
        return False
    return False


def find_image_file(directory, stem):
    """在目錄當層找 `stem` 加上 jpg／jpeg（大小寫皆可）。找不到回 None。"""
    if not directory or not os.path.isdir(directory):
        return None
    try:
        for name in os.listdir(directory):
            root, ext = os.path.splitext(name)
            if root == stem and ext.lower() in IMAGE_EXTENSIONS:
                path = os.path.join(directory, name)
                if os.path.isfile(path):
                    return path
    except OSError:
        return None
    return None
