import pkgutil
import importlib
import sys

def find_all_modules(package_name):
    package = importlib.import_module(package_name)
    for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
        print(f"모듈 발견: {name}")
        if is_pkg:
            find_all_modules(name)

if __name__ == "__main__":
    try:
        find_all_modules("hongikjiki")
    except Exception as e:
        print(f"오류 발생: {e}")
        print("hongikjiki 모듈이 정상적으로 설치되어 있지 않을 수 있습니다.")
        