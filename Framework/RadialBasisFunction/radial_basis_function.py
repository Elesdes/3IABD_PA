import sys
import os
import ctypes as ct

DLL_BASE_NAME = "libRadialBasisFunction"


def main(*argv):
    dll_name = os.path.join(
        "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/Framework/RadialBasisFunction/cmake-build-debug",
        (argv[0] if argv else DLL_BASE_NAME) + ".dll")
    rbf = ct.CDLL(dll_name)
    func_test = rbf.return_bool
    func_test.restype = ct.c_bool

    res = func_test()
    print(res)


if __name__ == "__main__":
    rc = main(*sys.argv[1:])
    print("\nDone")
    sys.exit(rc)
