#if WIN32
#define DLLEXPORT __declspec(dllexport)
#elif
#define DLLEXPORT
#endif

#include <stdlib.h>
#include <stdio.h>

#include <cstdint>
#include <iostream>


extern "C" {
    DLLEXPORT char return_bool() {
        return (3 + 2 == 5 ? 1 : 0);
    }
}
