#include "Posg.h"

#include <stdio.h>

namespace synthesis {
    void foo(storm::models::sparse::Pomdp<double>)
    {
        printf("bind successful\n");
    }
}