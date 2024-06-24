#include "../synthesis.h"

#include "Posg.h"

void bindings_posg(py::module &m) {

    m.def("foo", &synthesis::foo, py::arg("quotient"));
}