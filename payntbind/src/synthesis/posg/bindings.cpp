#include "../synthesis.h"

#include "Posg.h"

void bindings_posg(py::module &m) {

    py::class_<synthesis::Stochastic2PlayerGame>(m, "Stochastic2PlayerGame")
        .def(py::init<storm::storage::SparseMatrix<double>,
                        storm::storage::SparseMatrix<double>,
                        storm::models::sparse::StateLabeling>());
    
    py::class_<synthesis::Posg>(m, "Posg")
        .def(py::init<storm::models::sparse::Pomdp<double>, std::string>(), py::arg("quotient"), py::arg("player1_label"));
}