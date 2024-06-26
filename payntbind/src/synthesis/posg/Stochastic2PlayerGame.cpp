#include "Stochastic2PlayerGame.h"

namespace synthesis
{
    Stochastic2PlayerGame::Stochastic2PlayerGame(storm::storage::SparseMatrix<double>  player1Matrix,
                        storm::storage::SparseMatrix<double>  player2Matrix,
                        storm::models::sparse::StateLabeling statelabeling)
    {
        this->player1Matrix = player1Matrix;
        this->player2Matrix = player2Matrix;
        this->statelabeling = statelabeling;
    }

} // namespace synthesis
