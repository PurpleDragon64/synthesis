#include "Stochastic2PlayerGame.h"

namespace synthesis
{
    Stochastic2PlayerGame::Stochastic2PlayerGame(storm::storage::SparseMatrix<double>  player1Matrix,
                        storm::storage::SparseMatrix<double>  player2Matrix,
                        storm::models::sparse::StateLabeling statelabeling,
                        ItemTranslator stateToP1State,
                        ItemTranslator stateToP2State)
    {
        this->player1Matrix = player1Matrix;
        this->player2Matrix = player2Matrix;
        this->statelabeling = statelabeling;
        this->stateToP1State = stateToP1State;
        this->stateToP2State = stateToP2State;
    }

} // namespace synthesis
