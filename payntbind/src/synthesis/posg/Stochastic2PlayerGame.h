#pragma once

#include "storm/models/sparse/NondeterministicModel.h"
#include "src/synthesis/translation/ItemTranslator.h"


namespace synthesis
{
    /**
     * @brief
     * A class representing two player stochastic game, where both player can have distributions
     * in their actions.
     */
    class Stochastic2PlayerGame
    {
        public:
        // Destination states (columns) in player1's transtition matrix correspond to source states (row groups)
        // in player2's transition matrix and vice versa
        Stochastic2PlayerGame(storm::storage::SparseMatrix<double> player1Matrix,
                              storm::storage::SparseMatrix<double> player2Matrix,
                              storm::models::sparse::StateLabeling statelabeling,
                              ItemTranslator stateToP1State,
                              ItemTranslator stateToP2State);

        private:
        storm::storage::SparseMatrix<double> player1Matrix;
        storm::storage::SparseMatrix<double> player2Matrix;
        /** Labeling for both players. Use state transaltors */
        storm::models::sparse::StateLabeling statelabeling;
        ItemTranslator stateToP1State;
        ItemTranslator stateToP2State;
    };
} // namespace synthesis
