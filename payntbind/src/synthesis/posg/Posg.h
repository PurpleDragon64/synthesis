#pragma once

#include <storm/models/sparse/Pomdp.h>
#include <storm/adapters/RationalFunctionAdapter.h>
#include <storm/storage/SparseMatrix.h>
#include "Stochastic2PlayerGame.h"
#include "src/synthesis/translation/ItemTranslator.h"


namespace synthesis {
    /**
     * @brief 
     * A class representing quotient for Partialy Observable Stochastic Games
     */
    class Posg
    {        
        public:
        /**
         * @brief Construct a new Posg quotient
         * 
         * @param quotient Pomdp representing the game. States with label player1label are considered as
         * states from player 1, all other states are player 2's states
         * @param player1label Used to indicate player 1 states in quotient
         */
        Posg(storm::models::sparse::Pomdp<double> quotient, std::string player1label);

        private:        
        storm::models::sparse::Pomdp<double> pomdp;
        std::shared_ptr<Stochastic2PlayerGame> game;
        
        std::shared_ptr<Stochastic2PlayerGame> createGame(std::string p1label);

        void getStateTranslations(ItemTranslator &p1Translator, ItemTranslator &p2Translator, std::string p1label);
        
        storm::storage::SparseMatrix<double> createTransitionMatrix(
            ItemTranslator mainPlayerTranslator,
            ItemTranslator otherPlayerTraslator);
    };
}
