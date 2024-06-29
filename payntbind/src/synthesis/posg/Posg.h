#pragma once

#include <storm/models/sparse/Pomdp.h>
#include <storm/adapters/RationalFunctionAdapter.h>
#include <storm/storage/SparseMatrix.h>
#include "Stochastic2PlayerGame.h"
#include "src/synthesis/translation/ItemTranslator.h"


namespace synthesis {
    struct IntermediateState
    {
        uint64_t stateNumber;
        uint64_t dstState;
        bool isP1State;
    };

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
        /** Original POMDP representing the game (received in constructor) */
        storm::models::sparse::Pomdp<double> pomdp;
        /** Game stored in Stochastic2PlayerGame format */
        std::shared_ptr<Stochastic2PlayerGame> game;
        /** Label used to identify Player 1's states in original POMDP */
        std::string p1label;

        /**
         * @brief Find and return iterator of intermediate state with specified destination state or states.end() if
         * no such state found.
         *
         * @param states States to search in.
         * @param dstState Destination state to look for.
         * @return std::vector<IntermediateState>::iterator Interator pointing to found state or states.end() if not found.
         */
        std::vector<IntermediateState>::iterator findIntermediateState(std::vector<IntermediateState> &states, uint64_t dstState);

        /**
         * @brief Create and return a new transition matrix that is alternating by adding intermediate states
         * to original transitionMatrix.
         * @details  An intermediate state is added if there is a transition from one
         * player's state to the same player's state. This added state then belongs to the other player
         * and has a single transition leading to the original destination state with probability 1.
         *
         * @param transitionMatrix Original transition matrix
         * @param stateLabeling Labeling of states
         * @param p1label Label indicating which states are states of Player 1
         * @return std::pair<storm::storage::SparseMatrix<double> New alteranting transition matrix
         * std::vector<IntermediateState>> Vector of newly added intermediate states
         */
        std::pair<storm::storage::SparseMatrix<double>, std::vector<IntermediateState>> createAlternatingMatrix(
            storm::storage::SparseMatrix<double> &transitionMatrix,
            storm::models::sparse::StateLabeling &stateLabeling,
            std::string p1label);

        /** Make a game (represented by POMDP) alternating by adding intermediate states and a new
         * observation to those states.
         */
        void makeAlternating();

        /** Create a Stochastic2PlayerGame from an alternating POMDP based on p1label */
        std::shared_ptr<Stochastic2PlayerGame> createGame();

        /**
         * @brief Get translation of states based on p1label and store them in corresponding translators.
         *
         * @param p1Translator Will contain translation from original states to Player 1 states.
         * @param p2Translator Will contain translation from original states to Player 2 states.
         * @param p1label States with this label are Player 1 states. All other states are Player 2 States.
         */
        void getStateTranslations(ItemTranslator &p1Translator, ItemTranslator &p2Translator, std::string p1label);

        /**
         * @brief Create a new transtiton matrix by splitting the original transition matrix.
         * Call twice with swapped ItemTranslators if you want both matrices.
         *
         * @param mainPlayerTranslator Maps original states to Main Player states (those will be source states in returend matrix)
         * @param otherPlayerTraslator Maps original states to Other Player states (those will be destination states in returned matrix)
         * @return storm::storage::SparseMatrix<double> Transition matrix of mainPlayer
         */
        storm::storage::SparseMatrix<double> createTransitionMatrix(
            ItemTranslator mainPlayerTranslator,
            ItemTranslator otherPlayerTraslator);
    };
}
