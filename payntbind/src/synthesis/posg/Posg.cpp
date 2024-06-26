#include "Posg.h"

#include <stdio.h>

namespace synthesis {
    
    Posg::Posg(storm::models::sparse::Pomdp<double> quotient, std::string p1label)
    : pomdp(quotient) {
        this->game = createGame(p1label);
    }

    std::shared_ptr<Stochastic2PlayerGame> Posg::createGame(std::string p1label)
    {
        ItemTranslator stateToP1State;
        ItemTranslator stateToP2State;
        getStateTranslations(stateToP1State, stateToP2State, p1label);

        auto player1Matrix = createTransitionMatrix(stateToP1State, stateToP2State);
        auto player2Matrix = createTransitionMatrix(stateToP2State, stateToP1State);
        auto stateLabeling = pomdp.getStateLabeling();

        return std::make_shared<Stochastic2PlayerGame>(player1Matrix, player2Matrix, stateLabeling);
    }
    
    void Posg::getStateTranslations(ItemTranslator &p1Translator, ItemTranslator &p2Translator, std::string p1label)
    {
        auto numberOfStates = pomdp.getNumberOfStates();
        auto stateLabeling = pomdp.getStateLabeling();

        p1Translator = ItemTranslator(numberOfStates);
        p2Translator = ItemTranslator(numberOfStates);

        for (uint64_t state = 0; state < numberOfStates; state++)
        {
            if (stateLabeling.getStateHasLabel(p1label, state))
            {
                p1Translator.translate(state);
            }
            else
            {
                p2Translator.translate(state);
            }
        }
    }
    
    storm::storage::SparseMatrix<double> Posg::createTransitionMatrix(
        ItemTranslator mainPlayerTranslator,
        ItemTranslator otherPlayerTraslator
        )
    {        
        
        // ----- Testing matrix and translators ------
        // note: comment out transition matrix from pomdp
        // storm::storage::SparseMatrixBuilder<double> testBuilder(6, 4, 8, false, true);
        // testBuilder.newRowGroup(0);
        // testBuilder.addNextValue(0, 1, 10);
        // testBuilder.addNextValue(1, 1, 20);
        // testBuilder.addNextValue(1, 2, 30);
        // testBuilder.newRowGroup(2);
        // testBuilder.addNextValue(2, 3, 40);
        // testBuilder.addNextValue(3, 0, 50);
        // testBuilder.addNextValue(3, 3, 60);
        // testBuilder.newRowGroup(4);
        // testBuilder.addNextValue(4, 0, 70);
        // testBuilder.newRowGroup(5);
        // testBuilder.addNextValue(5, 2, 80);
        // auto transitionMatrix = testBuilder.build();

        // ItemTranslator stateToP1State(4);
        // ItemTranslator stateToP2State(4);
        // stateToP1State.translate(1);
        // stateToP1State.translate(2);
        // stateToP2State.translate(0);
        // stateToP2State.translate(3);

        // mainPlayerTranslator = stateToP1State;
        // otherPlayerTraslator = stateToP2State;
        
        auto transitionMatrix = pomdp.getTransitionMatrix();
        auto firstActionOfStates = transitionMatrix.getRowGroupIndices();

        storm::storage::SparseMatrixBuilder<double> matrixBuilder(0, 0, 0, false, true);
        uint64_t rowCount = 0;

        auto mainPlayerStateCount = mainPlayerTranslator.numTranslations();
        auto otherPlayerStateCount = otherPlayerTraslator.numTranslations();

        for (uint64_t mainPlayerState = 0; mainPlayerState < mainPlayerStateCount; mainPlayerState++)
        {
            auto origState = mainPlayerTranslator.retrieve(mainPlayerState);

            matrixBuilder.newRowGroup(rowCount);
            for (uint64_t action = firstActionOfStates[origState]; action < firstActionOfStates[origState + 1]; action++)
            {
                auto row = transitionMatrix.getRow(action);
                for (auto entry = row.begin(); entry != row.end(); entry++)
                {
                    auto destinationState = (*entry).getColumn();
                    auto probability = (*entry).getValue();

                    auto otherPlayerState = otherPlayerTraslator.translate(destinationState);
                    matrixBuilder.addNextValue(rowCount, otherPlayerState, probability);
                }
                rowCount++;
            }
        }
        // should be satisfied for alteranting games
        //assert(otherPlayerStateCount == otherPlayerTraslator.numTranslations());

        auto matrix = matrixBuilder.build();

        return matrix;
    }
}