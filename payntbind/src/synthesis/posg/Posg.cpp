#include "Posg.h"

#include <stdio.h>

namespace synthesis {

    Posg::Posg(storm::models::sparse::Pomdp<double> quotient, std::string p1label)
    : pomdp(quotient), p1label(p1label) {

        makeAlternating();
        this->game = createGame();
    }

    // taken from GameAbstractionSolver.cpp for debugging purposes
    void print_matrix(storm::storage::SparseMatrix<double> matrix) {
        auto const& row_group_indices = matrix.getRowGroupIndices();
        for(uint64_t state=0; state < matrix.getRowGroupCount(); state++) {
            std::cout << "state " << state << ": " << std::endl;
            for(uint64_t row=row_group_indices[state]; row<row_group_indices[state+1]; row++) {
                for(auto const &entry: matrix.getRow(row)) {
                    std::cout << state << "-> "  << entry.getColumn() << " ["  << entry.getValue() << "];";
                }
                std::cout << std::endl;
            }
        }
        std::cout << "-----" << std::endl;
    }

    std::vector<IntermediateState>::iterator Posg::findIntermediateState(std::vector<IntermediateState> &states, uint64_t dstState)
    {
        for (auto state = states.begin(); state != states.end(); state++)
        {
            if ((*state).dstState == dstState)
            {
                return state;
            }
        }

        return states.end();
    }

    std::pair<storm::storage::SparseMatrix<double>, std::vector<IntermediateState>> Posg::createAlternatingMatrix(
            storm::storage::SparseMatrix<double> &transitionMatrix,
            storm::models::sparse::StateLabeling &stateLabeling,
            std::string p1label)
    {
        auto numberOfStates = transitionMatrix.getColumnCount();
        auto firstActionOfStates = transitionMatrix.getRowGroupIndices();

        std::vector<IntermediateState> intermediateStates;
        uint64_t newStateNumber = numberOfStates;

        storm::storage::SparseMatrixBuilder<double> matrixBuilder(0, 0, 0, false, true);
        uint64_t rowCount = 0;

        // traverse and copy original matrix to new matrix
        // redirect not alternating transitions to new states and store those states in intermediateStates vector
        for (uint64_t srcState = 0; srcState < numberOfStates; srcState++)
        {
            bool isSrcP1 = stateLabeling.getStateHasLabel(p1label, srcState);

            matrixBuilder.newRowGroup(rowCount);
            for (uint64_t action = firstActionOfStates[srcState]; action < firstActionOfStates[srcState+1]; action++)
            {
                auto row = transitionMatrix.getRow(action);
                for (auto value = row.begin(); value != row.end(); value++) // todo for ( : )
                {
                    auto dstState = (*value).getColumn();
                    auto probability = (*value).getValue();

                    bool isDstP1 = stateLabeling.getStateHasLabel(p1label, dstState);
                    bool notAlternating = isSrcP1 == isDstP1;
                    if (notAlternating)
                    {
                        auto intermediateStateIter = findIntermediateState(intermediateStates, dstState);
                        if (intermediateStateIter == intermediateStates.end())
                        {
                            IntermediateState newState = {
                                .stateNumber = newStateNumber,
                                .dstState = dstState,
                                .isP1State = !isDstP1};
                            intermediateStates.push_back(newState);

                            dstState = newStateNumber++;
                        }
                        else
                        {
                            dstState = (*intermediateStateIter).stateNumber;
                        }
                    }

                    matrixBuilder.addNextValue(rowCount, dstState, probability);
                }
                rowCount++;
            }
        }

        // add intermediate states at the end of the new transition matrix
        for (auto state : intermediateStates)
        {
            matrixBuilder.newRowGroup(rowCount);
            matrixBuilder.addNextValue(rowCount, state.dstState, 1);
            rowCount++;
        }

        auto newMat = matrixBuilder.build();
        // for debugging
        // print_matrix(newMat);

        return {newMat, intermediateStates};
    }

    void Posg::makeAlternating()
    {
        // // testing data
        // storm::storage::SparseMatrixBuilder<double> testingBuilder(0, 0, 0, false, true);
        // testingBuilder.newRowGroup(0);
        // testingBuilder.addNextValue(0, 0, 10);
        // testingBuilder.addNextValue(0, 2, 20);
        // testingBuilder.newRowGroup(1);
        // testingBuilder.addNextValue(1, 0, 30);
        // testingBuilder.addNextValue(2, 3, 40);
        // testingBuilder.newRowGroup(3);
        // testingBuilder.addNextValue(3, 3, 50);
        // testingBuilder.addNextValue(4, 2, 60);
        // testingBuilder.newRowGroup(5);
        // testingBuilder.addNextValue(5, 0, 70);

        // auto oldMatrix = testingBuilder.build();

        // print_matrix(oldMatrix);

        // auto numberOfStates = oldMatrix.getColumnCount();
        // // auto firstActionOfStates = oldMatrix.getRowGroupIndices();
        // storm::models::sparse::StateLabeling oldStateLabeling(numberOfStates);
        // oldStateLabeling.addLabel(p1label);
        // oldStateLabeling.addLabelToState(p1label, 0);
        // oldStateLabeling.addLabelToState(p1label, 1);
        // oldStateLabeling.addLabel("goul");
        // oldStateLabeling.addLabelToState("goul", 2);

        // auto newObservation = 2;
        // std::__1::vector<uint32_t> observations = {0, 1, 1, 0};

        // // end of testing data

        // create matrix
        auto oldMatrix = pomdp.getTransitionMatrix();
        auto oldStateLabeling = pomdp.getStateLabeling();
        auto retVal = createAlternatingMatrix(oldMatrix, oldStateLabeling, p1label);
        auto newMatrix = retVal.first;
        auto intermediateStates = retVal.second;

        // create labeling
        auto newStateCount = newMatrix.getColumnCount();
        storm::models::sparse::StateLabeling newLabeling(newStateCount);

        // copy original labels
        auto oldLabels = oldStateLabeling.getLabels();
        for (auto label : oldLabels)
        {
            newLabeling.addLabel(label);
        }
        auto oldStateCount = oldMatrix.getColumnCount();
        for (uint64_t state = 0; state < oldStateCount; state++)
        {
            auto labels = oldStateLabeling.getLabelsOfState(state);
            for (auto label : labels)
            {
                newLabeling.addLabelToState(label, state);
            }
        }
        // add labels to new states
        for (auto state : intermediateStates)
        {
            if (state.isP1State)
            {
                newLabeling.addLabelToState(p1label, state.stateNumber);
            }
        }

        // get reward models
        auto rewardModels = pomdp.getRewardModels();

        // add new observation to intermediate states
        auto newObservation = pomdp.getNrObservations();
        auto observations = pomdp.getObservations();
        for (uint64_t _ = 0; _ < intermediateStates.size(); _++)
        {
            observations.push_back(newObservation);
        }

        // todo observation valuations
        //auto observationValuations = pomdp.getObservationValuations();

        storm::storage::sparse::ModelComponents<double> components(
            std::move(newMatrix),
            std::move(newLabeling),
            std::move(rewardModels)
        );
        components.observabilityClasses = observations;
        //components.observationValuations = observationValuations;

        storm::models::sparse::Pomdp<double> newPomdp(components, true); // isCanonic???

        this->pomdp = newPomdp;
    }

    std::shared_ptr<Stochastic2PlayerGame> Posg::createGame()
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
        // ------------------------------------------------

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
        assert(otherPlayerStateCount == otherPlayerTraslator.numTranslations());

        auto matrix = matrixBuilder.build();

        return matrix;
    }
}