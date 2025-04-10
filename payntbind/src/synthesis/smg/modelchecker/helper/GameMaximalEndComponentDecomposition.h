#pragma once

/* 
 * code in this file was taken from TEMPEST (https://github.com/PrangerStefan/TempestSynthesis)
 */

#include <storm/storage/Decomposition.h>
#include <storm/storage/MaximalEndComponent.h>
#include <storm/models/sparse/NondeterministicModel.h>

namespace synthesis  {

    /*!
        * This class represents the decomposition of a stochastic multiplayer game into its (irreducible) maximal end components.
        */
    template <typename ValueType>
    class GameMaximalEndComponentDecomposition : public storm::storage::Decomposition<storm::storage::MaximalEndComponent> {
    public:
        /*
            * Creates an empty MEC decomposition.
            */
        GameMaximalEndComponentDecomposition();

        /*
            * Creates an MEC decomposition of the given model.
            *
            * @param model The model to decompose into MECs.
            */
        template <typename RewardModelType = storm::models::sparse::StandardRewardModel<ValueType>>
        GameMaximalEndComponentDecomposition(storm::models::sparse::NondeterministicModel<ValueType, RewardModelType> const& model);

        /*
            * Creates an MEC decomposition of the given model (represented by a row-grouped matrix).
            *
            * @param transitionMatrix The transition relation of model to decompose into MECs.
            * @param backwardTransition The reversed transition relation.
            */
        GameMaximalEndComponentDecomposition(storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions);

        /*
            * Creates an MEC decomposition of the given subsystem of given model (represented by a row-grouped matrix).
            *
            * @param transitionMatrix The transition relation of model to decompose into MECs.
            * @param backwardTransition The reversed transition relation.
            * @param states The states of the subsystem to decompose.
            */
        GameMaximalEndComponentDecomposition(storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::storage::BitVector const& states);

        /*
            * Creates an MEC decomposition of the given subsystem of given model (represented by a row-grouped matrix).
            *
            * @param transitionMatrix The transition relation of model to decompose into MECs.
            * @param backwardTransition The reversed transition relation.
            * @param states The states of the subsystem to decompose.
            * @param choices The choices of the subsystem to decompose.
            */
        GameMaximalEndComponentDecomposition(storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> const& backwardTransitions, storm::storage::BitVector const& states, storm::storage::BitVector const& choices);

        /*!
            * Creates an MEC decomposition of the given subsystem in the given model.
            *
            * @param model The model whose subsystem to decompose into MECs.
            * @param states The states of the subsystem to decompose.
            */
        GameMaximalEndComponentDecomposition(storm::models::sparse::NondeterministicModel<ValueType> const& model, storm::storage::BitVector const& states);

        /*!
            * Creates an MEC decomposition by copying the contents of the given MEC decomposition.
            *
            * @param other The MEC decomposition to copy.
            */
        GameMaximalEndComponentDecomposition(GameMaximalEndComponentDecomposition const& other);

        /*!
            * Assigns the contents of the given MEC decomposition to the current one by copying its contents.
            *
            * @param other The MEC decomposition from which to copy-assign.
            */
        GameMaximalEndComponentDecomposition& operator=(GameMaximalEndComponentDecomposition const& other);

        /*!
            * Creates an MEC decomposition by moving the contents of the given MEC decomposition.
            *
            * @param other The MEC decomposition to move.
            */
        GameMaximalEndComponentDecomposition(GameMaximalEndComponentDecomposition&& other);

        /*!
            * Assigns the contents of the given MEC decomposition to the current one by moving its contents.
            *
            * @param other The MEC decomposition from which to move-assign.
            */
        GameMaximalEndComponentDecomposition& operator=(GameMaximalEndComponentDecomposition&& other);

    private:
        /*!
            * Performs the actual decomposition of the given subsystem in the given model into MECs. As a side-effect
            * this stores the MECs found in the current decomposition.
            *
            * @param transitionMatrix The transition matrix representing the system whose subsystem to decompose into MECs.
            * @param backwardTransitions The reversed transition relation.
            * @param states The states of the subsystem to decompose.
            * @param choices The choices of the subsystem to decompose.
            */
        void performGameMaximalEndComponentDecomposition(storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> backwardTransitions, storm::storage::BitVector const* states = nullptr, storm::storage::BitVector const* choices = nullptr);
        void singleMEC(storm::storage::SparseMatrix<ValueType> const& transitionMatrix, storm::storage::SparseMatrix<ValueType> backwardTransitions, storm::storage::BitVector const* states = nullptr, storm::storage::BitVector const* choices = nullptr);
    };
}
