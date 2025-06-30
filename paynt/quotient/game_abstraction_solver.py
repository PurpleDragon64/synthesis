import stormpy
import payntbind

import paynt.quotient.posmg
import paynt.verification.property

# class to solve game abstraction for family of POMDPs
class GameAbstractionSolver():
    def __init__(self, quotient_mdp, state_to_observation, prop, quotient_num_actions, choice_to_action):
        self.quotient_mdp = quotient_mdp
        self.state_to_observation = state_to_observation
        self.quotient_num_actions = quotient_num_actions
        self.choice_to_action = choice_to_action

        self.solution_value = None
        self.solution_state_values = [None for state in range(quotient_mdp.nr_states)]
        self.solution_state_to_player1_action = [None for state in range(quotient_mdp.nr_states)]
        self.solution_state_to_quotient_choice = [None for state in range(quotient_mdp.nr_states)]

        self.game_iterations = None

        self.posmg_specification = self.create_posmg_specification(prop)

    def specify_target_with_label(self, labeling, prop):
        '''
        If the target is specified by a label return this label.
        If the target is specified by an expression, mark all target states with a new
        label and return this label.
        '''
        target = prop.formula.subformula.subformula
        target_label = prop.get_target_label()

        target_is_label = isinstance(target, stormpy.logic.AtomicLabelFormula)
        if target_is_label:
            return target_label

        # target is an expression
        new_target_label = 'goal'

        while labeling.contains_label(new_target_label):
            new_target_label += '_' # add arbitrary character at the end to make new label unique

        labeling.add_label(new_target_label)
        target_states = labeling.get_states(target_label)
        labeling.set_states(new_target_label, target_states)

        return new_target_label

    def create_posmg_specification(self, prop):
        formula_str = prop.formula.__str__()

        target_label = prop.get_target_label()
        new_target_label = self.specify_target_with_label(self.quotient_mdp.labeling, prop)
        if target_label != new_target_label:
            formula_str = formula_str.replace(target_label, '"' + new_target_label + '"')

        optimizing_player = 0 # hard coded. Has to correspond with state_player_indications
        game_fromula_str = f"<<{optimizing_player}>> " + formula_str

        storm_property = stormpy.parse_properties(game_fromula_str)[0]
        property = paynt.verification.property.construct_property(storm_property, 0) # realtive error?
        specification = paynt.verification.property.Specification([property])

        return specification


    def calculate_state_to_player1_action(self, state_to_quotient_choice, choice_to_action, num_actions):
        num_choices = len(choice_to_action)

        state_to_player1_action = []
        for choice in state_to_quotient_choice:
            if choice == num_choices:
                state_to_player1_action.append(num_actions)
            else:
                state_to_player1_action.append(choice_to_action[choice])

        return state_to_player1_action

    def solve_smg(self, quotient_choice_mask):
        # initialize results
        self.solution_value = 0
        for state in range(self.quotient_mdp.nr_states):
            self.solution_state_to_player1_action[state] = self.quotient_num_actions
            self.solution_state_to_quotient_choice[state] = self.quotient_mdp.nr_choices
            self.solution_state_values[state] = 0

            self.game_iterations = 0

        # create game abstraction
        smg_abstraction = payntbind.synthesis.SmgAbstraction(
            self.quotient_mdp,
            self.quotient_num_actions,
            self.choice_to_action,
            quotient_choice_mask)

        # create posmg
        smg_state_observation = []
        for smg_state in range(smg_abstraction.smg.nr_states):
            quotient_state, _ = smg_abstraction.state_to_quotient_state_action[smg_state]
            obs = self.state_to_observation[quotient_state]
            smg_state_observation.append(obs)
        posmg = payntbind.synthesis.posmg_from_smg(smg_abstraction.smg,smg_state_observation)

        # solve posmg
            # the unfolding (if looking for k-FSCs) was already done in PomdpFamilyQuotient init, so set mem to 1 to prevent another unfold
        paynt.quotient.posmg.PosmgQuotient.initial_memory_size = 1
        posmgQuotient = paynt.quotient.posmg.PosmgQuotient(posmg, self.posmg_specification)
        synthesizer = paynt.synthesizer.synthesizer_ar.SynthesizerAR(posmgQuotient)
        assignment = synthesizer.synthesize(print_stats=False)

        # TODO modify for rewards
        #   assignment can be None even for optimality property if the value is infinity
        assert assignment is not None, 'The model contains a non-goal sink state. For such case, the reward model checking returns infinity (=non valid result)'

        self.game_iterations = synthesizer.stat.iterations_game

        # extract results
        state_player_indications = posmgQuotient.posmg_manager.get_state_player_indications()
        choices = posmgQuotient.coloring.selectCompatibleChoices(assignment.family)
        model, game_state_map, game_choice_map = posmgQuotient.restrict_mdp(posmgQuotient.quotient_mdp, choices)
        dtmc = paynt.models.models.Mdp(model)
        result = dtmc.check_specification(self.posmg_specification)

        # fill solution_state_to_player1_action
        for dtmc_state, game_state in enumerate(game_state_map):
            if state_player_indications[game_state] == 0:
                game_choice = game_choice_map[dtmc_state]
                quotient_choice = smg_abstraction.choice_to_quotient_choice[game_choice]
                selected_action = self.choice_to_action[quotient_choice]

                quotient_state, _ = smg_abstraction.state_to_quotient_state_action[game_state]
                self.solution_state_to_player1_action[quotient_state] = selected_action

        # fill solution_state_to_quotient_choices
        for dtmc_state, game_state in enumerate(game_state_map):
            if state_player_indications[game_state] != 1:
                continue
            quotient_state, selected_action = smg_abstraction.state_to_quotient_state_action[game_state]
            if selected_action != self.solution_state_to_player1_action[quotient_state]: # is this necessary? wont these states be removed during restrict mdp?
                continue
            game_choice = game_choice_map[dtmc_state]
            quotient_choice = smg_abstraction.choice_to_quotient_choice[game_choice]
            self.solution_state_to_quotient_choice[quotient_state] = quotient_choice

        # fill solution_value
        self.solution_value = result.optimality_result.result.at(self.quotient_mdp.initial_states[0])


        # fill solution_state_values
        for dtmc_state, game_state in enumerate(game_state_map):
            if state_player_indications[game_state] == 0:
                value = result.optimality_result.result.at(dtmc_state)
                quotient_state, _ = smg_abstraction.state_to_quotient_state_action[game_state]
                self.solution_state_values[quotient_state] = value
