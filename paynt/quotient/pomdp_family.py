import stormpy
import payntbind

import paynt.models.models
import paynt.quotient.quotient
import paynt.quotient.mdp_family
import paynt.quotient.game_abstraction_solver


import logging
logger = logging.getLogger(__name__)

class SubPomdp:
    '''
    Simple container for a (sub-)POMDP created from the quotient.
    '''
    def __init__(self, model, quotient, quotient_state_map, quotient_choice_map):
        # the Stormpy POMDP
        self.model = model
        # POMDP family quotient from which this POMDP was constructed
        # self.quotient = quotient
        # for each state of the POMDP, a state in the quotient
        self.quotient_state_map = quotient_state_map
        # for each choice of the POMDP, a choice in the quotient
        self.quotient_choice_map = quotient_choice_map


class PomdpFamilyQuotient(paynt.quotient.mdp_family.MdpFamilyQuotient):
    MAX_MEMORY = 1

    def __init__(self, quotient_mdp, family, coloring, specification, obs_evaluator):
        self.obs_evaluator = obs_evaluator
        self.unfolded_state_to_observation = None

        # for each memory size (1 ... MAX_MEMORY) a choice mask enabling corresponding memory updates in the quotient mdp
        self.restricted_choices = None

        if self.MAX_MEMORY > 1:
            quotient_mdp, self.unfolded_state_to_observation, coloring, self.restricted_choices = self.unfold_quotient(
                quotient_mdp, self.state_to_observation, specification, self.MAX_MEMORY, family, coloring)
        else: # max memory is 1
            self.unfolded_state_to_observation = self.state_to_observation
            self.restricted_choices = {self.MAX_MEMORY: stormpy.storage.BitVector(quotient_mdp.nr_choices, True)}

        super().__init__(quotient_mdp = quotient_mdp, family = family, coloring = coloring, specification = specification)

        # for each observation, a list of actions (indices) available
        self.observation_to_actions = None
        # POMDP manager used for unfolding the memory model into the quotient POMDP
        self.fsc_unfolder = None

        # identify actions available at each observation
        self.observation_to_actions = [None] * self.num_observations_unfolded
        state_to_observation = self.unfolded_state_to_observation
        for state,available_actions in enumerate(self.state_to_actions):
            obs = state_to_observation[state]
            if self.observation_to_actions[obs] is not None:
                assert self.observation_to_actions[obs] == available_actions,\
                    f"two states in observation cla ss {obs} differ in available actions"
                continue
            self.observation_to_actions[obs] = available_actions


    @property
    def num_observations(self):
        return self.obs_evaluator.num_obs_classes

    @property
    def num_observations_unfolded(self):
        return max(self.unfolded_state_to_observation) + 1

    @property
    def state_to_observation(self):
        return self.obs_evaluator.state_to_obs_class

    def observation_is_trivial(self, obs):
        return len(self.observation_to_actions[obs])==1

    # construct the quotient for the family
    # the family is a intersection of policy tree family and memory family
    def build(self, family):
        # TODO decide which memory size to use
        memory_size = self.MAX_MEMORY

        member_selection_choices = self.coloring.selectCompatibleChoices(family.family)
        memory_selection_choices = self.restricted_choices[memory_size]
        choices = member_selection_choices & memory_selection_choices
        family.mdp = self.build_from_choice_mask(choices)
        family.selected_choices = choices
        family.mdp.family = family

    # disallow all choices which update memory to more than max_memory
    # choices - bit vector to be restricted
    # choice_memory_update - memory update associated with each choice
    # max_memory - highest allowed memory update
    def restrict_choices(self, choices, choice_memory_update, max_memory):
        permitted_memory_updates = [memory for memory in range(max_memory)]

        for choice in range(len(choices)):
            memory_update = choice_memory_update[choice]
            if memory_update not in permitted_memory_updates:
                choices[choice] = False

        return choices

    # for memory = 1..max_memory calculate corresponding restricted choices
    def calculate_restricted_choices(self, choice_memory_update, max_memory):
        mem_restricted_choices = {}
        choice_count = len(choice_memory_update)

        for memory in range(1, max_memory):
            all_choices = stormpy.storage.BitVector(choice_count,  True)
            restricted_choices = self.restrict_choices(all_choices, choice_memory_update, memory)
            mem_restricted_choices[memory] = restricted_choices

        # for max memory, all choices are permitted
        mem_restricted_choices[max_memory] = stormpy.storage.BitVector(choice_count, True)

        return mem_restricted_choices

    # unfold the quotient pomdp (represented as mdp + observation map) to maximum memory
    def unfold_quotient(self, quotient_mdp, state_to_observation, specification, max_memory, family, coloring):
        new_quotient_mdp = None
        new_state_to_observation = None
        new_coloring = None
        restricted_choices = None

        # create pomdp manager
        pomdp = self.pomdp_from_mdp(quotient_mdp, state_to_observation)
        pomdp_manager = payntbind.synthesis.PomdpManager(pomdp, False)


        # set imperfect memory size for each observation
        observation_count = max(state_to_observation)+1

        observation_states = [0 for obs in range(observation_count)]
        for state in range(pomdp.nr_states):
            obs = pomdp.get_observation(state)
            observation_states[obs] += 1

        for obs in range(observation_count):
            memory_size = max_memory if observation_states[obs] > 1 else 1
            pomdp_manager.set_observation_memory_size(obs, memory_size)

        # create unfolded quotient
        unfolded_quotient = pomdp_manager.construct_mdp()
        state_prototypes = pomdp_manager.state_prototype
        choice_prototypes = pomdp_manager.row_prototype

        # Update state to observation mapping. Create new observations for states with memory>1
        state_memory = pomdp_manager.state_memory
        new_state_to_observation = []
        for state in range(unfolded_quotient.nr_states):
            state_prototype = state_prototypes[state]
            observation = state_to_observation[state_prototype]
            memory = state_memory[state]
            new_observation = memory * observation_count + observation
            new_state_to_observation.append(new_observation)

        # Update choice labeling. Append the memory update to each choice label.
        # (choice labeling of a model cannot be modified. therefore a new model is created)
        transition_matrix = unfolded_quotient.transition_matrix
        state_labeling = unfolded_quotient.labeling
        reward_models = unfolded_quotient.reward_models
        components = stormpy.SparseModelComponents(
            transition_matrix=transition_matrix,
            state_labeling=state_labeling,
            reward_models=reward_models)

        choice_memory_update = pomdp_manager.row_memory_option
        choice_labeling = stormpy.storage.ChoiceLabeling(unfolded_quotient.nr_choices)
        for choice in range(unfolded_quotient.nr_choices):
            memory_update = choice_memory_update[choice]
            choice_prototype = choice_prototypes[choice]
            labels = quotient_mdp.choice_labeling.get_labels_of_choice(choice_prototype)

            for label in labels:
                new_label = f'{label}_{memory_update}'
                if not choice_labeling.contains_label(new_label):
                    choice_labeling.add_label(new_label)
                choice_labeling.add_label_to_choice(new_label, choice)
        components.choice_labeling = choice_labeling

        new_quotient_mdp = stormpy.storage.SparseMdp(components)

        # Update coloring.
        choice_to_hole_options = coloring.getChoiceToAssignment()
        new_choice_to_hole_options = []
        for choice in range(new_quotient_mdp.nr_choices):
            choice_prototype = choice_prototypes[choice]
            hole_options = choice_to_hole_options[choice_prototype]
            new_choice_to_hole_options.append(hole_options)

        new_coloring = payntbind.synthesis.Coloring(
            family.family, new_quotient_mdp.nondeterministic_choice_indices, new_choice_to_hole_options)

        restricted_choices = self.calculate_restricted_choices(choice_memory_update, max_memory)

        return new_quotient_mdp, new_state_to_observation, new_coloring, restricted_choices


    def pomdp_from_mdp(self, mdp, observability_classes):
        transition_matrix = mdp.transition_matrix
        state_labeling = mdp.labeling
        reward_models = mdp.reward_models
        components = stormpy.SparseModelComponents(
            transition_matrix=transition_matrix,
            state_labeling=state_labeling,
            reward_models=reward_models)

        components.observability_classes=observability_classes

        if mdp.has_choice_labeling():
            components.choice_labeling = mdp.choice_labeling

        return stormpy.storage.SparsePomdp(components)

    def build_game_abstraction_solver(self, prop):
        return paynt.quotient.game_abstraction_solver.GameAbstractionSolver(self.quotient_mdp, self.unfolded_state_to_observation, prop, len(self.action_labels), self.choice_to_action)

    # mdp - SubMdp, represents one pomdp from the pomdp family
    # pomodp_quotient - quotient used for pomdp synthesis
    # assignment - result of pomdp synthesis
    def assignment_to_policy(self, mdp, pomdp_quotient, assignment):
        policy = self.empty_policy()

        choices = pomdp_quotient.coloring.selectCompatibleChoices(assignment.family)
        dtmc, mdp_state_map, mdp_choice_map = self.restrict_mdp(mdp.model, choices)

        for dtmc_state, mdp_state in enumerate(mdp_state_map):
            quotient_state = mdp.quotient_state_map[mdp_state]

            mdp_choice = mdp_choice_map[dtmc_state]
            quotient_choice = mdp.quotient_choice_map[mdp_choice]
            quotient_action = self.choice_to_action[quotient_choice]

            policy[quotient_state] = quotient_action

        return policy

################################################################################

    def build_pomdp(self, family):
        ''' Construct the sub-POMDP from the given hole assignment. '''
        assert family.size == 1, "expecting family of size 1"
        choices = self.coloring.selectCompatibleChoices(family.family)
        mdp,state_map,choice_map = self.restrict_quotient(choices)
        pomdp = self.obs_evaluator.add_observations_to_submdp(mdp,state_map)
        # for state,quotient_state in enumerate(state_map):
        #     assert pomdp.observations[state] == self.state_to_observation[quotient_state]
        # assert pomdp.nr_observations == self.num_observations
        return SubPomdp(pomdp,self,state_map,choice_map)


    def build_dtmc_sketch(self, fsc, negate_specification=True):
        '''
        Construct the family of DTMCs representing the execution of the given FSC in different environments.
        '''

        # create the product
        fsc.check_action_function(self.observation_to_actions)


        self.fsc_unfolder = payntbind.synthesis.FscUnfolder(
            self.quotient_mdp, self.state_to_observation, self.num_actions, self.choice_to_action
        )
        self.fsc_unfolder.apply_fsc(fsc.action_function, fsc.update_function)
        product = self.fsc_unfolder.product
        product_choice_to_choice = self.fsc_unfolder.product_choice_to_choice

        # the product inherits the design space
        product_family = self.family.copy()

        # the choices of the product inherit colors of the quotient
        product_choice_to_hole_options = []
        quotient_num_choces = self.quotient_mdp.nr_choices
        choice_to_hole_assignment = self.coloring.getChoiceToAssignment()
        for product_choice in range(product.nr_choices):
            choice = product_choice_to_choice[product_choice]
            if choice == quotient_num_choces:
                hole_options = []
            else:
                hole_options = [(hole,option) for hole,option in choice_to_hole_assignment[choice]]
            product_choice_to_hole_options.append(hole_options)
        product_coloring = payntbind.synthesis.Coloring(product_family.family, product.nondeterministic_choice_indices, product_choice_to_hole_options)

        # copy the specification
        product_specification = self.specification.copy()
        if negate_specification:
            product_specification = product_specification.negate()

        dtmc_sketch = paynt.quotient.quotient.Quotient(product, product_family, product_coloring, product_specification)
        return dtmc_sketch




### LEGACY CODE, NOT UP-TO-DATE ###

    def compute_qvalues_for_product_submdp(self, product_submdp : paynt.models.models.SubMdp):
        '''
        Given an MDP obtained after applying FSC to a family of POMDPs, compute for each state s, (reachable)
        memory node n, and action a, the Q-value Q((s,n),a).
        :note it is assumed that a randomized FSC was used
        :note it is assumed the provided DTMC sketch is the one obtained after the last unfolding, i.e. no other DTMC
            sketch was constructed afterwards
        :return a dictionary mapping (s,n,a) to Q((s,n),a)
        '''
        assert isinstance(self.product_pomdp_fsc, payntbind.synthesis.ProductPomdpRandomizedFsc), \
            "to compute Q-values, unfolder for randomized FSC must have been used"

        # model check
        prop = self.get_property()
        result = product_submdp.model_check_property(prop)
        product_state_sub_to_value = result.result.get_values()

        # map states of a sub-MDP to the states of the quotient-MDP to the state-memory pairs of the quotient POMDP
        # map states values to the resulting map
        product_state_to_state_memory_action = self.product_pomdp_fsc.product_state_to_state_memory_action.copy()
        state_memory_action_to_value = {}
        invalid_action = self.num_actions
        for product_state_sub in range(product_submdp.model.nr_states):
            product_state = product_submdp.quotient_state_map[product_state_sub]
            state,memory_action = product_state_to_state_memory_action[product_state]
            memory,action = memory_action
            if action == invalid_action:
                continue
            value = product_state_sub_to_value[product_state_sub]
            state_memory_action_to_value[(state,memory,action)] = value
        return state_memory_action_to_value


    def translate_path_to_trace(self, dtmc, path):
        invalid_choice = self.quotient_mdp.nr_choices
        trace = []
        for dtmc_state in path:
            product_choice = dtmc.quotient_choice_map[dtmc_state]
            choice = self.product_pomdp_fsc.product_choice_to_choice[product_choice]
            if choice == invalid_choice:
                # randomized FSC: we are in the intermediate state, move on to the next one
                continue

            product_state = dtmc.quotient_state_map[dtmc_state]
            state = self.product_pomdp_fsc.product_state_to_state[product_state]
            obs = self.state_to_observation[state]
            action = self.choice_to_action[choice]
            trace.append( (obs,action) )

        # in the last state, we remove the action since it was not actually used
        trace[-1] = (obs,None)

        # sanity check
        for obs,action in trace[:-1]:
            assert action in self.observation_to_actions[obs], "invalid trace"

        return trace


    def compute_witnessing_traces(self, dtmc_sketch, satisfying_assignment, num_traces, trace_max_length):
        '''
        Generate witnessing paths in the DTMC induced by the DTMC sketch and a satisfying assignment.
        If the set of target states is not reachable, then random traces are simulated
        :return a list of state-action pairs
        :note the method assumes that the DTMC sketch is the one that was last constructed using build_dtmc_sketch()
        '''
        fsc_is_randomized = isinstance(self.product_pomdp_fsc, payntbind.synthesis.ProductPomdpRandomizedFsc)
        if fsc_is_randomized:
            # double the trace length to account for intermediate states
            trace_max_length *= 2

        # logger.debug("constructing witnesses...")
        dtmc = dtmc_sketch.build_assignment(satisfying_assignment)

        # assuming a single probability reachability property
        spec = dtmc_sketch.specification
        assert spec.num_properties == 1, "expecting a single property"
        prop = spec.all_properties()[0]
        if prop.is_reward:
            logger.warning("WARNING: specification is a reward property, but generated traces \
                will be based on transition probabilities")

        target_label = self.extract_target_label()
        target_states = dtmc.model.labeling.get_states(target_label)

        traces = []
        if target_states.number_of_set_bits()==0:
            # target is not reachable: use Stormpy simulator to obtain some random walk in a DTMC
            logger.debug("target is not reachable, generating random traces...")
            simulator = stormpy.core._DiscreteTimeSparseModelSimulatorDouble(dtmc.model)
            for _ in range(num_traces):
                simulator.reset_to_initial_state()
                path = [simulator.get_current_state()]
                for _ in range(trace_max_length):
                    success = simulator.random_step()
                    if not success:
                        break
                    path.append(simulator.get_current_state())
                trace = self.translate_path_to_trace(dtmc,path)
                traces.append(trace)
        else:
            # target is reachable: use KSP
            logger.debug("target is reachable, computing shortest paths to...")
            if prop.minimizing:
                logger.debug("...target states")
                shortest_paths_generator = stormpy.utility.ShortestPathsGenerator(dtmc.model, target_label)
            else:
                logger.debug("...BSCCs from which target states are unreachable...")
                phi_states = stormpy.storage.BitVector(dtmc.model.nr_states,True)
                states0,_ = stormpy.core._compute_prob01states_double(dtmc.model,phi_states,target_states)
                shortest_paths_generator = stormpy.utility.ShortestPathsGenerator(dtmc.model, states0)
            for k in range(1,num_traces+1):
                path = shortest_paths_generator.get_path_as_list(k)
                path.reverse()
                trace = self.translate_path_to_trace(dtmc,path)
                traces.append(trace)
        return traces
