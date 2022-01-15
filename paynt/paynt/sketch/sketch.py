import stormpy

from .property import Property, OptimalityProperty, Specification
from .holes import Hole, Holes, DesignSpace
from ..synthesizers.models import MarkovChain
from ..synthesizers.quotient import JaniQuotientContainer, POMDPQuotientContainer

import os
import re
import uuid
from collections import OrderedDict

import logging
logger = logging.getLogger(__name__)


class Sketch:
    
    def __init__(self, sketch_path, properties_path, constant_str):
        self.prism = None

        self.expression_parser = None
        self.constant_map = None
        self.hole_expressions = None

        self.specification = None
        self.design_space = None

        logger.info(f"Loading sketch from {sketch_path}...")

        self.explicit_model = Sketch.read_explicit_mdp(sketch_path)
        if self.is_implicit:
            # template as a prism program
            self.prism, hole_definitions = Sketch.load_sketch(sketch_path)

            self.expression_parser = stormpy.storage.ExpressionParser(self.prism.expression_manager)
            self.expression_parser.set_identifier_mapping(dict())

            logger.info(f"Assuming constant definitions: '{constant_str}' ...")
            constant_map = Sketch.map_constants(self.prism, self.expression_parser, constant_str)
            self.prism = self.prism.define_constants(constant_map)
            self.prism = self.prism.substitute_constants()

            logger.info("Processing hole definitions ...")
            self.prism, holes, self.hole_expressions = Sketch.parse_holes(self.prism, self.expression_parser, hole_definitions)
            logger.info(f"Sketch has {holes.num_holes} holes")
            logger.info(f"Listing hole domains: {holes}")
            logger.info(f"Design space size: {holes.size}")
        else:
            # pomdp in explicit form
            logger.info(f"Loaded model from explicit format.")
            assert self.is_pomdp
            self.prism = None
            self.expression_parser = None
            self.hole_expressions = None
            constant_map = None
            holes = None

        logger.info(f"Loading properties from {properties_path} ...")
        self.specification = Sketch.parse_specification(self.prism, constant_map, properties_path)
        logger.info(f"Found the following constraints: {[str(p) for p in self.specification.constraints]}")
        if self.specification.has_optimality:
            logger.info(f"Found the following optimality property: {self.specification.optimality}")

        if self.is_implicit:
            self.design_space = DesignSpace(holes)
            self.design_space.property_indices = self.specification.all_indices()

        MarkovChain.initialize(self.specification.stormpy_formulae())
        if not self.is_pomdp:
            self.quotient = JaniQuotientContainer(self)
        else:
            self.quotient = POMDPQuotientContainer(self)
            self.quotient.pomdp_manager.set_memory_size(2)
            self.quotient.unfold_partial_memory()

    @property
    def is_implicit(self):
        return self.explicit_model is None

    @property
    def is_mdp(self):
        if self.is_implicit:
            return self.prism.model_type == stormpy.storage.PrismModelType.MDP
        else:
            return self.explicit_model.is_nondeterministic_model and not self.explicit_model.is_partially_observable

    @property
    def is_pomdp(self):
        if self.is_implicit:
            return self.prism.model_type == stormpy.storage.PrismModelType.POMDP
        else:
            return self.explicit_model.is_nondeterministic_model and self.explicit_model.is_partially_observable

    def restrict_prism(self, assignment):
        assert assignment.size == 1
        substitution = {prism.get_constant(hole.hame):self.hole_expressions[hole_index][hole.options[0]] for hole_index,hole in enumerate(assignment)}
        program = self.prism.define_constants(substitution)
        return program

    @classmethod
    def read_explicit_mdp(cls, path):
        model = None
        try:
            builder_options = stormpy.core.DirectEncodingParserOptions()
            builder_options.build_choice_labels = True
            model = stormpy.core._build_sparse_model_from_drn(path, builder_options)
        except:
            return None
        return model


    @classmethod
    def load_sketch(cls, sketch_path):
        # read lines
        with open(sketch_path) as f:
            sketch_lines = f.readlines()

        # strip hole definitions
        hole_re = re.compile(r'^hole\s+(.*?)\s+(.*?)\s+in\s+\{(.*?)\};$')
        sketch_output = []
        hole_definitions = OrderedDict()
        for line in sketch_lines:
            match = hole_re.search(line)
            if match is not None:
                hole_name = match.group(2)
                hole_definitions[hole_name] = match.group(3).replace(" ", "")
                line = f"const {match.group(1)} {hole_name};"
            sketch_output.append(line)

        # store stripped sketch to a temporary file
        tmp_path = sketch_path + str(uuid.uuid4())
        with open(tmp_path, 'w') as f:
            for line in sketch_output:
                print(line, end="", file=f)

        # try to parse temporary sketch and then delete it
        parse_error = False
        try:
            prism = stormpy.parse_prism_program(tmp_path)
        except:
            parse_error = True
        os.remove(tmp_path)
        if parse_error:
            exit()

        return prism, hole_definitions

    @classmethod
    def map_constants(cls, prism, expression_parser, constant_str):
        constant_str = constant_str.replace(" ", "")
        if constant_str == "":
            return dict()
        constant_map = dict()
        constant_definitions = constant_str.split(",")
        
        prism_constants = {c.name:c for c in prism.constants}

        for constant_definition in constant_definitions:
            key_value = constant_definition.split("=")
            if len(key_value) != 2:
                raise ValueError(f"Expected key=value pair, got '{constant_definition}'.")

            expr = expression_parser.parse(key_value[1])
            name = key_value[0]
            constant = prism_constants[name].expression_variable
            constant_map[constant] = expr
        return constant_map


    @classmethod
    def parse_holes(cls, prism, expression_parser, hole_definitions):

        # parse hole definitions
        holes = Holes()
        hole_expressions = []
        for hole_name,definition in hole_definitions.items():
            options = definition.split(",")
            expressions = [expression_parser.parse(o) for o in options]
            hole_expressions.append(expressions)

            options = list(range(len(expressions)))
            option_labels = [str(e) for e in expressions]
            hole = Hole(hole_name, options, option_labels)
            holes.append(hole)

        # check that all undefined constants are indeed the holes
        undefined_constants = [c for c in prism.constants if not c.defined]
        assert len(undefined_constants) == len(holes), "some constants were unspecified"

        # convert single-valued holes to a defined constant
        constant_definitions = {}
        nontrivial_holes = Holes()
        nontrivial_hole_expressions = []
        for hole_index,hole in enumerate(holes):
            if hole.size == 1:
                constant_definitions[prism.get_constant(hole.name).expression_variable] = hole_expressions[hole_index][0]
            else:
                nontrivial_holes.append(hole)
                nontrivial_hole_expressions.append(hole_expressions[hole_index])
        holes = nontrivial_holes
        hole_expressions = nontrivial_hole_expressions
        
        # define these constants in the program
        prism = prism.define_constants(constant_definitions)
        prism = prism.substitute_constants()

        # success
        return prism, holes, hole_expressions

 
    @classmethod
    def parse_specification(cls, prism, constant_map, properites_path):
        # read lines
        lines = []
        with open(properites_path) as file:
            for line in file:
                line = line.replace(" ", "")
                line = line.replace("\n", "")
                if not line or line == "" or line.startswith("//"):
                    continue
                lines.append(line)

        # strip relative error
        lines_properties = ""
        relative_error_re = re.compile(r'^(.*)\{(.*?)\}(=\?.*?$)')
        relative_error_str = None
        for line in lines:
            match = relative_error_re.search(line)
            if match is not None:
                relative_error_str = match.group(2)
                line = match.group(1) + match.group(3)
            lines_properties += line + ";"
        optimality_epsilon = float(relative_error_str) if relative_error_str is not None else 0

        # parse all properties
        properties = []
        optimality_property = None

        if prism is not None:
            props = stormpy.parse_properties_for_prism_program(lines_properties, prism)
        else:
            props = stormpy.parse_properties_without_context(lines_properties)
        for prop in props:
            rf = prop.raw_formula
            assert rf.has_bound != rf.has_optimality_type, "optimizing formula contains a bound or a comparison formula does not"
            if rf.has_bound:
                # comparison formula
                properties.append(prop)
            else:
                # optimality formula
                assert optimality_property is None, "two optimality formulae specified"
                optimality_property = prop

        if constant_map is not None:
            # substitute constants in properties
            for p in properties:
                p.raw_formula.substitute(constant_map)
            if optimality_property is not None:
                optimality_property.raw_formula.substitute(constant_map)

        # wrap properties
        properties = [Property(p) for p in properties]
        if optimality_property is not None:
            optimality_property = OptimalityProperty(optimality_property, optimality_epsilon)

        specification = Specification(properties,optimality_property)
        return specification
    

    