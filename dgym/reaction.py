import dgym as dg
import rdkit
import random
import inspect
import itertools
from dgym.molecule import Molecule
from collections import defaultdict
from collections.abc import Iterator
from typing import Optional, List, Union, Any, Iterable
from .utils import viewable
from rdkit.Chem.rdChemReactions import ChemicalReaction

class Reaction:

    def __init__(
        self,
        template: Union[str, ChemicalReaction],
        metadata: Optional[dict] = None,
        name: Optional[str] = None
    ) -> None:
        """
        Parameters
        ----------
        template : rdkit.Chem.rdChemReactions.ChemicalReaction
            An rdkit reaction template.
        """
        if isinstance(template, str):
            template = rdkit.Chem.AllChem.ReactionFromSmarts(template)

        self.name = name
        self.template = template
        self.products = list(template.GetProducts())
        self.agents = list(template.GetAgents())
        self.reactants = list(template.GetReactants())
        self.metadata = metadata
    
    def run(self, reagents):
        raise NotImplementedError

    def annotate_reactants(self, classes):

        assert len(self.reactants) == len(classes)
        for i, _ in enumerate(self.reactants):
            self.reactants[i].SetProp('class', classes[i])
        return self

    @staticmethod
    def flatten_and_randomize(nested_tuples, randomize=True):
        
        def _flatten(items):
            """Helper function to recursively flatten nested items."""
            for x in items:
                if isinstance(x, tuple):
                    yield from _flatten(x)
                else:
                    yield x

        flattened_items = list(_flatten(nested_tuples))

        if randomize:
            random.shuffle(flattened_items)

        for item in flattened_items:
            yield item
        
    def sanitize(self, mol):
        try:
            rdkit.Chem.SanitizeMol(mol)
            mol.Compute2DCoords()
            return mol
        except:
            pass
    
    def _repr_png_(self):
        return self.template._repr_png_()

    def is_compatible(self, product = None, reactants = None):
        
        # Use reactants from product if none provided
        if reactants is None:
            reactants = product.reactants

        if not product and not reactants:
            return False

        # If the length of reactants matches
        if len(reactants) != len(self.reactants):
            return False
        
        # If the identity of products match
        for reactant_order in itertools.permutations(reactants):
            if output := self.run(reactant_order):
                if any(
                    product.smiles == o.smiles
                    if product else True
                    for o in output
                ):
                    return True

        return False

    @staticmethod    
    def trace(reactants):
        """
        Trace every atom by its reactant of origin.
        
        """
        def _trace_reactant(mol, idx):
            for atom in mol.GetAtoms():
                atom.SetIntProp('reactant_idx', idx)
            return mol

        for idx, reactant in enumerate(reactants):
            reactant.mol = _trace_reactant(reactant.mol, idx)
        
        return reactants
    
    @staticmethod
    def protect(product, reactants):

        # Gather participating atoms
        reacting_atoms = defaultdict(list)
        passenger_atoms = defaultdict(list)
        for atom in product.GetAtoms():
            
            # Handle directly reacting atoms
            if atom.HasProp('old_mapno'):
                
                # RDKit uses 1-index
                reactant_idx = atom.GetIntProp('old_mapno') - 1
                reacting_atoms[reactant_idx].append(atom.GetIntProp('react_atom_idx'))
            
            # Gather passenger atoms
            elif atom.HasProp('reactant_idx'):
                
                reactant_idx = atom.GetIntProp('reactant_idx')
                passenger_atoms[reactant_idx].append(atom.GetIntProp('react_atom_idx'))
            
        # Protect unnecessary atoms
        for idx, reactant in enumerate(reactants):
            for atom in reactant.mol.GetAtoms():
                
                atom_index = atom.GetIdx()
                is_reacting = atom_index in reacting_atoms[idx]
                is_passenger = atom_index in passenger_atoms[idx]
                
                # If atom is unnecessary for reaction
                if not is_reacting and is_passenger:
                    atom.SetProp('_protected', '1')

        return reactants


class LazyReaction(Reaction):
    
    def __init__(
        self,
        template: Union[str, ChemicalReaction],
        metadata: Optional[dict] = None,
        id: Optional[str] = None
    ) -> None:
        """
        Parameters
        ----------
        template : rdkit.Chem.rdChemReactions.ChemicalReaction
            An rdkit reaction template.
        """
        super().__init__(template, metadata, id)
    
    @viewable
    def run(self, reactants, protect=False, strict=False):

        # If any of the reagents are generators
        if any(isinstance(r, Iterable) for r in reactants):
            
            # Convert ordinary reagents to infinite generators
            sequences = [
                itertools.repeat(x)
                if not isinstance(x, Iterable) else x
                for x in reactants
            ]

            # Run reactants lazily
            for combination in zip(*sequences):
                yield from self.run_single_step(combination, protect=protect, strict=strict)
        
        else:
            yield from self.run_single_step(reactants, protect=protect, strict=strict)

    def run_single_step(self, reactants, protect=False, strict=False):

        # check if reaction is even going to run
        if len(reactants) != len(self.reactants) \
            or not all(reactants):
            return ()

        if protect:
            reactants = self.trace(reactants)
        
        # Convert to RDKit mols to run reaction
        mols = [r.mol if isinstance(r, Molecule) else r for r in reactants]
        
        if strict:
            output = self.template.RunReactants(mols)
        else:
            output = (self.template.RunReactants(mols_)
                      for mols_ in itertools.permutations(mols))
        
        yield from self.parse_output(output, reactants, protect=protect)
        
    def parse_output(self, output, reactants, protect=False):
        
        output = self.flatten_and_randomize(output)
        for product in output:
        
            if product := self.sanitize(product):
                
                if protect:
                    reactants = self.protect(product, reactants)
                
                yield Molecule(product, reaction = self, reactants = reactants)
        
            else:
                continue