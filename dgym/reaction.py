import rdkit
import random
import inspect
import itertools
from dgym.molecule import Molecule
from typing import Optional, List, Union, Any
from rdkit.Chem.rdChemReactions import ChemicalReaction

class Reaction:

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
        if isinstance(template, str):
            template = rdkit.Chem.AllChem.ReactionFromSmarts(template)

        self.id = id
        self.template = template
        self.products = list(template.GetProducts())
        self.agents = list(template.GetAgents())
        self.reactants = list(template.GetReactants())
        self.metadata = metadata
    
    def run(self, reagents) -> List[Molecule]:
        raise NotImplementedError

    def annotate_reactants(self, classes):

        assert len(self.reactants) == len(classes)
        for i, _ in enumerate(self.reactants):
            self.reactants[i].SetProp('class', classes[i])
        return self

    @staticmethod
    def flatten_and_randomize(nested_tuples, randomize=True):
        
        flattened_items = []
        for item in nested_tuples:
            if isinstance(item, tuple):
                flattened_items.extend(item)
            else:
                flattened_items.append(item)

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

        # If neither product nor reactant are provided
        if not product and not reactants:
            return False

        # If the length of reactants matches
        if len(reactants) != len(self.reactants):
            return False
        
        # If the identity of products match
        for reactant_order in itertools.permutations(reactants):
            if output := self.run(reactant_order):
                if not product:
                    return True
                elif product:
                    return any(
                        product.smiles == o.smiles
                        for o in output
                    )

        return False


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
    
    def run(self, reagents, sanitize=True):
        
        # If any of the reagents are generators
        if any(inspect.isgenerator(r) for r in reagents):
            
            # Convert ordinary reagents to infinite generators
            sequences = [
                itertools.repeat(x)
                if not inspect.isgenerator(x) else x
                for x in reagents
            ]

            # Run reactants lazily
            for combination in zip(*sequences):
                yield from self.run_single_step(combination)
        
        else:
            yield from self.run_single_step(reagents)
        
    def run_single_step(self, reagents):
        mols = [r.mol if isinstance(r, Molecule) else r for r in reagents]
        output = self.template.RunReactants(mols)
        yield from self.parse_output(output, reagents)
        
    def parse_output(self, output, reactants):
        output = self.flatten_and_randomize(output)
        cache = set()
        for product in output:
            if product := self.sanitize(product):
                yield Molecule(product, reaction = self, reactants = reactants)
            else:
                continue
