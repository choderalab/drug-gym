import rdkit
from dgym.molecule import Molecule
from typing import Optional, List, Union, Any
import itertools
from rdkit.Chem.rdChemReactions import ChemicalReaction
from rdkit import Chem

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
    
    def run(self, reagents):
        reagents = [r.mol if isinstance(r, Molecule) else r for r in reagents]
        return self.template.RunReactants(reagents)

    def annotate_reactants(self, classes):

        assert len(self.reactants) == len(classes)
        for i, _ in enumerate(self.reactants):
            self.reactants[i].SetProp('class', classes[i])
        return self

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
                        product.smiles == Chem.MolToSmiles(o)
                        for o in output[0]
                    )

        return False