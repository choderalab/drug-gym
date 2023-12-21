import torch
import rdkit
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
    
    def run(self, reagents):
        # reagents = [r.mol if isinstance(r, Molecule) else r for r in reagents]
        return self.template.RunReactants(reagents)

    def annotate_reactants(self, classes):

        assert len(self.reactants) == len(classes)
        for i, _ in enumerate(self.reactants):
            self.reactants[i].SetProp('class', classes[i])
        return self

    def _repr_png_(self):
        return self.template._repr_png_()
