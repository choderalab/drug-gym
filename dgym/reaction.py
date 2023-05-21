import rdkit
from typing import Optional, List, Union
from rdkit.Chem.rdChemReactions import ChemicalReaction

class Reaction:
    def __init__(
        self,
        template: Union[str, ChemicalReaction],
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Parameters
        ----------
        template : rdkit.Chem.rdChemReactions.ChemicalReaction
            An rdkit reaction template.
        """
        if isinstance(template, str):
            template = rdkit.Chem.AllChem.ReactionFromSmarts(template)

        self.template = template
        self.products = list(template.GetProducts())
        self.agents = list(template.GetAgents())
        self.reactants = list(template.GetReactants())
        self.metadata = metadata
    
    def run(self, reagents):
        return self.template.RunReactants(reagents)

    def annotate_reactants(self, classes):

        assert len(self.reactants) == len(classes)
        for i, _ in enumerate(self.reactants):
            self.reactants[i].SetProp('class', classes[i])

        return self

    def poise(self, idx):
        
        def _move_idx_to_first(lst, idx):
            lst.insert(0, lst.pop(idx))
            return lst

        # if only one reactant, do nothing
        if len(self.reactants) == 1:
            return self

        # do not mutate in-place
        from copy import deepcopy
        temp = deepcopy(self)
        
        # reorder according to poised index
        temp.reactants = _move_idx_to_first(temp.reactants, idx)

        # update template
        new_template = rdkit.Chem.rdChemReactions.ChemicalReaction()
        for product in self.products:
            new_template.AddProductTemplate(product)
        for agent in self.agents:
            new_template.AddAgentTemplate(agent)
        for reactant in temp.reactants:
            new_template.AddReactantTemplate(reactant)

        temp.template = new_template
        return temp

    def _repr_png_(self):
        return self.template._repr_png_()
