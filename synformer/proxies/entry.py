from typing import List
from .seh_proxy import SEHProxyWrapper
from .tdc_wrapper import Oracle
from rdkit import Chem
from rdkit.Chem import QED, Descriptors

import torch


class SimpleProxyWrapper:
    def __init__(self, oracle_name: str):
        self.oracle = self._get_raw_oracle(oracle_name)
        self.cache = dict(reserve=100000)

    def __call__(self, smiles: str):
        if smiles in self.cache:
            proxy_value = self.cache[smiles]
        else:
            proxy_value = self.oracle([smiles])[0]
            self.cache[smiles] = proxy_value
        return proxy_value
    
    @staticmethod
    def _get_raw_oracle(oracle_name: str):
        oracle_name_lowered = oracle_name.lower()
        if oracle_name_lowered == 'seh':
            seh_proxy = SEHProxyWrapper()
            if torch.cuda.is_available():
                seh_proxy.device = 'cuda'
                seh_proxy.model.to('cuda')
            return seh_proxy
        elif oracle_name_lowered == 'gsk' or oracle_name_lowered == 'gsk3b':
            return Oracle('gsk3b')
        elif oracle_name_lowered == 'jnk3':
            return Oracle('jnk3')
        elif oracle_name_lowered == 'sa':
            return Oracle('sa')
        elif oracle_name_lowered == 'mw':
            return lambda x: [Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in x]
        elif oracle_name_lowered == 'qed':
            return lambda x: [QED.qed(Chem.MolFromSmiles(s)) for s in x]
        else:
            raise 'oracle not supported'
