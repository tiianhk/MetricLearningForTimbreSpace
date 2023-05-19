import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
import umap
from instrument_name_utils import INST_NAME_TO_INST_FAM_NAME_DICT
from paras import RNG_STATE

class visualizer():

    def __init__(self, x, inst, method='umap', n_components=2):
        """
        Arg:
            x: (n_samples, n_features)
            inst: a list of instrument labels
            method: dimension reduction method, 'umap'(default) or 'pca'
            n_components: number of dimensions to keep, 2(default) or 3
        """
        self.x = x
        self.inst = inst
        self.method = method
        self.n_components = n_components
        self._check_input()
        self.reducer = self.get_reducer()
        self.inst_list = list(INST_NAME_TO_INST_FAM_NAME_DICT.keys())
        self.inst_fam = [INST_NAME_TO_INST_FAM_NAME_DICT[inst] for inst in self.inst]

    def _check_input(self):
        assert self.x.shape[0] == len(self.inst)
        assert self.method == 'umap' or self.method == 'pca'
        assert self.n_components == 2 or self.n_components == 3

    def get_reducer(self):
        if self.method == 'umap':
            reducer = umap.UMAP(n_components=self.n_components, unique=True, random_state=RNG_STATE)
        elif self.method == 'pca':
            reducer = PCA(n_components=self.n_components)
        return reducer

    def generate(self):
        vec = self.reducer.fit_transform(self.x)
        scatter_fn = px.scatter if self.n_components == 2 else px.scatter_3d
        opacity = 0.5 if self.n_components == 2 else 1.0
        axes = ['x', 'y', 'z']
        proj = {}
        scatter_kwargs = {}
        for i in range(self.n_components):
            proj[axes[i]] = vec[:,i]
            scatter_kwargs[axes[i]] = axes[i]
        df = pd.DataFrame(dict(
            inst=self.inst,
            inst_fam=self.inst_fam,
            size=5,
            **proj,
        ))
        color_map = {self.inst_list[i]: px.colors.qualitative.Light24[i] \
                     for i in range(len(self.inst_list))}
        fig = scatter_fn(df, **scatter_kwargs, color='inst',
                         symbol='inst_fam', size='size',
                         color_discrete_map=color_map,
                         symbol_map={'string': 'circle',
                                     'brass': 'diamond',
                                     'woodwind': 'cross'},
                         category_orders={'inst': self.inst_list},
                         opacity=opacity, width=900, height=900)
        return fig

