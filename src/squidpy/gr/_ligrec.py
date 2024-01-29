"""Permutation test function as described in CellPhoneDB 2.0."""

from __future__ import annotations

from abc import ABC
from collections import namedtuple
from functools import partial
from itertools import product
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Sequence, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from numba import njit, prange  # noqa: F401
from scanpy import logging as logg
from scipy.sparse import csc_matrix
from spatialdata import SpatialData

from squidpy._constants._constants import ComplexPolicy, CorrAxis
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d, inject_docs
from squidpy._utils import NDArrayA, Signal, SigQueue, _get_n_cores, parallelize
from squidpy.gr._utils import (
    _assert_categorical_obs,
    _assert_positive,
    _check_tuple_needles,
    _create_sparse_df,
    _genesymbols,
    _save_data,
)

__all__ = ["ligrec", "PermutationTest"]

StrSeq = Sequence[str]
SeqTuple = Sequence[tuple[str, str]]
Interaction_t = Union[pd.DataFrame, Mapping[str, StrSeq], StrSeq, tuple[StrSeq, StrSeq], SeqTuple]
Cluster_t = Union[StrSeq, tuple[StrSeq, StrSeq], SeqTuple]

SOURCE = "source"
TARGET = "target"

TempResult = namedtuple("TempResult", ["means", "pvalues"])

_template = """
@njit(parallel={parallel}, cache=False, fastmath=False)
def _test_{n_cls}_{ret_means}_{parallel}(
    interactions: NDArrayA,  # [np.uint32],
    interaction_clusters: NDArrayA,  # [np.uint32],
    data: NDArrayA,  # [np.float64],
    clustering: NDArrayA,  # [np.uint32],
    mean: NDArrayA,  # [np.float64],
    mask: NDArrayA,  # [np.bool_],
    res: NDArrayA,  # [np.float64],
    {args}
) -> None:

    {init}
    {loop}
    {finalize}

    for i in prange(len(interactions)):
        rec, lig = interactions[i]
        for j in prange(len(interaction_clusters)):
            c1, c2 = interaction_clusters[j]
            m1, m2 = mean[rec, c1], mean[lig, c2]

            if np.isnan(res[i, j]):
                continue

            if m1 > 0 and m2 > 0:
                {set_means}
                if mask[rec, c1] and mask[lig, c2]:
                    # both rec, lig are sufficiently expressed in c1, c2
                    res[i, j] += (groups[c1, rec] + groups[c2, lig]) > (m1 + m2)
                else:
                    res[i, j] = np.nan
            else:
                # res_means is initialized with 0s
                res[i, j] = np.nan
"""


def _create_template(n_cls: int, return_means: bool = False, parallel: bool = True) -> str:
    if n_cls <= 0:
        raise ValueError(f"Expected number of clusters to be positive, found `{n_cls}`.")

    rng = range(n_cls)
    init = "".join(
        f"""
    g{i} = np.zeros((data.shape[1],), dtype=np.float64); s{i} = 0"""
        for i in rng
    )

    loop_body = """
        if cl == 0:
            g0 += data[row]
            s0 += 1"""
    loop_body = loop_body + "".join(
        f"""
        elif cl == {i}:
            g{i} += data[row]
            s{i} += 1"""
        for i in range(1, n_cls)
    )
    loop = f"""
    for row in prange(data.shape[0]):
        cl = clustering[row]
        {loop_body}
        else:
            assert False, "Unhandled case."
    """
    finalize = ", ".join(f"g{i} / s{i}" for i in rng)
    finalize = f"groups = np.stack(({finalize}))"

    if return_means:
        args = "res_means: NDArrayA,  # [np.float64]"
        set_means = "res_means[i, j] = (m1 + m2) / 2.0"
    else:
        args = set_means = ""

    return _template.format(
        n_cls=n_cls,
        parallel=bool(parallel),
        ret_means=int(return_means),
        args=args,
        init=init,
        loop=loop,
        finalize=finalize,
        set_means=set_means,
    )


def _fdr_correct(
    pvals: pd.DataFrame,
    corr_method: str,
    corr_axis: Literal["interactions", "clusters"] | CorrAxis,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Correct p-values for FDR along specific axis in ``pvals``."""
    from pandas.core.arrays.sparse import SparseArray
    from statsmodels.stats.multitest import multipletests

    def fdr(pvals: pd.Series) -> SparseArray:
        _, qvals, _, _ = multipletests(
            np.nan_to_num(pvals.values, copy=True, nan=1.0),
            method=corr_method,
            alpha=alpha,
            is_sorted=False,
            returnsorted=False,
        )
        qvals[np.isnan(pvals.values)] = np.nan

        return SparseArray(qvals, dtype=qvals.dtype, fill_value=np.nan)

    corr_axis = CorrAxis(corr_axis)

    if corr_axis == CorrAxis.CLUSTERS:
        # clusters are in columns
        pvals = pvals.apply(fdr)
    elif corr_axis == CorrAxis.INTERACTIONS:
        pvals = pvals.T.apply(fdr).T
    else:
        raise NotImplementedError(f"FDR correction for `{corr_axis}` is not implemented.")

    return pvals


@d.get_full_description(base="PT")
@d.get_sections(base="PT", sections=["Parameters"])
@d.dedent
class PermutationTestABC(ABC):
    """
    Class for receptor-ligand interaction testing.

    The expected workflow is::

        pt = PermutationTest(adata).prepare()
        res = pt.test("clusters")

    Parameters
    ----------
    %(adata)s
    use_raw
        Whether to access :attr:`anndata.AnnData.raw`.
    """

    def __init__(self, adata: AnnData, use_raw: bool = True):
        if not isinstance(adata, AnnData):
            raise TypeError(f"Expected `adata` to be of type `anndata.AnnData`, found `{type(adata).__name__}`.")
        if not adata.n_obs:
            raise ValueError("No cells are in `adata.obs_names`.")
        if not adata.n_vars:
            raise ValueError("No genes are in `adata.var_names`.")

        self._adata = adata
        if use_raw:
            if adata.raw is None:
                raise AttributeError("No `.raw` attribute found. Try specifying `use_raw=False`.")
            if adata.raw.n_obs != adata.n_obs:
                raise ValueError(f"Expected `{adata.n_obs}` cells in `.raw` object, found `{adata.raw.n_obs}`.")
            adata = adata.raw

        self._data = pd.DataFrame.sparse.from_spmatrix(
            csc_matrix(adata.X), index=adata.obs_names, columns=adata.var_names
        )

        self._interactions: pd.DataFrame | None = None
        self._filtered_data: pd.DataFrame | None = None

    @d.get_full_description(base="PT_prepare")
    @d.get_sections(base="PT_prepare", sections=["Parameters", "Returns"])
    @inject_docs(src=SOURCE, tgt=TARGET, cp=ComplexPolicy)
    def prepare(
        self, interactions: Interaction_t, complex_policy: Literal["min", "all"] | ComplexPolicy = ComplexPolicy.MIN.v
    ) -> PermutationTestABC:
        """
        Prepare self for running the permutation test.

        Parameters
        ----------
        interactions
            Interaction to test. The type can be one of:

                - :class:`pandas.DataFrame` - must contain at least 2 columns named `{src!r}` and `{tgt!r}`.
                - :class:`dict` - dictionary with at least 2 keys named `{src!r}` and `{tgt!r}`.
                - :class:`typing.Sequence` - Either a sequence of :class:`str`, in which case all combinations are
                  produced, or a sequence of :class:`tuple` of 2 :class:`str` or a :class:`tuple` of 2 sequences.

            If `None`, the interactions are extracted from :mod:`omnipath`. Protein complexes can be specified by
            delimiting the components with `'_'`, such as `'alpha_beta_gamma'`.
        complex_policy
            Policy on how to handle complexes. Valid options are:

                - `{cp.MIN.s!r}` - select gene with the minimum average expression. This is the same as in
                  :cite:`cellphonedb`.
                - `{cp.ALL.s!r}` - select all possible combinations between `{src!r}` and `{tgt!r}` complexes.

        Returns
        -------
        Sets the following attributes and returns :attr:`self`:

            - :attr:`interactions` - filtered interactions whose `{src!r}` and `{tgt!r}` are both in the data.
        """
        complex_policy = ComplexPolicy(complex_policy)

        if isinstance(interactions, Mapping):
            interactions = pd.DataFrame(interactions)

        if isinstance(interactions, pd.DataFrame):
            if SOURCE not in interactions.columns:
                raise KeyError(f"Column `{SOURCE!r}` is not in `interactions`.")
            if TARGET not in interactions.columns:
                raise KeyError(f"Column `{TARGET!r}` is not in `interactions`.")

            self._interactions = interactions.copy()
        elif isinstance(interactions, Iterable):
            interactions = tuple(interactions)
            if not len(interactions):
                raise ValueError("No interactions were specified.")

            if isinstance(interactions[0], str):
                interactions = list(product(interactions, repeat=2))
            elif len(interactions) == 2:
                interactions = tuple(zip(*interactions))

            if not all(len(i) == 2 for i in interactions):
                raise ValueError("Not all interactions are of length `2`.")

            self._interactions = pd.DataFrame(interactions, columns=[SOURCE, TARGET])
        else:
            raise TypeError(
                f"Expected either a `pandas.DataFrame`, `dict` or `iterable`, found `{type(interactions).__name__}`"
            )
        if TYPE_CHECKING:
            assert isinstance(self.interactions, pd.DataFrame)

        if self.interactions.empty:
            raise ValueError("The interactions are empty")

        # first uppercase, then drop duplicates
        self._data.columns = self._data.columns.str.upper()
        self.interactions[SOURCE] = self.interactions[SOURCE].str.upper()
        self.interactions[TARGET] = self.interactions[TARGET].str.upper()

        logg.debug("DEBUG: Removing duplicate interactions")
        self.interactions.dropna(subset=(SOURCE, TARGET), inplace=True, how="any")
        self.interactions.drop_duplicates(subset=(SOURCE, TARGET), inplace=True, keep="first")

        logg.debug("DEBUG: Removing duplicate genes in the data")
        n_genes_prior = self._data.shape[1]
        self._data = self._data.loc[:, ~self._data.columns.duplicated()]
        if self._data.shape[1] != n_genes_prior:
            logg.warning(f"Removed `{n_genes_prior - self._data.shape[1]}` duplicate gene(s)")

        self._filter_interactions_complexes(complex_policy)
        self._filter_interactions_by_genes()
        self._trim_data()

        # this is necessary because of complexes
        self.interactions.drop_duplicates(subset=(SOURCE, TARGET), inplace=True, keep="first")

        return self

    @d.get_full_description(base="PT_test")
    @d.get_sections(base="PT_test", sections=["Parameters"])
    @d.dedent
    @inject_docs(src=SOURCE, tgt=TARGET, fa=CorrAxis)
    def test(
        self,
        cluster_key: str,
        clusters: Cluster_t | None = None,
        n_perms: int = 1000,
        threshold: float = 0.01,
        seed: int | None = None,
        corr_method: str | None = None,
        corr_axis: Literal["interactions", "clusters"] | CorrAxis = CorrAxis.INTERACTIONS.v,
        alpha: float = 0.05,
        copy: bool = False,
        key_added: str | None = None,
        numba_parallel: bool | None = None,
        **kwargs: Any,
    ) -> Mapping[str, pd.DataFrame] | None:
        """
        Perform the permutation test as described in :cite:`cellphonedb`.

        Parameters
        ----------
        %(cluster_key)s
        clusters
            Clusters from :attr:`anndata.AnnData.obs` ``['{{cluster_key}}']``. Can be specified either as a sequence
            of :class:`tuple` or just a sequence of cluster names, in which case all combinations considered.
        %(n_perms)s
        threshold
            Do not perform permutation test if any of the interacting components is being expressed
            in less than ``threshold`` percent of cells within a given cluster.
        %(seed)s
        %(corr_method)s
        corr_axis
            Axis over which to perform the FDR correction. Only used when ``corr_method != None``. Valid options are:

                - `{fa.INTERACTIONS.s!r}` - correct interactions by performing FDR correction across the clusters.
                - `{fa.CLUSTERS.s!r}` - correct clusters by performing FDR correction across the interactions.
        alpha
            Significance level for FDR correction. Only used when ``corr_method != None``.
        %(copy)s
        key_added
            Key in :attr:`anndata.AnnData.uns` where the result is stored if ``copy = False``.
            If `None`, ``'{{cluster_key}}_ligrec'`` will be used.
        %(numba_parallel)s
        %(parallelize)s

        Returns
        -------
        %(ligrec_test_returns)s
        """
        _assert_positive(n_perms, name="n_perms")
        _assert_categorical_obs(self._adata, key=cluster_key)

        if corr_method is not None:
            corr_axis = CorrAxis(corr_axis)
        if TYPE_CHECKING:
            assert isinstance(corr_axis, CorrAxis)

        if len(self._adata.obs[cluster_key].cat.categories) <= 1:
            raise ValueError(
                f"Expected at least `2` clusters, found `{len(self._adata.obs[cluster_key].cat.categories)}`."
            )
        if TYPE_CHECKING:
            assert isinstance(self.interactions, pd.DataFrame)
            assert isinstance(self._filtered_data, pd.DataFrame)

        interactions = self.interactions[[SOURCE, TARGET]]
        self._filtered_data["clusters"] = self._adata.obs.copy()[cluster_key].astype("string").astype("category").values

        if clusters is None:
            clusters = list(map(str, self._adata.obs[cluster_key].cat.categories))
        if all(isinstance(c, str) for c in clusters):
            clusters = list(product(clusters, repeat=2))  # type: ignore[assignment]
        clusters = sorted(
            _check_tuple_needles(
                clusters,  # type: ignore[arg-type]
                self._filtered_data["clusters"].cat.categories,
                msg="Invalid cluster `{0!r}`.",
                reraise=True,
            )
        )
        clusters_flat = list({c for cs in clusters for c in cs})

        data = self._filtered_data.loc[np.isin(self._filtered_data["clusters"], clusters_flat), :]
        data["clusters"] = data["clusters"].cat.remove_unused_categories()
        cat = data["clusters"].cat

        cluster_mapper = dict(zip(cat.categories, range(len(cat.categories))))
        gene_mapper = dict(zip(data.columns[:-1], range(len(data.columns) - 1)))  # -1 for 'clusters'

        data.columns = [gene_mapper[c] if c != "clusters" else c for c in data.columns]
        clusters_ = np.array([[cluster_mapper[c1], cluster_mapper[c2]] for c1, c2 in clusters], dtype=np.uint32)

        data["clusters"] = cat.rename_categories(cluster_mapper)
        # much faster than applymap (tested on 1M interactions)
        interactions_ = np.vectorize(lambda g: gene_mapper[g])(interactions.values)

        n_jobs = _get_n_cores(kwargs.pop("n_jobs", None))
        start = logg.info(
            f"Running `{n_perms}` permutations on `{len(interactions)}` interactions "
            f"and `{len(clusters)}` cluster combinations using `{n_jobs}` core(s)"
        )
        res = _analysis(
            data,
            interactions_,
            clusters_,
            threshold=threshold,
            n_perms=n_perms,
            seed=seed,
            n_jobs=n_jobs,
            numba_parallel=numba_parallel,
            **kwargs,
        )
        res = {
            "means": _create_sparse_df(
                res.means,
                index=pd.MultiIndex.from_frame(interactions, names=[SOURCE, TARGET]),
                columns=pd.MultiIndex.from_tuples(clusters, names=["cluster_1", "cluster_2"]),
                fill_value=0,
            ),
            "pvalues": _create_sparse_df(
                res.pvalues,
                index=pd.MultiIndex.from_frame(interactions, names=[SOURCE, TARGET]),
                columns=pd.MultiIndex.from_tuples(clusters, names=["cluster_1", "cluster_2"]),
                fill_value=np.nan,
            ),
            "metadata": self.interactions[self.interactions.columns.difference([SOURCE, TARGET])],
        }
        res["metadata"].index = res["means"].index.copy()

        if TYPE_CHECKING:
            assert isinstance(res, dict)

        if corr_method is not None:
            logg.info(
                f"Performing FDR correction across the `{corr_axis.v}` "
                f"using method `{corr_method}` at level `{alpha}`"
            )
            res["pvalues"] = _fdr_correct(res["pvalues"], corr_method, corr_axis, alpha=alpha)

        if copy:
            logg.info("Finish", time=start)
            return res

        _save_data(self._adata, attr="uns", key=Key.uns.ligrec(cluster_key, key_added), data=res, time=start)

    def _trim_data(self) -> None:
        """Subset genes :attr:`_data` to those present in interactions."""
        if TYPE_CHECKING:
            assert isinstance(self._data, pd.DataFrame)
            assert isinstance(self.interactions, pd.DataFrame)

        logg.debug("DEBUG: Removing genes not in any interaction")
        self._filtered_data = self._data.loc[:, list(set(self.interactions[SOURCE]) | set(self.interactions[TARGET]))]

    def _filter_interactions_by_genes(self) -> None:
        """Subset :attr:`interactions` to only those for which we have the data."""
        if TYPE_CHECKING:
            assert isinstance(self.interactions, pd.DataFrame)

        logg.debug("DEBUG: Removing interactions with no genes in the data")
        self._interactions = self.interactions[
            self.interactions[SOURCE].isin(self._data.columns) & self.interactions[TARGET].isin(self._data.columns)
        ]

        if self.interactions.empty:
            raise ValueError("After filtering by genes, no interactions remain.")

    @inject_docs(src=SOURCE, tgt=TARGET, cp=ComplexPolicy)
    def _filter_interactions_complexes(self, complex_policy: ComplexPolicy) -> None:
        """
        Filter the :attr:`interactions` by extracting genes from complexes.

        Parameters
        ----------
        complex_policy
            Policy on how to handle complexes. Valid options are:

                - `{cp.MIN.s!r}` - select gene with the minimum average expression. This is the same as in
                  :cite:`cellphonedb`.
                - `{cp.ALL.s!r}` - select all possible combinations between `{src!r}` and `{tgt!r}` complexes.

        Returns
        -------
        Nothing, just updates the following fields:

            - :attr:`interactions` - filtered interactions whose `{src!r}` and `{tgt!r}` are both in the data.

        Note that for ``complex_policy={cp.ALL.s!r}``, all pairwise comparisons within a complex are created,
        but no filtering happens at this stage - genes not present in the data are filtered at a later stage.
        """

        def find_min_gene_in_complex(_complex: str | None) -> str | None:
            # TODO(michalk8): how can this happen?
            if _complex is None:
                return None
            if "_" not in _complex:
                return _complex
            complexes = [c for c in _complex.split("_") if c in self._data.columns]
            if not len(complexes):
                return None
            if len(complexes) == 1:
                return complexes[0]

            df = self._data[complexes].mean()
            try:
                return str(df.index[df.argmin()])
            except ValueError as e:
                if "attempt to get argmin of an empty sequence" in str(e):
                    return str(df.index[0])
                else:
                    logg.error(e)

        if TYPE_CHECKING:
            assert isinstance(self._interactions, pd.DataFrame)
            assert isinstance(self.interactions, pd.DataFrame)

        if complex_policy == ComplexPolicy.MIN:
            logg.debug("DEBUG: Selecting genes from complexes based on minimum average expression")
            self.interactions[SOURCE] = self.interactions[SOURCE].apply(find_min_gene_in_complex)
            self.interactions[TARGET] = self.interactions[TARGET].apply(find_min_gene_in_complex)
        elif complex_policy == ComplexPolicy.ALL:
            logg.debug("DEBUG: Creating all gene combinations within complexes")
            src = self.interactions.pop(SOURCE).apply(lambda s: str(s).split("_")).explode()
            src.name = SOURCE
            tgt = self.interactions.pop(TARGET).apply(lambda s: str(s).split("_")).explode()
            tgt.name = TARGET

            self._interactions = pd.merge(self.interactions, src, how="left", left_index=True, right_index=True)
            self._interactions = pd.merge(self.interactions, tgt, how="left", left_index=True, right_index=True)
        else:
            raise NotImplementedError(f"Complex policy {complex_policy!r} is not implemented.")

    @property
    def interactions(self) -> pd.DataFrame | None:
        """The interactions."""  # noqa: D401
        return self._interactions

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}"
            f"[n_interaction={len(self.interactions) if self.interactions is not None else None}]>"
        )

    def __str__(self) -> str:
        return repr(self)


@d.dedent
class PermutationTest(PermutationTestABC):
    """
    %(PT.full_desc)s

    Parameters
    ----------
    %(PT.parameters)s
    """  # noqa: D400

    @d.get_sections(base="PT_prepare_full", sections=["Parameters"])
    @d.dedent
    def prepare(
        self,
        interactions: Interaction_t | None = None,
        complex_policy: Literal["min", "all"] = ComplexPolicy.MIN.v,
        interactions_params: Mapping[str, Any] = MappingProxyType({}),
        transmitter_params: Mapping[str, Any] = MappingProxyType({"categories": "ligand"}),
        receiver_params: Mapping[str, Any] = MappingProxyType({"categories": "receptor"}),
        **_: Any,
    ) -> PermutationTest:
        """
        %(PT_prepare.full_desc)s

        Parameters
        ----------
        %(PT_prepare.parameters)s
        interactions_params
            Keyword arguments for :func:`omnipath.interactions.import_intercell_network` defining the interactions.
            These datasets from :cite:`omnipath` are used by default: `omnipath`, `pathwayextra`, `kinaseextra` and
            `ligrecextra`.
        transmitter_params
            Keyword arguments for :func:`omnipath.interactions.import_intercell_network` defining the transmitter
            side of intercellular connections.
        receiver_params
            Keyword arguments for :func:`omnipath.interactions.import_intercell_network` defining the receiver
            side of intercellular connections.

        Returns
        -------
        %(PT_prepare.returns)s
        """  # noqa: D400
        if interactions is None:
            from omnipath.interactions import import_intercell_network

            start = logg.info("Fetching interactions from `omnipath`")
            interactions = import_intercell_network(
                interactions_params=interactions_params,
                transmitter_params=transmitter_params,
                receiver_params=receiver_params,
            )
            if TYPE_CHECKING:
                assert isinstance(interactions, pd.DataFrame)

            logg.info(f"Fetched `{len(interactions)}` interactions\n    Finish", time=start)

            # we don't really care about these
            if SOURCE in interactions.columns:
                interactions.pop(SOURCE)
            if TARGET in interactions.columns:
                interactions.pop(TARGET)
            interactions.rename(
                columns={"genesymbol_intercell_source": SOURCE, "genesymbol_intercell_target": TARGET}, inplace=True
            )

            interactions[SOURCE] = interactions[SOURCE].str.replace("^COMPLEX:", "", regex=True)
            interactions[TARGET] = interactions[TARGET].str.replace("^COMPLEX:", "", regex=True)

        _ = super().prepare(interactions, complex_policy=complex_policy)
        return self


@d.dedent
def ligrec(
    adata: AnnData | SpatialData,
    cluster_key: str,
    interactions: Interaction_t | None = None,
    complex_policy: Literal["min", "all"] = ComplexPolicy.MIN.v,
    threshold: float = 0.01,
    corr_method: str | None = None,
    corr_axis: Literal["interactions", "clusters"] = CorrAxis.CLUSTERS.v,
    use_raw: bool = True,
    copy: bool = False,
    key_added: str | None = None,
    gene_symbols: str | None = None,
    **kwargs: Any,
) -> Mapping[str, pd.DataFrame] | None:
    """
    %(PT_test.full_desc)s

    Parameters
    ----------
    %(PT.parameters)s
    %(PT_prepare_full.parameters)s
    %(PT_test.parameters)s
    gene_symbols
        Key in :attr:`anndata.AnnData.var` to use instead of :attr:`anndata.AnnData.var_names`.

    Returns
    -------
    %(ligrec_test_returns)s
    """  # noqa: D400
    if isinstance(adata, SpatialData):
        adata = adata.table
    with _genesymbols(adata, key=gene_symbols, use_raw=use_raw, make_unique=False):
        return (  # type: ignore[no-any-return]
            PermutationTest(adata, use_raw=use_raw)
            .prepare(interactions, complex_policy=complex_policy, **kwargs)
            .test(
                cluster_key=cluster_key,
                threshold=threshold,
                corr_method=corr_method,
                corr_axis=corr_axis,
                copy=copy,
                key_added=key_added,
                **kwargs,
            )
        )


@d.dedent
def _analysis(
    data: pd.DataFrame,
    interactions: NDArrayA,
    interaction_clusters: NDArrayA,
    threshold: float = 0.1,
    n_perms: int = 1000,
    seed: int | None = None,
    n_jobs: int = 1,
    numba_parallel: bool | None = None,
    **kwargs: Any,
) -> TempResult:
    """
    Run the analysis as described in :cite:`cellphonedb`.

    This function runs the mean, percent and shuffled analysis.

    Parameters
    ----------
    data
        Array of shape `(n_cells, n_genes)`.
    interactions
        Array of shape `(n_interactions, 2)`.
    interaction_clusters
        Array of shape `(n_interaction_clusters, 2)`.
    threshold
        Percentage threshold for removing lowly expressed genes in clusters.
    %(n_perms)s
    %(seed)s
    n_jobs
        Number of parallel jobs to launch.
    numba_parallel
        Whether to use :class:`numba.prange` or not. If `None`, it's determined automatically.
    kwargs
        Keyword arguments for :func:`squidpy._utils.parallelize`, such as ``n_jobs`` or ``backend``.

    Returns
    -------
    Tuple of the following format:

        - `'means'` - array of shape `(n_interactions, n_interaction_clusters)` containing the means.
        - `'pvalues'` - array of shape `(n_interactions, n_interaction_clusters)` containing the p-values.
    """

    def extractor(res: Sequence[TempResult]) -> TempResult:
        assert len(res) == n_jobs, f"Expected to find `{n_jobs}` results, found `{len(res)}`."

        meanss: list[NDArrayA] = [r.means for r in res if r.means is not None]
        assert len(meanss) == 1, f"Only `1` job should've calculated the means, but found `{len(meanss)}`."
        means = meanss[0]
        if TYPE_CHECKING:
            assert isinstance(means, np.ndarray)

        pvalues = np.sum([r.pvalues for r in res if r.pvalues is not None], axis=0) / float(n_perms)
        assert means.shape == pvalues.shape, f"Means and p-values differ in shape: `{means.shape}`, `{pvalues.shape}`."

        return TempResult(means=means, pvalues=pvalues)

    groups = data.groupby("clusters")
    clustering = np.array(data["clusters"].values, dtype=np.int32)

    mean = groups.mean().values.T  # (n_genes, n_clusters)
    mask = groups.apply(lambda c: ((c > 0).sum() / len(c)) >= threshold).values.T  # (n_genes, n_clusters)
    # (n_cells, n_genes)
    data = np.array(data[data.columns.difference(["clusters"])].values, dtype=np.float64, order="C")
    # all 3 should be C contiguous

    return parallelize(  # type: ignore[no-any-return]
        _analysis_helper,
        np.arange(n_perms, dtype=np.int32).tolist(),
        n_jobs=n_jobs,
        unit="permutation",
        extractor=extractor,
        **kwargs,
    )(
        data,
        mean,
        mask,
        interactions,
        interaction_clusters=interaction_clusters,
        clustering=clustering,
        seed=seed,
        numba_parallel=numba_parallel,
    )


def _analysis_helper(
    perms: NDArrayA,
    data: NDArrayA,
    mean: NDArrayA,
    mask: NDArrayA,
    interactions: NDArrayA,
    interaction_clusters: NDArrayA,
    clustering: NDArrayA,
    seed: int | None = None,
    numba_parallel: bool | None = None,
    queue: SigQueue | None = None,
) -> TempResult:
    """
    Run the results of mean, percent and shuffled analysis.

    Parameters
    ----------
    perms
        Permutation indices. Only used to set the ``seed``.
    data
        Array of shape `(n_cells, n_genes)`.
    mean
        Array of shape `(n_genes, n_clusters)` representing mean expression per cluster.
    mask
        Array of shape `(n_genes, n_clusters)` containing `True` if the a gene within a cluster is
        expressed at least in ``threshold`` percentage of cells.
    interactions
        Array of shape `(n_interactions, 2)`.
    interaction_clusters
        Array of shape `(n_interaction_clusters, 2)`.
    clustering
        Array of shape `(n_cells,)` containing the original clustering.
    seed
        Random seed for :class:`numpy.random.RandomState`.
    numba_parallel
        Whether to use :class:`numba.prange` or not. If `None`, it's determined automatically.
    queue
        Signalling queue to update progress bar.

    Returns
    -------
    Tuple of the following format:

        - `'means'` - array of shape `(n_interactions, n_interaction_clusters)` containing the true test
          statistic. It is `None` if ``min(perms)!=0`` so that only 1 worker calculates it.
        - `'pvalues'` - array of shape `(n_interactions, n_interaction_clusters)`  containing `np.sum(T0 > T)`
          where `T0` is the test statistic under null hypothesis and `T` is the true test statistic.
    """
    rs = np.random.RandomState(None if seed is None else perms[0] + seed)

    clustering = clustering.copy()
    n_cls = mean.shape[1]
    return_means = np.min(perms) == 0

    # ideally, these would be both sparse array, but there is no numba impl. (sparse.COO is read-only and very limited)
    # keep it f64, because we're setting NaN
    res = np.zeros((len(interactions), len(interaction_clusters)), dtype=np.float64)
    numba_parallel = (
        (np.prod(res.shape) >= 2**20 or clustering.shape[0] >= 2**15) if numba_parallel is None else numba_parallel  # type: ignore[assignment]
    )

    fn_key = f"_test_{n_cls}_{int(return_means)}_{bool(numba_parallel)}"
    if fn_key not in globals():
        exec(
            compile(_create_template(n_cls, return_means=return_means, parallel=numba_parallel), "", "exec"), globals()  # type: ignore[arg-type]
        )
    _test = globals()[fn_key]

    if return_means:
        res_means: NDArrayA | None = np.zeros((len(interactions), len(interaction_clusters)), dtype=np.float64)
        test = partial(_test, res_means=res_means)
    else:
        res_means = None
        test = _test

    for _ in perms:
        rs.shuffle(clustering)
        test(interactions, interaction_clusters, data, clustering, mean, mask, res=res)

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return TempResult(means=res_means, pvalues=res)
