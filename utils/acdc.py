# ACDC style 1,2 centered features from rascaline
# depending on the target decide what kind of features must be computed

from rascaline import SphericalExpansionByPair as PairExpansion
import ase

from rascaline import SphericalExpansion
from metatensor import TensorMap
import metatensor.operations as operations
import numpy as np
import warnings


from .acdc_utils import (
    acdc_standardize_keys,
    cg_increment,
    cg_combine,
    relabel_keys,
    fix_gij,
    _pca,
)
from typing import List, Union


def single_center_features(frames, hypers, order_nu, lcut=None, cg=None, **kwargs):
    calculator = SphericalExpansion(**hypers)
    rhoi = calculator.compute(frames)
    rhoi = rhoi.keys_to_properties(["species_neighbor"])
    # print(rhoi[0].samples)
    rho1i = acdc_standardize_keys(rhoi)

    if order_nu == 1:
        return rho1i
    if lcut is None:
        lcut = 10
    if cg is None:
        from .symmetry import ClebschGordanReal

        L = max(lcut, hypers["max_angular"])
        cg = ClebschGordanReal(lmax=L)
    rho_prev = rho1i
    # compute nu order feature recursively
    for _ in range(order_nu - 2):
        rho_x = cg_combine(
            rho_prev,
            rho1i,
            clebsch_gordan=cg,
            lcut=lcut,
            other_keys_match=["species_center"],
        )
        rho_prev = _pca(
            rho_x, kwargs.get("npca", None), kwargs.get("slice_samples", None)
        )

    rho_x = cg_increment(
        rho_prev,
        rho1i,
        clebsch_gordan=cg,
        lcut=lcut,
        other_keys_match=["species_center"],
    )
    if kwargs.get("pca_final", False):
        warnings.warn("PCA final features")
        rho_x = _pca(rho_x, kwargs.get("npca", None), kwargs.get("slice_samples", None))
    return rho_x


def pair_features(
    frames: List[ase.Atoms],
    hypers: dict,
    hypers_pair: dict = None,
    cg=None,
    rhonu_i: TensorMap = None,
    order_nu: Union[List[int], int] = None,
    all_pairs: bool = False,
    both_centers: bool = False,
    lcut: int = 3,
    max_shift=None,
    **kwargs,
):
    if not isinstance(frames, list):
        frames = [frames]
    if lcut is None:
        lcut = 10
    if cg is None:
        from .symmetry import ClebschGordanReal

        L = max(lcut, hypers["max_angular"])
        cg = ClebschGordanReal(lmax=L)
        # cg = ClebschGordanReal(lmax=lcut)
    if hypers_pair is None:
        hypers_pair = hypers
    calculator = PairExpansion(**hypers_pair)

    rho0_ij = calculator.compute(frames) 

    if all_pairs:
        hypers_allpairs = hypers.copy()
        if max_shift is None and hypers["cutoff"] < np.max(
            [np.max(f.get_all_distances()) for f in frames]
        ):
            hypers_allpairs["cutoff"] = np.ceil(
                np.max([np.max(f.get_all_distances()) for f in frames])
            )
            nmax = int(
                hypers_allpairs["max_radial"]
                / hypers["cutoff"]
                * hypers_allpairs["cutoff"]
            )
            hypers_allpairs["max_radial"] = nmax
        elif max_shift is not None:
            repframes = [f.repeat(max_shift) for f in frames]
            hypers_allpairs["cutoff"] = np.ceil(
                np.max([np.max(f.get_all_distances()) for f in repframes])
            )

            warnings.warn(
                f"Using cutoff {hypers_allpairs['cutoff']} for all pairs feature"
            )
        else:
            warnings.warn(f"Using unchanged hypers for all pairs feature")
        calculator_allpairs = PairExpansion(**hypers_allpairs)

        rho0_ij = calculator_allpairs.compute(frames)

    rho0_ij = fix_gij(rho0_ij)
    rho0_ij = acdc_standardize_keys(rho0_ij)

    if isinstance(order_nu, list):
        assert (
            len(order_nu) == 2
        ), "specify order_nu as [nu_i, nu_j] for correlation orders for i and j respectively"
        order_nu_i, order_nu_j = order_nu
    else:
        assert isinstance(order_nu, int), "specify order_nu as int or list of 2 ints"
        order_nu_i = order_nu

    if not (frames[0].pbc.any()):
        for _ in ["cell_shift_a", "cell_shift_b", "cell_shift_c"]:
            rho0_ij = operations.remove_dimension(rho0_ij, axis="samples", name=_)

    # must compute rhoi as sum of rho_0ij
    if rhonu_i is None:
        rhonu_i = single_center_features(
            frames, order_nu=order_nu_i, hypers=hypers, lcut=lcut, cg=cg, kwargs=kwargs
        )
        # rhonu_i = _standardize(rhonu_i)
    # if not both_centers:
    rhonu_ij = cg_combine(
        rhonu_i,
        rho0_ij,
        clebsch_gordan=cg,
        other_keys_match=["species_center"],
        lcut=lcut,
        feature_names=kwargs.get("feature_names", None),
    )
    if not both_centers:
        return rhonu_ij

    else:
        if "order_nu_j" not in locals():
            warnings.warn("nu_j not defined, using nu_i for nu_j as well")
            order_nu_j = order_nu_i
        if order_nu_j != order_nu_i:
            rhonup_j = single_center_features(
                frames,
                order_nu=order_nu_j,
                hypers=hypers,
                lcut=lcut,
                cg=cg,
                kwargs=kwargs,
            )
        else:
            rhonup_j = rhonu_i.copy()

        rhoj = relabel_keys(rhonup_j, "species_neighbor")

        # build rhoj x gij
        rhonu_nupij = cg_combine(
            rhoj,
            rhonu_ij,
            # rhoj,
            lcut=lcut,
            other_keys_match=["species_neighbor"],
            clebsch_gordan=cg,
            mp=True,  # for combining with neighbor
            feature_names=kwargs.get("feature_names", None),
        )
        # combine with rhoi
        # rhonu_nupij = cg_combine(
        #     rhonu_i,
        #     rhonuij,
        #     lcut=lcut,
        #     other_keys_match=["species_center"],
        #     clebsch_gordan=cg,
        #     feature_names=kwargs.get("feature_names", None),
        # )

        return rhonu_nupij
    
    
  
