# ACDC style 1,2 centered features from rascaline
# depending on the target decide what kind of features must be computed

from rascaline import SphericalExpansionByPair as PairExpansion
import ase

from rascaline import SphericalExpansion
from metatensor import TensorMap
import metatensor.operations as operations
import numpy as np
import warnings


from acdc_utils import (
    acdc_standardize_keys,
    cg_increment,
    cg_combine,
    relabel_key_contract,
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
        from symmetry import ClebschGordanReal

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
        from symmetry import ClebschGordanReal

        L = max(lcut, hypers["max_angular"])
        cg = ClebschGordanReal(lmax=L)
        # cg = ClebschGordanReal(lmax=lcut)

    calculator = PairExpansion(**hypers)
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

    # rho0_ij = acdc_standardize_keys(rho0_ij)
    rho0_ij = fix_gij(rho0_ij)
    rho0_ij = acdc_standardize_keys(rho0_ij)

    if not (frames[0].pbc.any()):
        for _ in ["cell_shift_a", "cell_shift_b", "cell_shift_c"]:
            rho0_ij = operations.remove_dimension(rho0_ij, axis="samples", name=_)

    if rhonu_i is None:
        rhonu_i = single_center_features(
            frames, order_nu=order_nu, hypers=hypers, lcut=lcut, cg=cg, kwargs=kwargs
        )
    if not both_centers:
        rhonu_ij = cg_combine(
            rhonu_i,
            rho0_ij,
            clebsch_gordan=cg,
            other_keys_match=["species_center"],
            lcut=lcut,
            feature_names=kwargs.get("feature_names", None),
        )
        return rhonu_ij

    else:
        # build the feature with atom-centered density on both centers
        # rho_ij = rho_i x gij x rho_j
        rhonu_ip = relabel_key_contract(rhonu_i)
        # gji = relabel_key_contract(gij)
        rho0_ji = relabel_key_contract(rho0_ij)

        rhonu_ijp = cg_increment(
            rhonu_ip,
            rho0_ji,
            lcut=lcut,
            other_keys_match=["species_contract"],
            clebsch_gordan=cg,
            mp=True,
        )

        rhonu_nuijp = cg_combine(
            rhonu_i,
            rhonu_ijp,
            lcut=lcut,
            other_keys_match=["species_center"],
            clebsch_gordan=cg,
        )
        return rhonu_nuijp
