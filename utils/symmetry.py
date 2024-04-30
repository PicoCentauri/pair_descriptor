import torch
import numpy as np
import wigners

from typing import Dict


I_SQRT_2 = 1.0 / np.sqrt(2)
SQRT_2 = np.sqrt(2)


def _real2complex(L: int):
    """
    Computes the transformation matrix that goes from a set
    of real spherical harmonics, ordered as:

        (l, -l), (l, -l + 1), ..., (l, l - 1), (l, l)
    to complex-valued ones(following the Condon-Shortley phase convention)

    _complex2real is obtained simply by taking the conjugate transpose of the matrix.
    """
    dim = 2 * L + 1
    mat = torch.zeros((dim, dim), dtype=torch.complex128)
    # m = 0
    mat[L, L] = 1.0

    if L == 0:
        return mat
    for m in range(1, L + 1):
        # m > 0
        mat[L + m, L + m] = I_SQRT_2 * (-1) ** m
        mat[L + m, L - m] = I_SQRT_2 * 1j * (-1) ** m
        # m < 0
        mat[L - m, L + m] = I_SQRT_2
        mat[L - m, L - m] = -I_SQRT_2 * 1j

    return mat


def _complex_clebsch_gordan_matrix(l1: int, l2: int, L: int, device: str = None):
    """
    Computes the Clebsch-Gordan (CG) matrix for
    transforming complex-valued spherical harmonics.
    The CG matrix is computed as a 3D array of elements
    < l1 m1 l2 m2 | L M >
    where the first axis loops over m1, the second loops over m2,
    and the third one loops over M. The matrix is real.

    For example, using the relation:

        | l1 l2 L M > = \sum_{m1, m2} <l1 m1 l2 m2 | L M > | l1 m1 > | l2 m2 >

    Args:
        l1: Order of the first set of spherical harmonics
        l2: Order of the second set of spherical harmonics
        L: Order of the coupled spherical harmonics
    Returns:
        real_cg: CG matrix for transforming complex-valued spherical harmonics
    """
    if abs(l1 - l2) > L or (l1 + l2) < L:
        return torch.zeros(
            (2 * l1 + 1, 2 * l2 + 1, 2 * L + 1), dtype=torch.double, device=device
        )
    else:
        return torch.from_numpy(wigners.clebsch_gordan_array(l1, l2, L)).to(device)


def _real_clebsch_gordan_matrix(
    l1: int, l2: int, L: int, r2c_l1, r2c_l2, c2r_L, device: str = None
):
    """
    Compute the Clebsch Gordan (CG) matrix for *real* valued spherical harmonics,
    constructed by contracting the CG matrix for complex-valued
    spherical harmonics with the matrices that transform between
    real-valued and complex-valued spherical harmonics.

    Args:
        l1: Order of the first set of spherical harmonics
        l2: Order of the second set of spherical harmonics
        L: Order of the coupled spherical harmonics
    Returns:
        real_cg: CG matrix for transforming real-valued spherical harmonics
    """
    complex_cg = _complex_clebsch_gordan_matrix(l1, l2, L).to(device)
    real_cg = torch.einsum(
        "ijk,il,jm,nk->lmn",
        complex_cg.type(torch.complex128),
        r2c_l1.to(device),
        r2c_l2.to(device),
        c2r_L.to(device),
    )

    if (l1 + l2 + L) % 2 == 0:
        return torch.real(real_cg)
    else:
        return torch.imag(real_cg)


class ClebschGordanReal:
    def __init__(self, lmax: int, device: str = None):
        self.lmax = lmax
        self._cg = {}
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.r2c = {}
        self.c2r = {}
        for L in range(0, lmax + 1):
            self.r2c[L] = _real2complex(L).to(self.device)
            self.c2r[L] = torch.conj(self.r2c[L]).T

        # real-to-complex and complex-to-real transformations as matrices
        for l1 in range(self.lmax + 1):
            for l2 in range(self.lmax + 1):
                for L in range(abs(l1 - l2), min(self.lmax, (l1 + l2)) + 1):
                    rcg = _real_clebsch_gordan_matrix(
                        l1,
                        l2,
                        L,
                        r2c_l1=self.r2c[l1],
                        r2c_l2=self.r2c[l2],
                        c2r_L=self.c2r[L],
                        device=self.device,
                    )

                    # sparsify: take only the non-zero entries (indices
                    # of m1 and m2 components) for each M
                    new_cg = []
                    for M in range(2 * L + 1):
                        cg_nonzero = torch.where(abs(rcg[:, :, M]) > 1e-15)
                        cg_M = torch.zeros(
                            (len(cg_nonzero[0]), 3),
                            # dtype=[(torch.int32, torch.int32, torch.int32)],
                            device=self.device,
                        )
                        cg_M[:, 0] = cg_nonzero[0].type(torch.int)
                        cg_M[:, 1] = cg_nonzero[1].type(torch.int)
                        cg_M[:, 2] = rcg[cg_nonzero[0], cg_nonzero[1], M]
                        new_cg.append(cg_M)

                    self._cg[(l1, l2, L)] = new_cg
        # self._cg.to(self.device)
        # self._cg = {
        #     k: v.to(device=self.device, non_blocking=True) for k, v in self._cg.items()
        # }

    def combine(
        self, y1: torch.tensor, y2: torch.tensor, L: int, combination_string: str
    ) -> torch.tensor:
        """
        Combines two spherical tensors y1, y2 into a single one.
        |Y; LM> = \sum_{m1,m2} <l1 m1 l2 m2|LM> |y1; l1 m1> |y2; l2 m2>
        """
        # automatically infer l1 and l2 from the size of the coefficients vectors
        if y1.device != self.device:
            y1 = y1.to(self.device)
            y2 = y2.to(self.device)
        l1 = (y1.shape[1] - 1) // 2
        l2 = (y2.shape[1] - 1) // 2
        if L > self.lmax or l1 > self.lmax or l2 > self.lmax:
            raise ValueError(
                "Requested CG entry ", (l1, l2, L), " has not been precomputed"
            )

        n_items = y1.shape[0]
        if y1.shape[0] != y2.shape[0]:
            raise IndexError(
                "Cannot combine feature blocks with different number of items"
            )

        # infers the shape of the output using the einsum internals
        ycombine = torch.einsum(combination_string, y1[:, 0, ...], y2[:, 0, ...]).shape
        Y = torch.zeros((n_items, 2 * L + 1) + ycombine[1:], device=self.device)

        if (l1, l2, L) in self._cg:
            for M in range(2 * L + 1):
                for m1, m2, cg in self._cg[(l1, l2, L)][M]:
                    m1 = m1.type(torch.int)
                    m2 = m2.type(torch.int)
                    # print(m1, m2, M, m1.dtype)
                    Y[:, M, ...] += torch.einsum(
                        combination_string, y1[:, m1, ...], y2[:, m2, ...] * cg
                    )

        return Y

    def couple(self, decoupled: Dict, iterate: int = 0) -> Dict:
        """
        Goes from an uncoupled product basis to a coupled basis.
        A (2l1+1)x(2l2+1) matrix transforming like the outer product of
        Y^m1_l1 Y^m2_l2 can be rewritten as a list of coupled vectors,
        each transforming like a Y^M_L.
        This transformation is accomplished through the following relation:

        |L M> = |l1 l2 L M> = \sum_{m1 m2} <l1 m1 l2 m2|L M> |l1 m1> |l2 m2>

        The process can be iterated: a D dimensional array that is the product
        of D Y^m_l can be turned into a set of multiple terms transforming as
        a single Y^M_L.

        Args:
            decoupled: (...)x(2l1+1)x(2l2+1) array containing coefficients that
                       transform like products of Y^l1 and Y^l2 harmonics.
                       Can also be called on a array of higher dimensionality,
                       in which case the result will contain matrices of entries.
                       If the further index also correspond to spherical harmonics,
                       the process can be iterated, and couple() can be called onto
                       its output, in which case the coupling is applied to each
                       entry.

            iterate: calls couple iteratively the given number of times.
                     Equivalent to:

                         couple(couple(... couple(decoupled)))

        Returns:
            coupled: A dictionary tracking the nature of the coupled objects.
                     When called one time, it returns a dictionary containing (l1, l2)
                     [the coefficients of the parent Ylm] which in turns is a
                     dictionary of coupled terms, in the form

                        L:(...)x(2L+1)x(...)

                    When called multiple times, it applies the coupling to each
                    term, and keeps track of the additional l terms, so that,
                    e.g., when called with iterate=1 the return dictionary contains
                    terms of the form

                        (l3,l4,l1,l2) : { L: array }

                    Note that this coupling scheme is different from the
                    NICE-coupling where angular momenta are coupled from
                    left to right as (((l1 l2) l3) l4)... )

                    Thus results may differ when combining more than two angular
                    channels.
        """

        coupled = {}

        # when called on a matrix, turns it into a dict form to which we can
        # apply the generic algorithm
        if not isinstance(decoupled, dict):
            l2 = (decoupled.shape[-1] - 1) // 2
            decoupled = {(): {l2: decoupled}}

        # runs over the tuple of (partly) decoupled terms
        for ltuple, lcomponents in decoupled.items():
            # each is a list of L terms
            for lc in lcomponents.keys():
                # this is the actual matrix-valued coupled term,
                # of shape (..., 2l1+1, 2l2+1), transforming as Y^m1_l1 Y^m2_l2
                dec_term = lcomponents[lc]
                l1 = (dec_term.shape[-2] - 1) // 2
                l2 = (dec_term.shape[-1] - 1) // 2

                # there is a certain redundance: the L value is also the last entry
                # in ltuple
                if lc != l2:
                    raise ValueError(
                        "Inconsistent shape for coupled angular momentum block."
                    )

                # in the new coupled term, prepend (l1,l2) to the existing label
                coupled[(l1, l2) + ltuple] = {}
                for L in range(
                    max(l1, l2) - min(l1, l2), min(self.lmax, (l1 + l2)) + 1
                ):
                    # ensure that Lterm is created on the same device as the dec_term
                    device = dec_term.device
                    if device != self.device:
                        dec_term = dec_term.to(self.device)
                    Lterm = torch.zeros(
                        size=dec_term.shape[:-2] + (2 * L + 1,), device=self.device
                    )
                    for M in range(2 * L + 1):
                        for m1, m2, cg in self._cg[(l1, l2, L)][M]:
                            Lterm[..., M] += (
                                dec_term[..., m1.type(torch.int), m2.type(torch.int)]
                                * cg
                            )
                    coupled[(l1, l2) + ltuple][L] = Lterm

        # repeat if required
        if iterate > 0:
            coupled = self.couple(coupled, iterate - 1)
        return coupled

    def decouple(self, coupled: Dict, iterate: int = 0) -> Dict:
        """
        Transform from coupled to uncoupled basis

        |l1 m1> |l2 m2> = \sum_{L M} <L M |l1 m1 l2 m2> |l1 l2 L M>
        """
        decoupled = {}
        # applies the decoupling to each entry in the dictionary
        for ltuple, lcomponents in coupled.items():
            # the initial pair in the key indicates the decoupled terms that generated
            # the L entries
            l1, l2 = ltuple[:2]

            # shape of the coupled matrix (last entry is the 2L+1 M terms)
            shape = next(iter(lcomponents.values())).shape[:-1]

            dec_term = torch.zeros(
                shape
                + (  # noqa
                    2 * l1 + 1,
                    2 * l2 + 1,
                ),
                device=self.device,
            )
            for L in range(max(l1, l2) - min(l1, l2), min(self.lmax, (l1 + l2)) + 1):
                # supports missing L components, e.g. if they are zero because of symmetry
                if L not in lcomponents:
                    continue
                for M in range(2 * L + 1):
                    for m1, m2, cg in self._cg[(l1, l2, L)][M]:
                        dec_term[..., m1.type(torch.int), m2.type(torch.int)] += (
                            cg * lcomponents[L][..., M]
                        )
            # stores the result with a key that drops the l's we have just decoupled
            if not ltuple[2:] in decoupled:
                decoupled[ltuple[2:]] = {}
            decoupled[ltuple[2:]][l2] = dec_term

        # rinse, repeat
        if iterate > 0:
            decoupled = self.decouple(decoupled, iterate - 1)

        # if we got a fully decoupled state, just return an array
        if ltuple[2:] == ():
            decoupled = next(iter(decoupled[()].values()))
        return decoupled
