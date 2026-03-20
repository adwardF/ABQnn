from typing import Tuple
import torch
from torch import nn


@torch.jit.script
def psi_F_derivates_to_UMAT_2D(F, J, psi_F, psi_FF):
    Kirchhoff = psi_F @ F.T
    Cauchy = Kirchhoff / J

    indx = [(0, 0), (1, 1), (0, 1)]

    EYE = torch.eye(2, dtype=F.dtype, device=F.device)
    Stiff = torch.einsum("i q k s, j q, l s -> i j k l", psi_FF, F, F)
    Stiff += 0.5 * (
        torch.einsum("i k, j l -> i j k l", Kirchhoff, EYE)
        + torch.einsum("i l, j k -> i j k l", Kirchhoff, EYE)
        + torch.einsum("j k, i l -> i j k l", Kirchhoff, EYE)
        - torch.einsum("j l, i k -> i j k l", Kirchhoff, EYE)
    )  # MINUS
    DDSDDE = torch.zeros(3, 3, dtype=F.dtype, device=F.device)
    for ki in range(3):
        for kj in range(3):
            DDSDDE[ki, kj] = Stiff[indx[ki][0], indx[ki][1], indx[kj][0], indx[kj][1]]

    DDSDDE /= J
    Cauchy_v = torch.zeros(3, dtype=F.dtype, device=F.device)

    for ki in range(3):
        Cauchy_v[ki] = Cauchy[indx[ki][0], indx[ki][1]]
    return Cauchy_v, DDSDDE


@torch.jit.script
def psi_F_derivates_to_UMAT_3D(F, J, psi_F, psi_FF):
    Kirchhoff = psi_F @ F.T
    Cauchy = Kirchhoff / J

    assert torch.allclose(Kirchhoff, Kirchhoff.T, atol=1e-6, rtol=1e-4), (
        "Kirchhoff stress symmetry violated"
    )

    indx = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]

    EYE = torch.eye(3, dtype=F.dtype, device=F.device)
    Stiff = torch.einsum("i q k s, j q, l s -> i j k l", psi_FF, F, F)

    Stiff += 0.5 * (
        torch.einsum("i k, j l -> i j k l", Kirchhoff, EYE)
        + torch.einsum("i l, j k -> i j k l", Kirchhoff, EYE)
        + torch.einsum("j k, i l -> i j k l", Kirchhoff, EYE)
        - torch.einsum("j l, i k -> i j k l", Kirchhoff, EYE)
    )

    assert torch.allclose(Stiff, Stiff.permute(1, 0, 3, 2), atol=1e-6, rtol=1e-4), (
        "Stiffness tensor symmetry violated A"
    )
    assert torch.allclose(Stiff, Stiff.permute(2, 3, 1, 0), atol=1e-6, rtol=1e-4), (
        f"Stiffness tensor symmetry violated B {Stiff - Stiff.permute(2, 3, 1, 0)} {F}"
    )

    DDSDDE = torch.zeros(6, 6, dtype=F.dtype, device=F.device)
    for ki in range(6):
        for kj in range(6):
            DDSDDE[ki, kj] = Stiff[indx[ki][0], indx[ki][1], indx[kj][0], indx[kj][1]]

    DDSDDE /= J
    Cauchy6 = torch.zeros(6, dtype=F.dtype, device=F.device)

    for ki in range(6):
        Cauchy6[ki] = Cauchy[indx[ki][0], indx[ki][1]]
    return Cauchy6, DDSDDE


@torch.jit.script
def psi_C_derivates_to_UMAT_3D(
    C: torch.Tensor,
    F: torch.Tensor,
    J: torch.Tensor,
    psi_C: torch.Tensor,
    psi_CC: torch.Tensor,
):
    indx_3D = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    Kirchhoff = 2 * F @ psi_C @ F.T
    Cauchy = Kirchhoff / J

    EYE = torch.eye(3, dtype=C.dtype, device=C.device)
    Stiff = 4 * torch.einsum(
        "p q r s, i p, j q, k r, l s -> i j k l", psi_CC, F, F, F, F
    )
    Stiff += 0.5 * (
        torch.einsum("i k, j l -> i j k l", Kirchhoff, EYE)
        + torch.einsum("i l, j k -> i j k l", Kirchhoff, EYE)
        + torch.einsum("j k, i l -> i j k l", Kirchhoff, EYE)
        + torch.einsum("j l, i k -> i j k l", Kirchhoff, EYE)
    )
    DDSDDE = torch.zeros(6, 6, dtype=C.dtype, device=C.device)
    for ki in range(6):
        for kj in range(6):
            DDSDDE[ki, kj] = Stiff[
                indx_3D[ki][0], indx_3D[ki][1], indx_3D[kj][0], indx_3D[kj][1]
            ]

    DDSDDE /= J
    Cauchy_vec = torch.zeros(6, dtype=C.dtype, device=C.device)

    for ki in range(6):
        Cauchy_vec[ki] = Cauchy[indx_3D[ki][0], indx_3D[ki][1]]
    return Cauchy_vec, DDSDDE


# Neohookean 3D model for testing
class NH3D(nn.Module):
    def __init__(self):
        super(NH3D, self).__init__()

    # Pretend a neural network model predicts psi and P given F
    # the gradient of P w.r.t F should be preserved
    def model_forward(
        self, F_in: torch.Tensor, mat_par: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert mat_par.shape == (2,), (
            f"Expected mat_par of shape (2,), got {mat_par.shape}, {mat_par}"
        )

        c1 = mat_par[0]
        c2 = mat_par[1]

        F = F_in
        FinvT = torch.inverse(F).T
        I1 = torch.trace(F.T @ F)
        J = torch.det(F)

        psi = c1 * (J ** (-2 / 3) * I1 - 3) + c2 * (J - 1) ** 2

        P = (
            2 * c1 * J ** (-2 / 3) * (F - (1 / 3) * I1 * FinvT)
            + 2 * c2 * (J - 1) * J * FinvT
        )

        return psi, P

    def forward(
        self, F_in: torch.Tensor, mat_par: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        F = F_in.clone().requires_grad_(True)
        psi, P = self.model_forward(F, mat_par)

        P_F = torch.zeros(3, 3, 3, 3, dtype=F_in.dtype, device=F_in.device)
        for i in range(3):
            for j in range(3):
                grad_P_F_ij = torch.autograd.grad([P[i, j]], [F], retain_graph=True)[0]
                assert grad_P_F_ij is not None
                P_F[i, j, :, :] = grad_P_F_ij

        Cauchy, DDSDDE = psi_F_derivates_to_UMAT_3D(F_in, torch.det(F_in), P, P_F)

        return psi.detach(), Cauchy.detach(), DDSDDE.detach()


class NH_PE(nn.Module):
    def __init__(self):
        super(NH_PE, self).__init__()

    # Pretend a neural network model predicts psi and P given (2D) F
    # the gradient of P w.r.t F should be preserved
    def model_forward(self, F_in: torch.Tensor, mat_par: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        F = F_in
        FinvT = torch.inverse(F).T
        I1 = torch.trace(F.T @ F) + 1.0
        J = torch.det(F[:2, :2])

        c1 = mat_par[0]
        c2 = mat_par[1]

        psi = c1 * J ** (-2 / 3) * I1 - 3 + c2 * (J - 1) ** 2

        P = 2 * c1 * J ** (-2 / 3) * (F - (1 / 3) * I1 * FinvT) + 2 * c2 * (J - 1) * J * FinvT

        return psi, P

    def forward(
        self, F_in: torch.Tensor, mat_par: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        F22 = F_in[:2, :2].clone().requires_grad_(True)
        psi, P = self.model_forward(F22, mat_par)

        P_F = torch.zeros(2, 2, 2, 2, dtype=F_in.dtype, device=F_in.device)
        for i in range(2):
            for j in range(2):
                grad_P_F_ij = torch.autograd.grad([P[i, j]], [F22], retain_graph=True)[
                    0
                ]
                assert grad_P_F_ij is not None
                P_F[i, j, :, :] = grad_P_F_ij

        Cauchy, DDSDDE = psi_F_derivates_to_UMAT_2D(
            F22.detach(), torch.det(F22.detach()), P, P_F
        )

        # Ignore all out-of-plane components
        Cauchy4 = torch.zeros(4, dtype=Cauchy.dtype, device=Cauchy.device)
        DDSDDE44 = torch.zeros(4, 4, dtype=DDSDDE.dtype, device=DDSDDE.device)

        Cauchy4[0] = Cauchy[0]
        Cauchy4[1] = Cauchy[1]
        Cauchy4[3] = Cauchy[2]

        DDSDDE44[:2, :2] = DDSDDE[:2, :2]
        DDSDDE44[:2, 3] = DDSDDE[:2, 2]
        DDSDDE44[3, :2] = DDSDDE[2, :2]
        DDSDDE44[2, 2] = 1.0
        DDSDDE44[3, 3] = DDSDDE[2, 2]

        return psi.detach(), Cauchy4.detach(), DDSDDE44.detach()


class VUMATBatchNH3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, F_batch: torch.Tensor, mat_par: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if F_batch.dim() != 3 or F_batch.size(-1) != 3 or F_batch.size(-2) != 3:
            raise RuntimeError("Expected F_batch shape [nblock, 3, 3]")
        if mat_par.numel() < 2:
            raise RuntimeError("Expected at least 2 material parameters")

        c1 = mat_par[0]
        c2 = mat_par[1]

        EYE = torch.eye(3, dtype=F_batch.dtype, device=F_batch.device)
        
        C = F_batch.swapaxes(-1, -2) @ F_batch

        J = torch.det(F_batch)
        I1 = C[:, 0, 0] + C[:, 1, 1] + C[:, 2, 2]

        energy = c1 * (J ** (-2 / 3) * I1 - 3.0) + c2 * (J - 1) ** 2

        co_rot_Cauchy = (
            c1
            * J[:, None, None] ** (-5 / 3)
            * (C - I1[:, None, None] / 3 * EYE[None, :, :])
            + (2 * c2 * (J - 1))[:, None, None] * EYE[None, :, :]
        )

        stress_vumat = torch.stack(
            [
                co_rot_Cauchy[:, 0, 0],
                co_rot_Cauchy[:, 1, 1],
                co_rot_Cauchy[:, 2, 2],
                co_rot_Cauchy[:, 0, 1],
                co_rot_Cauchy[:, 1, 2],
                co_rot_Cauchy[:, 2, 0],
            ],
            dim=1,
        )

        return energy, stress_vumat

class VUMATBatchNHPE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, F_batch: torch.Tensor, mat_par: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if F_batch.dim() != 3 or F_batch.size(-1) != 3 or F_batch.size(-2) != 3:
            raise RuntimeError("Expected F_batch shape [nblock, 3, 3]")
        if mat_par.numel() < 2:
            raise RuntimeError("Expected at least 2 material parameters")

        c1 = mat_par[0]
        c2 = mat_par[1]
        
        EYE = torch.eye(3, dtype=F_batch.dtype, device=F_batch.device)
        
        C = F_batch.swapaxes(-1, -2) @ F_batch

        J = torch.det(F_batch)
        I1 = C[:, 0, 0] + C[:, 1, 1] + C[:, 2, 2]

        energy = c1 * (J ** (-2 / 3) * I1 - 3.0) + c2 * (J - 1) ** 2

        co_rot_Cauchy = (
            c1
            * J[:, None, None] ** (-5 / 3)
            * (C - I1[:, None, None] / 3 * EYE[None, :, :])
            + (2 * c2 * (J - 1))[:, None, None] * EYE[None, :, :]
        )

        stress_vumat = torch.stack(
            [
                co_rot_Cauchy[:, 0, 0],
                co_rot_Cauchy[:, 1, 1],
                co_rot_Cauchy[:, 2, 2],
                co_rot_Cauchy[:, 0, 1],
            ],
            dim=1,
        )

        return energy, stress_vumat

if __name__ == "__main__":
    model = NH3D()
    scripted_model = torch.jit.script(model)
    scripted_model = torch.jit.optimize_for_inference(scripted_model)
    # should be executed in the root directory
    scripted_model.save("models/NH_3D.pt")

    model = NH_PE()
    scripted_model = torch.jit.script(model)
    scripted_model = torch.jit.optimize_for_inference(scripted_model)
    # should be executed in the root directory
    scripted_model.save("models/NH_PE.pt")

    model = VUMATBatchNH3D()
    scripted_model = torch.jit.script(model)
    scripted_model = torch.jit.optimize_for_inference(scripted_model)
    scripted_model.save("models/VUMAT_NH_3D.pt")

    model = VUMATBatchNHPE()
    scripted_model = torch.jit.script(model)
    scripted_model = torch.jit.optimize_for_inference(scripted_model)
    scripted_model.save("models/VUMAT_NH_PE.pt")

    # --- test run ---
    """ F_test = torch.eye(3)[None, :, :] + torch.randn(100, 3, 3) * 0.2
    F = F_test[0, :, :]

    m_3D = NH3D()

    psi, Cauchy, DDSDDE = m_3D(F, torch.tensor([1.0, 10.0]))
    print("=== NH_3D_test ===")
    print(f"{F=}, {psi=}, {Cauchy=}, {DDSDDE=}")

    m_PE = NH_PE()
    F = F_test[0]
    psi, Cauchy, DDSDDE = m_PE(F, torch.tensor([1.0, 10.0]))
    print("=== NH_PE_test ===")
    print(f"{F=}, {psi=}, {Cauchy=}, {DDSDDE=}") """

    """ for i in range(100):
        F = F_test[i,:,:].double()
        psi1, Cauchy1, DDSDDE1 = m_3D(F)
        psi2, Cauchy2, DDSDDE2 = m_3D_ref(F)
        
        assert torch.allclose(Cauchy1, Cauchy2, atol=1e-10, rtol=1e-5),\
            f"Cauchy mismatch at {i}, {Cauchy1=} vs {Cauchy2=}, {Cauchy1 - Cauchy2=}"
        assert torch.allclose(DDSDDE1, DDSDDE2, atol=1e-10, rtol=1e-5),\
            f"DDSDDE mismatch at {i}, {DDSDDE1=} vs {DDSDDE2=}, {DDSDDE1 - DDSDDE2=}"
    """
