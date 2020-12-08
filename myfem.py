from fenics_shells import *
from fenics import *
from fenics_adjoint import *
import numpy as np
from ufl import RestrictedElement, operators
from torch_fenics import *
import torch
from torch import optim
import torch.nn as nn

def main():
    bsz = 1
    node_count = 2655
    it_count = 150
    f_list = []

    model = Net(simulator, node_count, bsz)
    optimizer = optim.Adagrad(model.parameters(), lr=0.1)
    for i in range(it_count):
        out = model()
        optimizer.zero_grad()
        loss = out
        loss.backward()
        optimizer.step()
        print('\rIteration:{} | Cost:{}'.format(i+1, loss.item()), end='')
        f_list.append(loss.item())

    return model.state_dict()['x'], f_list

class Plate(torch_fenics.FEniCSModule):
    def __init__(self, mesh, validation_mode=False):
        super().__init__()
        self.validation_mode = validation_mode
        self.mesh = mesh
        self.U = VectorFunctionSpace(self.mesh, "CG", 1)
        self.X = FunctionSpace(self.mesh, "CG", 1)
        self.P = FunctionSpace(self.mesh, "Lagrange", 1)

    def input_templates(self):
        return Constant((0, 0)), Function(self.X)

    def solve(self, T, x):

        # Elasticity constants
        h = 1.
        E1 = 40.
        E2 = 1.
        G12 = 0.5
        G13 = G12
        nu12 = 0.25
        G23 = 0.4
        n_layers = 1
        hs = h*np.ones(n_layers)/n_layers
        X = FunctionSpace(self.mesh, 'CG', 1)
        x_ = project(x, X)

        # Should be represented by vector.
        theta = [x_]
        
        # Bondary conditions
        facets = MeshFunction("size_t", self.mesh, 1)
        facets.set_all(0)
        Top().mark(facets, 1)
        Left().mark(facets, 2)
        Bottom().mark(facets, 3)
        ds = Measure('ds', subdomain_data=facets)
        bc = [DirichletBC(self.U.sub(0), Constant(0.), facets, 2),
              DirichletBC(self.U.sub(1), Constant(0.), facets, 3)]

        # Laminates stiffness
        self.A, self.B, self.D = laminates.ABD(E1, E2, G12, nu12, hs, theta)
        self.F = laminates.F(G13, G23, hs, theta)

        u_trial = TrialFunction(self.U)
        u_test = TestFunction(self.U)
        u = Function(self.U)

        a = self.energydense(u_trial, u_test) * dx
        l = dot(T, u_test) * ds(0)

        solve(a == l, u, bc)
        p = project(self.energydense(u, u), self.P)

        if self.validation_mode:
            Orient = VectorFunctionSpace(self.mesh, "CG", 1)
            v = as_vector((operators.cos(x), operators.sin(x)))
            orient = project(v, Orient)
            xdmffile1 = XDMFFile('output/Orient.xdmf')
            xdmffile1.write(orient)
            xdmffile3 = XDMFFile('output/displacement.xdmf')
            xdmffile3.write(u)
            Sigma = TensorFunctionSpace(self.mesh, 'CG', 1)
            sig_ = project(self.inplane_sig(u), Sigma)
            xdmffile5 = XDMFFile('output/sigma.xdmf')
            xdmffile5.write(sig_)

        return p

    def energydense(self, u, v):
        # Inplane response
        in_eps_trial = strain_to_voigt(sym(grad(u)))
        in_sig_trial = stress_from_voigt(self.A * in_eps_trial)
        in_eps_test = sym(grad(v))
        in_energy_dense = inner(in_sig_trial, in_eps_test)

        return in_energy_dense
    def inplane_sig(self, u):
        in_eps = strain_to_voigt(sym(grad(u)))
        return stress_from_voigt(self.A * in_eps)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] > 0.999 and on_boundary
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < 0.001 and on_boundary
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] < 0.001 and on_boundary

def simulator(z1):
    mesh = Mesh('sample.xml')
    problem = Plate(mesh)
    T = torch.tensor([[1e-1, 0.]], requires_grad=False, dtype=torch.float64)
    p = problem(T, z1)
    return p.sum()

class Net(nn.Module):
    def __init__(self, func, node_count, bsz=1):
        super(Net, self).__init__()
        self.func = func
        self.x = nn.Parameter(torch.rand((bsz, node_count), dtype=torch.float64) * 0.0)
    def forward(self):
        return self.func(self.x)

if __name__ == '__main__':
    main()