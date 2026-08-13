/* Expose the single-AO-leg RHF potential derivative needed by molecular EPH. */
#include "psi4/psi4-dec.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/factory.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/twobody.h"
#include "psi4/libmints/wavefunction.h"

namespace psi { namespace qdynamics_eph_deriv {

extern "C" PSI_API int read_options(std::string name, Options& options) {
    if (name == "QDYNAMICS_EPH_DERIV" || options.read_globals()) {
        options.add_int("ATOM", 0);
    }
    return true;
}

extern "C" PSI_API SharedWavefunction qdynamics_eph_deriv(
        SharedWavefunction wfn, Options& options) {
    const int atom = options.get_int("ATOM");
    auto basis = wfn->basisset();
    const int nbf = basis->nbf();
    if (atom < 0 || atom >= basis->molecule()->natom()) {
        throw PSIEXCEPTION("QDYNAMICS_EPH_DERIV: ATOM is out of range");
    }
    for (int shell = 0; shell < basis->nshell(); ++shell) {
        if (basis->shell(shell).nfunction() !=
                basis->shell(shell).ncartesian()) {
            throw PSIEXCEPTION(
                "QDYNAMICS_EPH_DERIV currently requires Cartesian shells");
        }
    }

    auto density_a = wfn->Da();
    auto density_b = wfn->Db();

    std::vector<SharedMatrix> result;
    for (int xyz = 0; xyz < 3; ++xyz) {
        auto value = std::make_shared<Matrix>("AO LEG DERIVATIVE", nbf, nbf);
        result.push_back(value);
    }

    IntegralFactory factory(basis);
    auto eri1 = std::shared_ptr<TwoBodyAOInt>(factory.eri(1));
    auto Da = density_a->pointer();
    auto Db = density_b->pointer();
    for (int P = 0; P < basis->nshell(); ++P) {
        const auto& ps = basis->shell(P);
        if (ps.ncenter() != atom) continue;
        for (int Q = 0; Q < basis->nshell(); ++Q) {
            const auto& qs = basis->shell(Q);
            for (int R = 0; R < basis->nshell(); ++R) {
                const auto& rs = basis->shell(R);
                for (int S = 0; S < basis->nshell(); ++S) {
                    const auto& ss = basis->shell(S);
                    eri1->compute_shell_deriv1(P, Q, R, S);
                    const auto& buffers = eri1->buffers();
                    size_t index = 0;
                    for (int p = 0; p < ps.nfunction(); ++p) {
                        const int mu = ps.function_index() + p;
                        for (int q = 0; q < qs.nfunction(); ++q) {
                            const int nu = qs.function_index() + q;
                            for (int r = 0; r < rs.nfunction(); ++r) {
                                const int ka = rs.function_index() + r;
                                for (int s = 0; s < ss.nfunction(); ++s, ++index) {
                                    const int la = ss.function_index() + s;
                                    for (int xyz = 0; xyz < 3; ++xyz) {
                                        const double integral = buffers[xyz][index];
                                        // dJ from total density; dK_alpha from Da.
                                        result[xyz]->add(mu, nu,
                                            integral * (Da[ka][la] + Db[ka][la]));
                                        result[xyz]->add(mu, la,
                                            -integral * Da[nu][ka]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (int xyz = 0; xyz < 3; ++xyz) {
        wfn->set_array_variable(
            "QDYNAMICS EPH TWO ELECTRON LEG " + std::to_string(xyz),
            result[xyz]);
    }
    return wfn;
}

}}  // namespace psi::qdynamics_eph_deriv
