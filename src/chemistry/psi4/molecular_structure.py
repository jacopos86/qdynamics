


#
#   This module performs geometry optimization
#
#   define the molecule structure
#



def geometry_optimization():
    init_struct = psi4_molecule_class(p.coordinate_file)
    basis_name = p.basis_file_name[:-4]
    name = "scf/" + basis_name
    E_SCF_psi, wfn_psi = psi4.optimize(name, molecule=init_struct.geometry, return_wfn=True)
    # save data on file
    natoms= init_struct.geometry.natom()
    # open file
    f = open(p.optimized_coordinate_file, 'w')
    f.write("%d\n" % natoms)
    f.write("Angstrom\n")
    for i in range(natoms):
        symb = init_struct.geometry.label(i).lower()
        symb = symb.capitalize()
        # bohr
        Ri = np.array([init_struct.geometry.x(i),init_struct.geometry.y(i),init_struct.geometry.z(i)])
        Ri[:] = Ri[:] * bohr_to_ang
        # ang units
        f.write(symb + "       " + str(Ri[0]) + "       " + str(Ri[1]) + "       " + str(Ri[2]) + "\n")
    f.close()
    return E_SCF_psi, wfn_psi