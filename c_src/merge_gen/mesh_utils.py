import os
def write_obj(folder,
              v_pos=None,
              v_nrm=None,
              v_tex=None,
              t_pos_idx=None,
              t_nrm_idx=None,
              t_tex_idx=None,
              save_material=True,
              file_name = 'mesh.obj'):
    obj_file = os.path.join(folder, file_name)
    print("Writing mesh: ", obj_file)
    with open(obj_file, "w") as f:
        f.write("mtllib mesh.mtl\n")
        f.write("g default\n")

        v_pos = v_pos if v_pos is not None else None
        v_nrm = v_nrm if v_nrm is not None else None
        v_tex = v_tex if v_tex is not None else None

        t_pos_idx = t_pos_idx if t_pos_idx is not None else None
        t_nrm_idx = t_nrm_idx if t_nrm_idx is not None else None
        t_tex_idx = t_tex_idx if t_tex_idx is not None else None

        print("    writing %d vertices" % len(v_pos))
        for v in v_pos:
            f.write('v {} {} {} \n'.format(v[0], v[1], v[2]))

        if v_tex is not None:
            print("    writing %d texcoords" % len(v_tex))
            assert(len(t_pos_idx) == len(t_tex_idx))
            for v in v_tex:
                f.write('vt {} {} \n'.format(v[0], 1.0 - v[1]))

        # if v_nrm is not None:
        #     print("    writing %d normals" % len(v_nrm))
        #     assert(len(t_pos_idx) == len(t_nrm_idx))
        #     for v in v_nrm:
        #         f.write('vn {} {} {}\n'.format(v[0], v[1], v[2]))

        # faces
        f.write("s 1 \n")
        f.write("g pMesh1\n")
        f.write("usemtl defaultMat\n")

        # Write faces
        print("    writing %d faces" % len(t_pos_idx))
        for i in range(len(t_pos_idx)):
            f.write("f ")
            for j in range(3):
                f.write(' %s/%s/%s' % (str(t_pos_idx[i][j]+1), '' if v_tex is None else str(t_tex_idx[i][j]+1), '' if v_nrm is None else str(t_nrm_idx[i][j]+1)))
            f.write("\n")


    print("Done exporting mesh")
