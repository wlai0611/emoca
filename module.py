def make_obj(verts,triangles,filename):
  with open(filename, "w") as f:

    for v in verts:
        f.write(f"v {v[0]} {v[1]} {v[2]}\n")

    for tri in triangles:
        tri = tri + 1
        f.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")

