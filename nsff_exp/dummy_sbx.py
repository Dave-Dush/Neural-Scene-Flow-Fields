import numpy as np

fwd = np.load("/usr/stud/dave/storage/user/dave/kid-running/dense/flow_i1/00000_fwd.npz")

print(fwd["flow"].shape, fwd["mask"].shape)
print(fwd["flow"], fwd["mask"])