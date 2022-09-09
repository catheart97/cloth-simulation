#!/usr/bin/env python3

import os
import numpy
import array
import vedo
import vedo.shapes

def get(i, j, GRID_DIM):
    return j * GRID_DIM + i

def build_faces(GRID_DIM):
    faces = []
    for i in range(GRID_DIM - 1):
        for j in range(GRID_DIM - 1):
            faces.append([
                get(i, j, GRID_DIM), 
                get(i, j + 1, GRID_DIM), 
                get(i + 1, j + 1, GRID_DIM),
                get(i + 1, j, GRID_DIM)
            ])
    return faces

if __name__ == "__main__":

    # ensure correct directory paths
    root = os.path.abspath(os.path.dirname(__file__))
    path = root + os.sep + "pngs" + os.sep
    os.makedirs(path, exist_ok=True)

    # read in data
    with open(root + "/../data.bin", "rb") as f:
        data = numpy.array(array.array("d", f.read()))

    # config
    GRID_DIM = int(data[0])
    SIZE = 400 # px

    # prepare dataset
    endpoint = GRID_DIM * GRID_DIM * 3
    endpoint = ((data.shape[0] - 1) // endpoint) * endpoint
    # print(f"endpoint : {endpoint}")
    dataset = data[1:endpoint + 1].reshape((-1, 3, GRID_DIM * GRID_DIM))

    # setup cloth "look"
    cloth_grid = vedo.Mesh([list(zip(*dataset[0])), build_faces(GRID_DIM)], c='lightblue', alpha=.9)
    cloth_grid.wireframe(False).lw(0).bc('blue9').flat()

    # setup sphere looks
    with open(root + "/../scene.bin", "rb") as f:
        scene = numpy.array(array.array("d", f.read())).reshape(-1, 4)
    spheres = [
        vedo.shapes.Sphere((v[0], v[1] ,v[2]), r= v[3] * 0.98, c="r5") for v in scene
    ]

    for (c, (x, y, z)) in enumerate(dataset):
        cloth_grid.points(list(zip(x, y, z)))
        plt = vedo.Plotter(offscreen=True)
        # cloth_grid_shadow = cloth_grid.addShadow(z=-1.5, culling=-1)
        # cloth_grid_shadow_ = cloth_grid.addShadow(z=-1.5, culling=1)
        # sphere_shadow = [s.addShadow(z=-1.5) for s in spheres] + [cloth_grid_shadow, cloth_grid_shadow_]
        plt.show(cloth_grid, *spheres, camera={'pos' : (7,7,7)}, viewup='z', size=(SIZE,SIZE))
        plt.screenshot(path + f"{c:04d}.png")
    print("- All done!")
