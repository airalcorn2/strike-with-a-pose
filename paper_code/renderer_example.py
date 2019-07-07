from renderer import Renderer
from strike_utils import *


if __name__ == "__main__":
    # Initialize neural network.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(device).to(device)

    # Initialize renderer.
    renderer = Renderer(
        "objects/Jeep/Jeep.obj", "objects/Jeep/Jeep.mtl", "backgrounds/medium.jpg"
    )

    # Render scene.
    image = renderer.render()
    image.save("pose_a.png")
    # image.show()

    # Get neural network probabilities.
    with torch.no_grad():
        out = model(image)

    probs = torch.nn.functional.softmax(out, dim=1)
    target_class = 609
    print(probs[0][target_class].item())

    # Alter renderer parameters.
    R_obj = gen_rotation_matrix(np.pi / 4, np.pi / 4, 0)
    renderer.prog["R_obj"].write(R_obj.T.astype("f4").tobytes())
    renderer.prog["x"].value = 0.5
    renderer.prog["y"].value = 0.5
    renderer.prog["z"].value = -4
    renderer.prog["amb_int"].value = 0.3
    renderer.prog["dif_int"].value = 0.9
    DirLight = np.array([1.0, 1.0, 1.0])
    DirLight /= np.linalg.norm(DirLight)
    renderer.prog["DirLight"].value = tuple(DirLight)

    # Render new scene.
    image = renderer.render()
    image.save("pose_b.png")
    # image.show()

    # Get depth map.
    depth = np.frombuffer(
        renderer.fbo2.read(attachment=-1, dtype="f4"), dtype=np.dtype("f4")
    )
    depth = 1 - depth.reshape(renderer.window_size)
    min_pos = depth[depth > 0].min()
    depth[depth > 0] = depth[depth > 0] - min_pos
    depth_normed = depth / depth.max()
    depth_map = np.uint8(255 * depth_normed)
    depth_map = ImageOps.flip(Image.fromarray(depth_map, "L").convert("RGB"))
    depth_map.save("depth_map.png")
    # depth_map.show()

    # Get normal map.
    # See: https://stackoverflow.com/questions/5281261/generating-a-normal-map-from-a-height-map
    # and: https://stackoverflow.com/questions/34644101/calculate-surface-normals-from-depth-image-using-neighboring-pixels-cross-produc
    # and: https://en.wikipedia.org/wiki/Normal_mapping#How_it_works.
    depth_pad = np.pad(depth_normed, 1, "constant")
    (dx, dy) = (1 / depth.shape[1], 1 / depth.shape[0])
    dz_dx = (depth_pad[1:-1, 2:] - depth_pad[1:-1, :-2]) / (2 * dx)
    dz_dy = (depth_pad[2:, 1:-1] - depth_pad[:-2, 1:-1]) / (2 * dy)
    norms = np.stack([-dz_dx.flatten(), -dz_dy.flatten(), np.ones(dz_dx.size)])
    magnitudes = np.linalg.norm(norms, axis=0)
    norms /= magnitudes
    norms = norms.T
    norms[:, :2] = 255 * (norms[:, :2] + 1) / 2
    norms[:, 2] = 127 * norms[:, 2] + 128
    norms = np.uint8(norms).reshape((*depth.shape, 3))
    norm_map = ImageOps.flip(Image.fromarray(norms))
    norm_map.save("normal_map.png")
    # norm_map.show()

    # Get neural network probabilities.
    with torch.no_grad():
        out = model(image)

    probs = torch.nn.functional.softmax(out, dim=1)
    print(probs[0][target_class].item())

    # Get screen coordinates of vertices.
    (screen_coords, screen_img) = renderer.get_vertex_screen_coordinates()
    screen_img.save("screen_coords.png")
    # screen_img.show()
