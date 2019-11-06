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
    save = False
    image = renderer.render()
    if save:
        image.save("pose_1.png")
    else:
        image.show()

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
    if save:
        image.save("pose_2.png")
    else:
        image.show()

    # Get neural network probabilities.
    with torch.no_grad():
        out = model(image)

    probs = torch.nn.functional.softmax(out, dim=1)
    print(probs[0][target_class].item())

    # Get depth map.
    depth_map = renderer.get_depth_map()
    if save:
        depth_map.save("depth_map.png")
    else:
        depth_map.show()

    # Get normal map.
    norm_map = renderer.get_normal_map()
    if save:
        norm_map.save("normal_map.png")
    else:
        norm_map.show()

    # Get screen coordinates of vertices.
    (screen_coords, screen_img) = renderer.get_vertex_screen_coordinates()
    if save:
        screen_img.save("screen_coords.png")
    else:
        screen_img.show()

    # Use azimuth and elevation to generate rotation matrix.
    R_obj = gen_rotation_matrix_from_azim_elev(np.pi / 4, np.pi / 4)
    renderer.prog["R_obj"].write(R_obj.T.astype("f4").tobytes())

    image = renderer.render()
    if save:
        image.save("camera.png")
    else:
        image.show()
