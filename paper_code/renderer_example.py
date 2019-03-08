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
    image.save("first.png")
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
    image.save("second.png")
    # image.show()

    # Get neural network probabilities.
    with torch.no_grad():
        out = model(image)

    probs = torch.nn.functional.softmax(out, dim=1)
    print(probs[0][target_class].item())

    # Get screen coordinates of vertices.
    (screen_coords, screen_img) = renderer.get_vertex_screen_coordinates()
    screen_img.save("third.png")
    # screen_img.show()
