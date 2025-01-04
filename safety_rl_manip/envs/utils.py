import numpy as np

def signed_dist_fn_rectangle(grid_x, x_target_min, x_target_max, obstacle=False, plot=False):
    # Compute distances to each edge of the rectangle
    dist_from_walls = np.maximum(x_target_min - grid_x, grid_x - x_target_max)
    signed_distance_grid = np.max(dist_from_walls, axis=-1)
    if obstacle:
        signed_distance_grid = -1*signed_distance_grid
    return signed_distance_grid

def signed_dist_fn_rectangle_obstacle(grid_x, x_target_min, x_target_max, obstacle=True):
    # Compute distances to each edge of the rectangle
    # g(x)>0 is obstacle
    dist_from_walls = np.minimum(grid_x - x_target_min, x_target_max - grid_x)
    signed_distance_grid = np.min(dist_from_walls, axis=-1)
    return signed_distance_grid

def create_grid(x_min, x_max, N_x):
    X = [np.linspace(x_min[i], x_max[i], N_x[i]) for i in range(len(x_min))]
    grid = np.meshgrid(*X, indexing='ij')
    return np.stack(grid, axis=-1)

def create_centered_polygon_with_halfsize(size_x: float, size_y: float):
    return np.array([
        [-size_x, -size_y],
        [-size_x, size_y],
        [size_x, size_y],
        [size_x, -size_y],
    ])

import mujoco
from robosuite.models.base import MujocoModel
def get_contacts(env, model):
    """
    Checks for any contacts with @model (as defined by @model's contact_geoms) and returns the set of
    geom names currently in contact with that model (excluding the geoms that are part of the model itself).
    Args:
        sim (MjSim): Current simulation model
        model (MujocoModel): Model to check contacts for.
    Returns:
        set: Unique geoms that are actively in contact with this model.
    Raises:
        AssertionError: [Invalid input type]
    """
    # Make sure model is MujocoModel type
    assert isinstance(model, MujocoModel), "Inputted model must be of type MujocoModel; got type {} instead!".format(
        type(model)
    )
    contact_set = set()
    for contact in env.data.contact[: env.data.ncon]:
        # check contact geom in geoms; add to contact set if match is found
        # g1, g2 = env.model.geom_id2name(contact.geom1), env.model.geom_id2name(contact.geom2)

        g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

        if g1 in model.contact_geoms and g2 not in model.contact_geoms:
            contact_set.add(g2)
        elif g2 in model.contact_geoms and g1 not in model.contact_geoms:
            contact_set.add(g1)
    return contact_set

def check_contact(env, geoms_1, geoms_2=None):
    """
    Finds contact between two geom groups.
    Args:
        geoms_1 (str or list of str or MujocoModel): an individual geom name or list of geom names or a model. If
            a MujocoModel is specified, the geoms checked will be its contact_geoms
        geoms_2 (str or list of str or MujocoModel or None): another individual geom name or list of geom names.
            If a MujocoModel is specified, the geoms checked will be its contact_geoms. If None, will check
            any collision with @geoms_1 to any other geom in the environment
    Returns:
        bool: True if any geom in @geoms_1 is in contact with any geom in @geoms_2.
    """
    # Check if either geoms_1 or geoms_2 is a string, convert to list if so
    if type(geoms_1) is str:
        geoms_1 = [geoms_1]
    elif isinstance(geoms_1, MujocoModel):
        geoms_1 = geoms_1.contact_geoms
    if type(geoms_2) is str:
        geoms_2 = [geoms_2]
    elif isinstance(geoms_2, MujocoModel):
        geoms_2 = geoms_2.contact_geoms
    for i in range(env.data.ncon):
        contact = env.data.contact[i]
        # check contact geom in geoms
        c1_in_g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) in geoms_1
        c2_in_g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) in geoms_2 if geoms_2 is not None else True
        # check contact geom in geoms (flipped)
        c2_in_g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) in geoms_1
        c1_in_g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) in geoms_2 if geoms_2 is not None else True
        if (c1_in_g1 and c2_in_g2) or (c1_in_g2 and c2_in_g1):
            return True
    return False


from PIL import Image, ImageDraw, ImageFont
import pillow_heif, os, cv2

def save_png_from_heic(img_path):
    heic_files = [f for f in os.listdir(img_path) if f.lower().endswith(".heic")]
    for i, heic_file in enumerate(heic_files):
        file = pillow_heif.open_heif(img_path+heic_file)
        image = Image.frombytes(
            file.mode, 
            file.size, 
            file.data, 
            "raw"
        )
        scale = 0.1  # Scale down to 50%
        new_width = int(image.width * scale)
        new_height = int(image.height * scale)
        low_res_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        low_res_image.save(img_path+f"img_low_res_{i}.png", "PNG")

def append_and_save_png(img_path, png_files, save_img_name):
    images = [Image.open(img_path+png_file) for png_file in png_files]

    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create a blank canvas
    combined_image = Image.new("RGBA", (total_width, max_height))

    # Paste images onto the canvas
    x_offset = 0
    for i, img in enumerate(images):

        draw = ImageDraw.Draw(img)
        text = f"Image {i}"
        position = (10, 10)  # Top-left corner for text
        text_color = (255, 255, 255)  # White color (R, G, B)
        # Add the text to the image
        draw.text(position, text, fill=text_color, font=ImageFont.load_default())
        # draw.text(position, text, fill=text_color, font=ImageFont.truetype("arial.ttf", 40))

        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width

    # Save the combined image
    combined_image.save(img_path+f"{save_img_name}.png", "PNG")

def mov_to_pngs(mov_path):
    output_folder = os.path.dirname(mov_path)
    cap = cv2.VideoCapture(mov_path)

    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return
    
    scale_factor = 0.25
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        if frame_count < 100:
            frame_count += 1
            continue

        if frame_count % 5 == 0:
            original_height, original_width = frame.shape[:2]
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            new_resolution = (new_width, new_height)

            resized_frame = cv2.resize(frame, new_resolution)
            # Save each frame as a PNG
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_filename, resized_frame)
        frame_count += 1
    
    cap.release()
    print(f"Saved {frame_count} frames to {output_folder}")

import textwrap
def add_text_to_img(img, text):

    if isinstance(img, str):
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif not isinstance(img, np.ndarray):
        raise ValueError("Input must be either a file path (str) or a NumPy array.")

    # Font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_thickness = 1
    text_color = (0, 0, 0)  # Black text

    text_height = cv2.getTextSize("Sample", font, font_scale, font_thickness)[0][1]

    line_spacing = 5  # Space between lines
    # Split the text into lines and wrap each line
    max_width = img.shape[1] - 5  # Account for some padding
    wrapped_lines = []
    for line in text.splitlines():
        wrapped_lines.extend(textwrap.wrap(line, width=max_width // 5))  # Estimate characters per line

    num_lines = len(wrapped_lines)
    new_height = img.shape[0] + num_lines * (text_height + line_spacing)

    canvas = np.ones((new_height, img.shape[1], 3), dtype=np.uint8) * 255  
    canvas[:img.shape[0], :img.shape[1]] = img

    # Draw each line of text below the image
    y_offset = img.shape[0] + text_height # Start position below the image
    for line in wrapped_lines:
        cv2.putText(canvas, line, (10, y_offset), font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)
        y_offset += text_height + line_spacing
    return canvas
