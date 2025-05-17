import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# === CONFIG ===
IMG_SIZE = 256
INITIAL_MUTATION_SCALE = 0.04
MIN_MUTATION_SCALE = 0.001

# === Load and preprocess target ===
def load_target_image(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img) / 255.0

# === Generate mutation ===
def mutate(image, mutation_scale, target, current_score):
    mutation = np.random.randn(*image.shape) * mutation_scale
    mutated = np.clip(image + mutation, 0, 1)
    if similarity_score(mutated, target) > current_score:
        return mutated
    return image

# === Get how similar the image is to the target ===
def similarity_score(candidate, target):
    return -np.mean((candidate - target) ** 2)

# === Display current best image ===
def save_current(image):
    clipped = np.clip(image, 0.0, 1.0)
    img_uint8 = (clipped * 255).astype(np.uint8)
    return Image.fromarray(img_uint8)

# === MAIN ===
def main():
    target = load_target_image("image.png")
    weights = load_target_image("start.jpg")
    mutation_scale = INITIAL_MUTATION_SCALE
    step = 0
    frames = []
    global MIN_MUTATION_SCALE
    while True:
        step += 1
        current_score = similarity_score(weights, target)

        # Try mutations and keep best if better
        for _ in range(1000):
            weights = mutate(weights, mutation_scale, target, current_score)
            mutation_scale = max(mutation_scale * 0.9995, MIN_MUTATION_SCALE)

        # print(f"[{step}] Score: {current_score:.5f} | Mutation: {mutation_scale:.4f}")
        frame = save_current(weights)
        frames.append(frame)

        if(step % 10 == 0):
            print("gif saved!")
            mutation_scale = (INITIAL_MUTATION_SCALE + mutation_scale) / 2
            frames[0].save("evolution.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
        
        if current_score > -0.001:
            print("Target matched well. Stopping.")
            frames[0].save("evolution.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
            break

main()
