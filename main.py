import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# === CONFIG ===
IMG_SIZE = 128
INITIAL_MUTATION_SCALE = 0.018
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
    weights = np.random.rand(IMG_SIZE, IMG_SIZE, 3)
    frames = []
    rows_per_group = 4
    weights_list = []
    for i in range(0, IMG_SIZE, rows_per_group):
        mutation_scale = INITIAL_MUTATION_SCALE
        if(i + rows_per_group) > IMG_SIZE:
            rows_per_group = IMG_SIZE - i
        past_score = similarity_score(weights[i:i+rows_per_group], target[i:i+rows_per_group]) * 4
        while True:
            current_score = similarity_score(weights[i:i+rows_per_group], target[i:i+rows_per_group])

            # Try mutations and keep best if better
            for _ in range(5):
                weights[i:i+rows_per_group] = mutate(weights[i:i+rows_per_group], mutation_scale, target[i:i+rows_per_group], current_score)
                current_score = similarity_score(weights[i:i+rows_per_group], target[i:i+rows_per_group])
            mutation_scale = max(mutation_scale * 0.9, MIN_MUTATION_SCALE)

            # print(f"[{step}] Score: {current_score:.5f} | Mutation: {mutation_scale:.4f}")
            if(past_score / 4 <= current_score):
                past_score = current_score
                weights_list.append(weights.copy())
                print(f"score: {current_score}")
            
            if current_score > -0.001:
                print(f"Target {i/rows_per_group} matched well. Moving on!")
                break
    print("gififying!")
    for weight in weights_list:
        frame = save_current(weight)
        frames.append(frame)
    frames[0].save("evolution.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)

main()
