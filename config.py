import os

MODEL_DIR = "imagenet"
NUM_PREDICTIONS = 5
label_lookup_path = os.path.join(
  MODEL_DIR, 'imagenet_2012_challenge_label_map_proto.pbtxt')
uid_lookup_path = os.path.join(
  MODEL_DIR, 'imagenet_synset_to_human_label_map.txt')
graph_file = os.path.join(MODEL_DIR, "classify_image_graph_def.pb")

IMAGES_ROOT = "/home/alse/Pictures"

