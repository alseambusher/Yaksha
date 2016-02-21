import pexif
import tensorflow as tf
import numpy as np
import re
from tensorflow.python.platform import gfile
from config import *


class Lookup(object):
  def __init__(self):
    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    self.node_lookup = node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def main(_):
  # read graph
  with gfile.FastGFile(graph_file, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

  # tensorflow session
  session = tf.Session()
  softmax_tensor = session.graph.get_tensor_by_name("softmax:0")
  lookup = Lookup()

  # iterate through all the images
  # if a folder has a file called .nomedia then ignore the folder and its sub folders

  for root, subFolders, files in os.walk(IMAGES_ROOT):
    if ".nomedia" in files:
      continue

    for _file in files:
      if os.path.splitext(_file)[1] in [".jpg", ".jpeg"]:
        image_data = gfile.FastGFile(os.path.join(root, _file), "rb").read()
        pretictions = session.run(softmax_tensor, {"DecodeJpeg/contents:0": image_data})
        pretictions = np.squeeze(pretictions)

        # take top 5 predictions
        result = []
        for _id in pretictions.argsort()[-NUM_PREDICTIONS:][::-1]:
          result.append(lookup.id_to_string(_id).replace(",", " "))

        result = " ".join(result)
        print os.path.join(root, _file), "=>", result
        img = pexif.JpegFile.fromFile(os.path.join(root, _file))
        img.exif.primary.ImageDescription = result
        img.writeFile(os.path.join(root, _file))

if __name__ == "__main__":
  tf.app.run()
